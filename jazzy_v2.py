from music21 import converter, instrument, note, chord, stream
import glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
import time
import pickle

#tf.keras.backend.clear_session()
#
#resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
#tf.config.experimental_connect_to_cluster(resolver)
## This is the TPU initialization code that has to be at the beginning.
#tf.tpu.experimental.initialize_tpu_system(resolver)
#print("All devices: ", tf.config.list_logical_devices('TPU'))
#
#strategy = tf.distribute.TPUStrategy(resolver)

class Vocabulary:
    def __init__(self):
        from music21 import note

        on_notes = ['{}_ON'.format(note.pitch.Pitch(n).nameWithOctave) for n in range(128)]
        off_notes = ['{}_OFF'.format(note.pitch.Pitch(n).nameWithOctave) for n in range(128)]
        time_shifts = ['TIME_SHIFT {}'.format(x) for x in range(10, 1010, 10)]
        set_velocities = ['SET_VELOCITY {}'.format(x) for x in range(0, 128, 4)]

        vocabulary = {k: v for v, k in enumerate(on_notes + off_notes + time_shifts + set_velocities)}
        self.size = len(vocabulary)

        self.note_to_int = dict((n, e) for e, n in enumerate(vocabulary))
        self.int_to_note = dict((e, n) for e, n in enumerate(vocabulary))

    def encode_note(self, n):
        return self.note_to_int[n]

    def decode_note(self, e):
        return self.int_to_note[e]


vocab = Vocabulary()


def get_model(sequence_length, load_timestamp=None):
    #with strategy.scope():
    m = tf.keras.Sequential()
    m.add(tf.keras.layers.LSTM(
        256,
        input_shape=(sequence_length, 1),
        return_sequences=True
    ))
    m.add(tf.keras.layers.Dropout(0.3))
    m.add(tf.keras.layers.LSTM(512, return_sequences=True))
    m.add(tf.keras.layers.Dropout(0.3))
    m.add(tf.keras.layers.LSTM(256))
    m.add(tf.keras.layers.Dense(256))
    m.add(tf.keras.layers.Dropout(0.3))
    m.add(tf.keras.layers.Dense(vocab.size))
    m.add(tf.keras.layers.Activation('softmax'))
    m.compile(loss='categorical_crossentropy', optimizer='adam')

    # m = tf.keras.Sequential()
    # m.add(tf.keras.layers.LSTM(
    #     512,
    #     input_shape=(sequence_length, 1),
    #     recurrent_dropout=0.3,
    #     return_sequences=True
    # ))
    # m.add(tf.keras.layers.LSTM(512, return_sequences=True, recurrent_dropout=0.3, ))
    # m.add(tf.keras.layers.LSTM(512))
    # m.add(tf.keras.layers.BatchNormalization())
    # m.add(tf.keras.layers.Dropout(0.3))
    # m.add(tf.keras.layers.Dense(256))
    # m.add(tf.keras.layers.Activation('relu'))
    # m.add(tf.keras.layers.BatchNormalization())
    # m.add(tf.keras.layers.Dropout(0.3))
    # m.add(tf.keras.layers.Dense(n_vocab))
    # m.add(tf.keras.layers.Activation('softmax'))
    # m.compile(loss='categorical_crossentropy', optimizer='adam')

    if load_timestamp is not None:
        saves = glob.glob('./model_saves/{}/*.hdf5'.format(load_timestamp))
        best_save = sorted(saves)[-1]
        best_save_name = os.path.basename(best_save)
        print('Loading best model: model_saves/{0}/{1}'.format(load_timestamp, best_save_name))
        m.load_weights('./model_saves/{0}/{1}'.format(load_timestamp, best_save_name))

    return m

def get_events_from_file(file):
    try:
        midi = converter.parse(file)
    except Exception as x:
        print(x)
        return

    parts = instrument.partitionByInstrument(midi)

    piano_part = None

    if parts is not None:
        for i, element in enumerate(parts.elements):
            if str(element)[21:][:-1] == 'Piano':
                piano_part = parts.parts[i].recurse()
                break

    if piano_part is None:
        piano_part = midi.flat.notesAndRests

    offset_dictionary = {}

    current_velocity = -1

    for element in piano_part:
        offset = element.offset
        if isinstance(element, (note.Note, chord.Chord)):
            if offset not in offset_dictionary:
                offset_dictionary[offset] = []

            if isinstance(element, note.Note):
                note_names = [element.nameWithOctave]
            else:
                note_names = [str(n.nameWithOctave) for n in element.pitches]

            velocity = round((element.volume.velocity + 1) / 4) * 4
            if velocity != current_velocity:
                offset_dictionary[offset].append('SET_VELOCITY {}'.format(velocity))
                current_velocity = velocity

            for noteName in note_names:
                offset_dictionary[offset].append(noteName + '_ON')
                end_offset = offset + element.quarterLength
                if end_offset not in offset_dictionary:
                    offset_dictionary[end_offset] = []
                offset_dictionary[end_offset].append(noteName + '_OFF')

    if 0 not in offset_dictionary:
        offset_dictionary[0] = []

    offset_dictionary = dict(sorted(offset_dictionary.items()))

    #for k in offset_dictionary:
    #   print(k)
    #   print(offset_dictionary[k])

    quarter_ms = 500  # quarter note gets 500 ms

    offset_keys = list(offset_dictionary.keys())
    event_list = []
    for i in range(len(offset_keys)):
        k = offset_keys[i]
        event_list.extend(offset_dictionary[k])
        if i+1 < len(offset_keys):
            k_next = offset_keys[i+1]
            quarter_time = float(k_next - k)
            ms = round(quarter_time * quarter_ms / 10) * 10
            while ms > 1000:
                event_list.append('TIME_SHIFT {}'.format(min(1000, ms)))
                ms -= 1000
            if ms > 0:
                event_list.append('TIME_SHIFT {}'.format(ms))

    print(event_list)

    return event_list


def process_notes(song_files, sequence_length):
    print('Processing songs...')

    songs_network_input = []
    songs_network_output = []

    for file in tqdm(song_files):
        events = get_events_from_file(file)

        if events is None or len(events) < 50:
            continue

        song_network_input = []
        song_network_output = []

        for i in range(0, len(events) - sequence_length, 1):
            song_network_input.append(events[i:i + sequence_length])
            song_network_output.append(events[i + sequence_length])

        songs_network_input.extend(song_network_input)
        songs_network_output.extend(song_network_output)

    songs_network_input = [[vocab.encode_note(n) for n in notes] for notes in songs_network_input]
    songs_network_output = [vocab.encode_note(n) for n in songs_network_output]

    n_patterns = len(songs_network_input)
    print('{} training samples generated'.format(n_patterns))

    # Shuffle inputs and outputs in unison
    temp = list(zip(songs_network_input, songs_network_output))
    random.shuffle(temp)
    songs_network_input, songs_network_output = zip(*temp)

    numpy_songs_network_input = np.reshape(songs_network_input, (n_patterns, sequence_length, 1))
    numpy_songs_network_input = numpy_songs_network_input / float(vocab.size)

    numpy_songs_network_output = tf.keras.utils.to_categorical(songs_network_output, num_classes=vocab.size)

    return numpy_songs_network_input, numpy_songs_network_output, songs_network_input[0]


def get_prediction(model, sequence_length, seed_data, filename, amount_of_notes=100, create_file=True):
    prediction_output = []

    pattern = seed_data

    if len(pattern) > sequence_length:
        pattern = pattern[:sequence_length]

    print('Generating {0} notes from {1}'.format(amount_of_notes, [vocab.decode_note(n) for n in pattern]))
    for _ in tqdm(range(amount_of_notes)):
        prediction_input = np.reshape(pattern, (1, sequence_length, 1))
        prediction_input = prediction_input / float(vocab.size)

        # print(prediction_input)
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = vocab.decode_note(index)
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    print(prediction_output)

    if create_file:
        event_list = prediction_output
        quarter_ms = 500

        decoded_notes = []
        i = 0
        offset = 0
        last_off_note = -1
        current_velocity = 0
        while i < len(event_list):
            event = event_list[i]
            if event.endswith('ON'):  # note start
                note_name = event.split('_')[0]
                elapsed_time = 0
                for j in range(i + 1, len(event_list), 1):
                    if event_list[j].startswith(note_name):  # note end
                        last_off_note = j
                        break
                    elif event_list[j].startswith('TIME_SHIFT'):  # add more time to note
                        elapsed_time += float(event_list[j].split(' ')[1])

                #print(note_name, elapsed_time / quarter_ms, offset / quarter_ms)
                parsed_note = note.Note(note_name)
                parsed_note.offset = offset / quarter_ms
                parsed_note.quarterLength = elapsed_time / quarter_ms
                parsed_note.storedInstrument = instrument.Piano()
                parsed_note.volume.velocity = current_velocity

                decoded_notes.append(parsed_note)
            elif event.startswith('TIME_SHIFT') and i > last_off_note:
                duration = 0
                time_shift_events = 0
                for j in range(i, len(event_list), 1):
                    if event_list[j].startswith('TIME_SHIFT'):
                        time_shift_events += 1
                        duration += float(event_list[j].split(' ')[1])
                    else:
                        break

                #print('rest', duration / quarter_ms, offset / quarter_ms)

                new_rest = note.Rest()
                new_rest.offset = offset / quarter_ms
                new_rest.storedInstrument = instrument.Piano()
                new_rest.quarterLength = duration / quarter_ms
                decoded_notes.append(new_rest)

                offset += duration

                i += time_shift_events

                continue
            elif event.startswith('TIME_SHIFT'):
                offset += float(event_list[i].split(' ')[1])
            elif event.startswith('SET_VELOCITY'):
                current_velocity = float(event_list[i].split(' ')[1])

            i += 1

        print('Creating file...')
        # decoded_notes.insert(0, tempo.MetronomeMark(number=BPM))
        midi_stream = stream.Stream(decoded_notes)
        midi_stream.write('midi', fp=filename)
        print('Midi file created as:', filename)

    return prediction_output


prediction_history = []


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, sequence_length, seed_data):
        #super().__init__()
        self.model = model
        self.sequence_length = sequence_length
        self.seedData = seed_data

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            prediction = get_prediction(self.model, self.sequence_length, self.seedData, 'temp.mid', amount_of_notes=100, create_file=True)
            prediction_history.append(prediction)


def train_model(model, inputs, outputs, epochs, batch_size, seed, sequence_length, final_prediction_length, song_filepath):
    timestamp = int(time.time())

    if not os.path.isdir('./model_saves'):
        print('Creating model_saves directory')
        os.mkdir('model_saves')

    print('Creating model_saves/{} directory'.format(timestamp))
    os.mkdir('model_saves/{}'.format(timestamp))

    pickle.dump(sequence_length, open('model_saves/{}/pickleData.p'.format(timestamp), 'wb'))

    checkpoint_filepath = "model_saves/" + str(timestamp) + "/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min',
        save_freq=500
    )

    progress_matrix_callback = CustomCallback(model, sequence_length, seed)

    print('Training model...')
    history = model.fit(
        inputs,
        outputs,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            checkpoint_callback,
            #progress_matrix_callback
        ]
    )

    print(get_prediction(model, sequence_length, seed, song_filepath, amount_of_notes=final_prediction_length, create_file=True))

    #plt.imshow(prediction_history)
    #plt.colorbar()
    #plt.show()

    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def get_songs(filepath, numberOfSongs=None, sort=True):
    songs = glob.glob("{0}*.mid".format(filepath))
    if sort:
        return sorted(songs)[:numberOfSongs]

    return songs[:numberOfSongs]

def get_seed_from_file(filepath, sequence_length=None):
    print('Extracting seed from {}'.format(filepath))
    song = glob.glob(filepath)[0]
    notes = get_events_from_file(song)
    return notes[:sequence_length]

def get_prediction_from_save(timestamp, seed, amount_of_notes):
    save_folder = './model_saves/{}/'.format(timestamp)
    if not os.path.isdir(save_folder):
        print('Could not find {} directory'.format(save_folder))
        return

    sequence_length = pickle.load(open(save_folder + 'pickleData.p', 'rb'))

    print('Successfully loaded pickled data!')
    print('Sequence length: {}'.format(sequence_length))

    model = get_model(sequence_length, timestamp)

    print('Successfully loaded model!')

    seed = seed[:sequence_length]
    translated_seed = [vocab.encode_note(note) for note in seed]

    print(get_prediction(model, sequence_length, translated_seed, 'pickle.mid', amount_of_notes=amount_of_notes, create_file=True))

# get_prediction_from_save('1617215950', get_seed_from_file('midi_classical_songs/appass_1.mid'), 1000)
# exit()

songs_folder = './midi_classical_songs/'
songs_to_train = 1  # Number of songs to take from the dataset
sequence_length = 25  # Number of reference notes the network uses to generate a prediction note
epochs = 1
batchSize = 512
final_prediction_length = 100  # Length of the song produced at the end of training

song_files = get_songs(songs_folder, numberOfSongs=songs_to_train)

inputs, outputs, seed = process_notes(song_files, sequence_length)
model = get_model(sequence_length)
train_model(model, inputs, outputs, epochs, batchSize, seed, sequence_length, final_prediction_length, os.path.basename(song_files[0]))
