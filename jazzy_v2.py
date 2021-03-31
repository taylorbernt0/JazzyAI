from music21 import converter, instrument, note, chord, stream
import glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
import sys
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
#strategy = tf.distribute.experimental.TPUStrategy(resolver)

on_notes = ['{}_ON'.format(note.pitch.Pitch(n).nameWithOctave) for n in range(128)]
off_notes = ['{}_OFF'.format(note.pitch.Pitch(n).nameWithOctave) for n in range(128)]
time_shifts = ['TIME_SHIFT {}'.format(x) for x in range(10,1010,10)]

vocabulary = {k: v for v, k in enumerate(on_notes + off_notes + time_shifts)}
n_vocab = len(vocabulary)

note_to_int = dict((note, number) for number, note in enumerate(vocabulary))
int_to_note = dict((number, note) for number, note in enumerate(vocabulary))

def defineModel(n_vocab, sequence_length, loadTimestamp=None):
    #with strategy.scope():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(
        256,
        input_shape=(sequence_length, 1),
        return_sequences=True
    ))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(512, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(256))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(n_vocab))
    model.add(tf.keras.layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.LSTM(
    #     512,
    #     input_shape=(sequence_length, 1),
    #     recurrent_dropout=0.3,
    #     return_sequences=True
    # ))
    # model.add(tf.keras.layers.LSTM(512, return_sequences=True, recurrent_dropout=0.3, ))
    # model.add(tf.keras.layers.LSTM(512))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Dropout(0.3))
    # model.add(tf.keras.layers.Dense(256))
    # model.add(tf.keras.layers.Activation('relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Dropout(0.3))
    # model.add(tf.keras.layers.Dense(n_vocab))
    # model.add(tf.keras.layers.Activation('softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam')

    if loadTimestamp is not None:
        saves = glob.glob('./model_saves/{}/*.hdf5'.format(loadTimestamp))
        bestSave = sorted(saves)[-1]
        bestSaveName = os.path.basename(bestSave)
        print('Loading best model: model_saves/{0}/{1}'.format(loadTimestamp, bestSaveName))
        model.load_weights('./model_saves/{0}/{1}'.format(loadTimestamp, bestSaveName))

    return model

def getEventsFromFile(file):
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

    for element in piano_part:
        offset = element.offset
        if isinstance(element, (note.Note, chord.Chord)):
            if offset not in offset_dictionary:
                offset_dictionary[offset] = []

            if isinstance(element, note.Note):
                noteNames = [element.nameWithOctave]
            else:
                noteNames = [str(n.nameWithOctave) for n in element.pitches]

            for noteName in noteNames:
                offset_dictionary[offset].append(noteName + '_ON')
                end_offset = offset + element.quarterLength
                if end_offset not in offset_dictionary:
                    offset_dictionary[end_offset] = []
                offset_dictionary[end_offset].append(noteName + '_OFF')

    if 0 not in offset_dictionary:
        offset_dictionary[0] = []

    offset_dictionary = dict(sorted(offset_dictionary.items()))

    #for k in offset_dictionary:
    #    print(k)
    #    print(offset_dictionary[k])

    quarter_ms = 500 # quarter note gets 500 ms

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

    return event_list

def processNotes(song_files, sequence_length):
    print('Processing songs...')

    songs_network_input = []
    songs_network_output = []

    for file in tqdm(song_files):
        events = getEventsFromFile(file)

        if events is None or len(events) < 50:
            continue

        song_network_input = []
        song_network_output = []

        for i in range(0, len(events) - sequence_length, 1):
            song_network_input.append(events[i:i + sequence_length])
            song_network_output.append(events[i + sequence_length])

        songs_network_input.extend(song_network_input)
        songs_network_output.extend(song_network_output)

    songs_network_input = [[note_to_int[note] for note in notes] for notes in songs_network_input]
    songs_network_output = [note_to_int[notes] for notes in songs_network_output]

    n_patterns = len(songs_network_input)
    print('{} training samples generated'.format(n_patterns))

    # Shuffle inputs and outputs in unison
    temp = list(zip(songs_network_input, songs_network_output))
    random.shuffle(temp)
    songs_network_input, songs_network_output = zip(*temp)

    numpy_songs_network_input = np.reshape(songs_network_input, (n_patterns, sequence_length, 1))
    numpy_songs_network_input = numpy_songs_network_input / float(n_vocab)

    numpy_songs_network_output = tf.keras.utils.to_categorical(songs_network_output, num_classes=n_vocab)

    return (numpy_songs_network_input, numpy_songs_network_output, songs_network_input[0])

def getPrediction(model, sequence_length, seedData, filename, amountOfNotes=100, createFile=True):
    prediction_output = []

    pattern = seedData

    if len(pattern) > sequence_length:
        pattern = pattern[:sequence_length]

    print('Generating {0} notes from {1}'.format(amountOfNotes, [int_to_note[n] for n in pattern]))
    for note_index in tqdm(range(amountOfNotes)):
        prediction_input = np.reshape(pattern, (1, sequence_length, 1))
        prediction_input = prediction_input / float(n_vocab)

        # print(prediction_input)
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    print(prediction_output)

    if createFile:
        event_list = prediction_output
        quarter_ms = 500

        decoded_notes = []
        i = 0
        offset = 0
        last_off_note = -1
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

            i += 1

        print('Creating file...')
        # decoded_notes.insert(0, tempo.MetronomeMark(number=BPM))
        midi_stream = stream.Stream(decoded_notes)
        midi_stream.write('midi', fp=filename)
        print('Midi file created as:', filename)

    return prediction_output


prediction_history = []


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, sequence_length, seedData):
        #super().__init__()
        self.model = model
        self.sequence_length = sequence_length
        self.seedData = seedData

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            prediction = getPrediction(self.model, self.sequence_length, self.seedData, 'temp.mid', amountOfNotes=100, createFile=True)
            prediction_history.append(prediction)

def trainModel(model, inputs, outputs, epochs, batchSize, seed, sequence_length, finalPredictionLength, song_filepath):
    timestamp = int(time.time())

    if not os.path.isdir('./model_saves'):
        print('Creating model_saves directory')
        os.mkdir('model_saves')

    print('Creating model_saves/{} directory'.format(timestamp))
    os.mkdir('model_saves/{}'.format(timestamp))

    checkpoint_filepath = "model_saves/" + str(timestamp) + "/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min',
        save_freq=10
    )

    progress_matrix_callback = CustomCallback(model, sequence_length, seed)

    print('Training model...')
    history = model.fit(
        inputs,
        outputs,
        epochs=epochs,
        batch_size=batchSize,
        callbacks=[
            checkpoint_callback,
            #progress_matrix_callback
        ]
    )

    print(getPrediction(model, sequence_length, seed, song_filepath, amountOfNotes=finalPredictionLength, createFile=True))

    #plt.imshow(prediction_history)
    #plt.colorbar()
    #plt.show()

    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def getSongs(filepath, numberOfSongs=None, sort=True):
    songs = glob.glob("{0}/*.mid".format(filepath))
    if sort:
        return sorted(songs)[:numberOfSongs]

    return songs[:numberOfSongs]

def getSeedFromFile(filepath, sequence_length=None):
    print('Extracting seed from {}'.format(filepath))
    song = glob.glob(filepath)[0]
    notes = getEventsFromFile(song)
    return notes[:sequence_length]

def getPredictionFromSave(timestamp, seed, amoundOfNotes):
    saveFolder = './model_saves/{}/'.format(timestamp)
    if not os.path.isdir(saveFolder):
        print('Could not find {} directory'.format(saveFolder))
        return

    model = defineModel(n_vocab, sequence_length, timestamp)

    print('Successfully loaded model!')

    seed = seed[:sequence_length]
    translatedSeed = [note_to_int[note] for note in seed]

    print(getPrediction(model, sequence_length, translatedSeed, 'pickle.mid', amountOfNotes=amoundOfNotes, createFile=True))
    print(sequence_length, int_to_note)

#getPredictionFromSave('1617067412_small_classical_2000', getSeedFromFile('midi_classical_songs/appass_1.mid'), 1000)
#exit()

songs_to_train = 1  # Number of songs to take from the dataset
sequence_length = 25 # Number of reference notes the network uses to generate a prediction note
epochs = 1
batchSize = 512
finalPredictionLength = 300 # Length of the song produced at the end of training

song_files = getSongs('./midi_classical_songs', numberOfSongs=songs_to_train)

inputs, outputs, seed = processNotes(song_files, sequence_length)
model = defineModel(n_vocab, sequence_length)
trainModel(model, inputs, outputs, epochs, batchSize, seed, sequence_length, finalPredictionLength, os.path.basename(song_files[0]))