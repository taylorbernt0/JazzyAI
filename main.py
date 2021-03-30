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

def getNotesFromFile(file):
    notes = []

    try:
        midi = converter.parse(file)
    except Exception as x:
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

    for element in piano_part:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch) + '|' + str(float(element.quarterLength)))
        elif isinstance(element, note.Rest):
            notes.append(' ' + '|' + str(float(element.quarterLength)))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder) + '|' + str(float(element.quarterLength)))

    # Remove space at beginning
    if notes[0] == ' ':
        notes = notes[1:]

    return notes

def sortByMusicalScale(notes):
    letters = ['A','A#','B-','B','C','C#','D-','D','D#','E-','E','F','F#','G-','G','G#','A-']
    alphabet = [' ']

    for n in range(10):
        alphabet.extend([l+str(n) for l in letters])

    return sorted(notes, key=lambda word: [alphabet.index(word.split('|')[0]) if word.split('|')[0] in alphabet else -1])

def processNotes(song_files):
    print('Processing songs...')

    allPitchNames = set()

    songs_network_input = []
    songs_network_output = []

    for file in tqdm(song_files):
        #song_filepath = os.path.basename(file)

        notes = getNotesFromFile(file)

        if notes is None or len(notes) < 50:
            continue

        # Add notes to note set
        allPitchNames = allPitchNames.union(set(item for item in notes))

        song_network_input = []
        song_network_output = []

        for i in range(0, len(notes) - sequence_length, 1):
            song_network_input.append(notes[i:i + sequence_length])
            song_network_output.append(notes[i + sequence_length])

        songs_network_input.extend(song_network_input)
        songs_network_output.extend(song_network_output)

    # Sort notes
    allPitchNames = sortByMusicalScale(allPitchNames)
    n_vocab = len(allPitchNames)

    # Get not conversion dictionaries
    note_to_int = dict((note, number) for number, note in enumerate(allPitchNames))
    int_to_note = dict((number, note) for number, note in enumerate(allPitchNames))

    print('Vocab size:', len(allPitchNames))
    print('Vocab-note dictionary:', note_to_int)

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

    return (n_vocab, note_to_int, int_to_note, numpy_songs_network_input, numpy_songs_network_output, songs_network_input[0])

def getPrediction(model, int_to_note, sequence_length, seedData, filename, amountOfNotes=100, createFile=True):
    prediction_output = []
    vocab_length = len(int_to_note)

    pattern = seedData

    if len(pattern) > sequence_length:
        pattern = pattern[:sequence_length]

    print('Generating {0} notes from {1}'.format(amountOfNotes, [int_to_note[n] for n in pattern]))
    for note_index in tqdm(range(amountOfNotes)):
        prediction_input = np.reshape(pattern, (1, sequence_length, 1))
        prediction_input = prediction_input / float(vocab_length)

        # print(prediction_input)
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    print(prediction_output)

    if createFile:
        offset = 0
        output_notes = []
        # create note and chord objects based on the values generated by the model
        for pattern in prediction_output:
            note_pattern, duration_pattern = pattern.split('|')
            duration_pattern = float(duration_pattern)
            # pattern is a chord
            if ('.' in note_pattern) or note_pattern.isdigit():
                notes_in_chord = note_pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                new_chord.quarterLength = duration_pattern
                output_notes.append(new_chord)
            # pattern is a rest
            elif note_pattern == ' ':
                new_rest = note.Rest()
                new_rest.offset = offset
                new_rest.storedInstrument = instrument.Piano()
                new_rest.quarterLength = duration_pattern
                output_notes.append(new_rest)
            # pattern is a note
            else:
                new_note = note.Note(note_pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                new_note.quarterLength = duration_pattern
                output_notes.append(new_note)
            # increase offset each iteration so that notes do not stack
            offset += 0.5

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp=filename)
        print('Midi file created as:', filename)

    return prediction_output


prediction_history = []


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, seedData):
        #super().__init__()
        self.model = model
        self.seedData = seedData

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            prediction = getPrediction(self.model, self.seedData, 'temp.mid', amountOfNotes=100, createFile=True)
            prediction_history.append(prediction)

def trainModel(model, inputs, outputs, epochs, batchSize, seed, finalPredictionLength, song_filepath):
    timestamp = int(time.time())

    if not os.path.isdir('./model_saves'):
        print('Creating model_saves directory')
        os.mkdir('model_saves')

    print('Creating model_saves/{} directory'.format(timestamp))
    os.mkdir('model_saves/{}'.format(timestamp))

    pickle.dump((sequence_length, int_to_note), open('model_saves/{}/pickleData.p'.format(timestamp), 'wb'))

    checkpoint_filepath = "model_saves/" + str(timestamp) + "/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )

    progress_matrix_callback = CustomCallback(model, seed)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    print('Training model...')
    history = model.fit(
        inputs,
        outputs,
        epochs=epochs,
        batch_size=batchSize,
        callbacks=[
            checkpoint,
            early_stopping_callback
            #progress_matrix_callback
        ]
    )

    print(getPrediction(model, int_to_note, sequence_length, seed, song_filepath, amountOfNotes=finalPredictionLength, createFile=True))

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
    notes = getNotesFromFile(song)
    return notes[:sequence_length]

def getPredictionFromSave(timestamp, seed, amoundOfNotes):
    saveFolder = './model_saves/{}/'.format(timestamp)
    if not os.path.isdir(saveFolder):
        print('Could not find {} directory'.format(saveFolder))
        return

    (sequence_length, int_to_note) = pickle.load(open(saveFolder + 'pickleData.p', 'rb'))

    note_to_int = {v: k for k, v in int_to_note.items()}
    n_vocab = len(int_to_note)

    print('Successfully loaded pickled data!')
    print('Sequence length: {}'.format(sequence_length))
    print('Vocab size: {}'.format(n_vocab))

    model = defineModel(n_vocab, sequence_length, timestamp)

    print('Successfully loaded model!')

    seed = seed[:sequence_length]
    translatedSeed = [note_to_int[note] for note in seed]

    print(getPrediction(model, int_to_note, sequence_length, translatedSeed, 'pickle.mid', amountOfNotes=amoundOfNotes, createFile=True))
    print(sequence_length, int_to_note)

#getPredictionFromSave(1617059767, getSeedFromFile('midi_songs/ArtPepper_Anthropology_FINAL.mid'), 300)
#exit()

songs_to_train = 1  # Number of songs to take from the dataset
sequence_length = 25 # Number of reference notes the network uses to generate a prediction note
epochs = 1000
batchSize = 512
finalPredictionLength = 300 # Length of the song produced at the end of training

song_files = getSongs('./midi_classical_songs', numberOfSongs=songs_to_train)

n_vocab, note_to_int, int_to_note, inputs, outputs, seed = processNotes(song_files)
model = defineModel(n_vocab, sequence_length)
trainModel(model, inputs, outputs, epochs, batchSize, seed, finalPredictionLength, os.path.basename(song_files[0]))