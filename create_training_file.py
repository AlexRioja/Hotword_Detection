import numpy as np
import os
from pydub import AudioSegment
from scipy.io import wavfile
import matplotlib.pyplot as plt

Ty = 1375


def overlaps(piece, previous_pieces):
    segment_start, segment_end = piece
    overlap = False
    for previous_start, previous_end in previous_pieces:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True
    return overlap


def take_a_piece(audio_ms):
    start_piece = np.random.randint(low=0, high=10000 - audio_ms)  # Starts 0-10 but with margin
    end_piece = start_piece + audio_ms - 1
    print("PIECE GOES:"+str(start_piece), str(end_piece))
    return start_piece, end_piece


def insert_into_base_clip(base_clip, audio, covered_zones):
    #print("BASE CLIP LENGTH:"+str(len(base_clip)))
    audio_ms = len(audio)
    piece = take_a_piece(audio_ms)
    while overlaps(piece, covered_zones):
        piece = take_a_piece(audio_ms)
    modded_base_clip = base_clip.overlay(audio, position=piece[0])
    covered_zones.append(piece)
    return modded_base_clip, piece


def get_files(route):
    audio = []
    for root, dirs, files in os.walk(route):
        for file in files:
            if file.endswith("wav"):
                path = os.path.join(root, file)
                # print("Procesando :" + path)
                sound = AudioSegment.from_wav(path)
                audio.append(sound)
    return audio


def mark_positives(y_train, end_time):
    # Ty is the number of steps we want to take (discrete steps in spectrogram)
    positive_occurrence = int(end_time * Ty / 10000.0)
    print("Positive:"+str(positive_occurrence))
    for i in range(positive_occurrence + 1, positive_occurrence + 51):
        if i < Ty:
            y_train[0, i] = 1
    return y_train


def mark_positives_start(y_train, start_time):
    positive_occurrence = int(start_time * Ty / 10000.0)
    print(positive_occurrence)
    for i in range(positive_occurrence + 1, positive_occurrence + 51):
        if i < Ty:
            y_train[0, i] = 1
            #print("poniendo unos")
    return y_train


def graph_spectrogram(wav_file, plot=False):
    rate, data = wavfile.read(wav_file)
    nfft = 200  # Length of each window segment
    fs = 8000  # Sampling frequencies
    noverlap = 120  # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap=noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:, 0], nfft, fs, noverlap=noverlap)
    print(pxx.shape)
    if plot:
        plt.show()
    return pxx


def create_training_wav(base_clip, label):
    positives = get_files('dataset/positive')
    negatives = get_files('dataset/negative')

    base_clip = base_clip - 15

    covered_zones = []
    y_train = np.zeros((1, Ty))

    # We pick random positive and negative files from dataset
    positive_indexes = np.random.randint(len(positives), size=np.random.randint(1, 4))
    negative_indexes = np.random.randint(len(negatives), size=np.random.randint(1, 3))
    rand_positives = [positives[i] for i in positive_indexes]
    rand_negatives = [negatives[i] for i in negative_indexes]

    for rand_pos in rand_positives:
        #print("Inserting positive")
        base_clip, piece = insert_into_base_clip(base_clip, rand_pos, covered_zones)
        #print("Piece ends: "+str(piece[1]))
        y_train = mark_positives(y_train, end_time=piece[1])  # Keeping record of when the positive occurs
        #y_train= mark_positives_start(y_train, piece[0])
    for rand_neg in rand_negatives:
        #print("Inserting negative")
        base_clip, _ = insert_into_base_clip(base_clip, rand_neg, covered_zones)

    base_clip = base_clip.apply_gain(-20.0 - base_clip.dBFS)  # Keepeing dbs in order
    print("Guardando archivo de entreamiento: " + str(label))
    base_clip.export("dataset/training_data/train_" + str(label) + ".wav", format="wav")
    x_train = graph_spectrogram("dataset/training_data/train_" + str(label) + ".wav", False)
    print("SHAPES WHEN PROCESSED DATASET: " + str(x_train.shape), str(y_train.shape))
    print(y_train)
    return x_train, y_train
