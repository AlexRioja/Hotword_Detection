import numpy as np
import os
from pydub import AudioSegment
from scipy.io import wavfile
import matplotlib.pyplot as plt

def overlaps(piece, previous_pieces):
    segment_start, segment_end = piece
    overlap = False

    for previous_start, previous_end in previous_pieces:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True

    return overlap


def retrieve_piece(time):
    start_piece = np.random.randint(low=0, high=10000 - time)  # Starts 0-10 but with margin
    end_piece = start_piece + time - 1

    return start_piece, end_piece


def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the
    audio segment does not overlap with existing segments.

    Arguments:
    background -- a 10 second background audio recording.
    audio_clip -- the audio clip to be inserted/overlaid.
    previous_segments -- times where audio segments have already been placed

    Returns:
    new_background -- the updated background audio
    """

    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)

    # Step 1: Use one of the helper functions to pick a random time segment onto which to insert
    # the new audio clip. (≈ 1 line)
    piece = retrieve_piece(segment_ms)

    # Step 2: Check if the new segment_time overlaps with one of the previous_segments. If so, keep
    # picking new segment_time at random until it doesn't overlap. (≈ 2 lines)
    while overlaps(piece, previous_segments):
        piece = retrieve_piece(segment_ms)

    # Step 3: Add the new segment_time to the list of previous_segments (≈ 1 line)
    previous_segments.append(piece)

    # Step 4: Superpose audio segment and background
    new_background = background.overlay(audio_clip, position=piece[0])

    return new_background, piece


def insert_ones(y, segment_end_ms, steps=1400):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 followinf labels should be ones.


    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms

    Returns:
    y -- updated labels
    """

    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * steps / 10000.0)

    # Add 1 to the correct index in the background label (y)
    for i in range(segment_end_y + 1, segment_end_y + 51):
        if i < steps:
            y[0, i] = 1

    return y

def graph_spectrogram(wav_file):
    rate, data = wavfile.read(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

def get_files(route):
    audio = []
    for root, dirs, files in os.walk(route):
        for file in files:
            if file.endswith("wav"):
                path = os.path.join(root, file)
                print("Procesando :" + path)
                sound = AudioSegment.from_wav(path)
                audio.append(sound)
    return audio


def create_training_example(noise, steps=1400):
    """
    Creates a training example with a given background, positives, and negatives.

    Arguments:
    background -- a 10 second background audio recording
    positives -- a list of audio segments of the word "activate"
    negatives -- a list of audio segments of random words that are not "activate"

    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """
    positives = get_files('dataset/positive')
    negatives = get_files('dataset/negative')

    # Set the random seed
    np.random.seed(15)

    # Make background quieter
    noise = noise - 20

    # Step 1: Initialize y (label vector) of zeros (≈ 1 line)
    labels = np.zeros((1, steps))

    previous_segments = []

    # Select 0-4 random "activate" audio clips from the entire list of "positives" recordings
    number_of_positives = np.random.randint(0, 5)
    random_indices = np.random.randint(len(positives), size=number_of_positives)
    random_positives = [positives[i] for i in random_indices]

    # Step 3: Loop over randomly selected "activate" clips and insert in background
    for random_activate in random_positives:
        # Insert the audio clip on the background
        noise, segment_time = insert_audio_clip(noise, random_activate, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y"
        y = insert_ones(y, segment_end_ms=segment_end)

    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    # Step 4: Loop over randomly selected negative clips and insert in background
    for random_negative in random_negatives:
        # Insert the audio clip on the background
        noise, _ = insert_audio_clip(noise, random_negative, previous_segments)

    # Standardize the volume of the audio clip
    background = noise.apply_gain(-20.0 - noise.dBFS)

    # Export new training example
    file_handle = noise.export("train" + ".wav", format="wav")
    print("File (train.wav) was saved in your directory.")

    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    x = graph_spectrogram("train.wav")

    return x, y
noises=get_files('dataset/noise')
create_training_example(noises[0])