import pyaudio
from queue import Queue
import numpy as np
import sys
from matplotlib import mlab
import argparse
from keras.models import load_model
from threading import Thread
import matplotlib.pyplot as plt

# argparse para traer el valor de los parametros de entrada
from scipy.io import wavfile

ap = argparse.ArgumentParser(description="Main process in HotWord detection")
ap.add_argument("-t", "--threshold", help="Introduce it to obtain threshold noise levels of the mic",
                action="store_true")
args = vars(ap.parse_args())

# instantiate PyAudio
p = pyaudio.PyAudio()
print(p.get_default_input_device_info())
# This way audio chunks get notified to our main thread
queue = Queue()
feed_duration = 10
fs = 44100  # Sampling rate of the microphone
chunk_duration = 0.5  # In seconds, each read
chunk_samples = int(fs * chunk_duration)  # samples of each read, to extract features from
feed_samples = int(fs * feed_duration)
min_threshold = 150  # Configure with the mic
assert feed_duration / chunk_duration == int(feed_duration / chunk_duration)
model = load_model('alex_test_model.h5')

data_from_queue = np.zeros(feed_samples, dtype='int16')


# Callback function of for PyAudio async data input

def callback(in_data, frame_count, time_info, status):
    global data_from_queue, queue, min_threshold
    readed_data = np.frombuffer(in_data, dtype='int16')
    if args['threshold'] is not None and args['threshold']:
        print("Readed noise (4 callibrate threshold): " + str(np.abs(readed_data).mean()))

    if np.abs(readed_data).mean() < min_threshold:
        print('-', end='')
        return in_data, pyaudio.paContinue
    else:
        print('.', end='')
    # queue.put(readed_data)  # Appending the data to the queue
    data_from_queue = np.append(data_from_queue, readed_data)
    if len(data_from_queue) > feed_samples:
        data_from_queue = data_from_queue[-feed_samples:]
        # Process data async by sending a queue.
        queue.put(data_from_queue)
    return in_data, pyaudio.paContinue


# Creating the input audio stream
def open_audio_stream(callback):
    stream = p.open(
        format=pyaudio.paInt16,
        channels=2,
        rate=fs,
        input=True,
        frames_per_buffer=chunk_samples,
        input_device_index=0,
        stream_callback=callback)
    return stream


def detect_trigger(x):
    global model
    x = x.swapaxes(0, 1)
    #print(x.shape)
    x = np.expand_dims(x, axis=0)
    #print(x.shape)
    predictions = model.predict(x)
    return predictions.reshape(-1)


def has_new_triggerword(predictions, chunk_duration, feed_duration, threshold=0.55):
    #print(predictions)
    predictions = predictions > threshold
    #print(predictions)
    chunk_predictions_samples = int(len(predictions) * chunk_duration / feed_duration)
    chunk_predictions = predictions[-chunk_predictions_samples:]
    #print(chunk_predictions)
    level = chunk_predictions[0]
    for pred in chunk_predictions:
        if pred > level:
            return True
        else:
            level = pred
    return False

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

def detect_triggerword(filename):
    plt.subplot(2, 1, 1)
    x = graph_spectrogram(filename)
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)

    #predictions.reshape(-1)
    plt.subplot(2, 1, 2)
    plt.plot(predictions[0, :, 0])
    plt.ylabel('probability')
    plt.show()
    return predictions


def extract_features_spectrum(data):
    # The `specgram` method returns 4 objects. They are:
    # - Pxx: the periodogram
    # - freqs: the frequency vector
    # - bins: the centers of the time bins
    # - im: the matplotlib.image.AxesImage instance representing the data in the plot

    # different for different channel number
    if data.ndim == 2:
        pxx, _, _ = mlab.specgram(data[:, 0], NFFT=200, Fs=8000, noverlap=120)
        return pxx
    elif data.ndim == 1:
        # pxx, freqs, bins, im = plt.specgram(data, 200, 8000, noverlap=120)
        pxx, freqs, bins = mlab.specgram(data, NFFT=200, Fs=8000, noverlap=120)
        #print("TETE: "+str(t.shape))
        #plt.show()
        return pxx

detect_triggerword('dataset/training_data/train_testing3_4.wav')

# stream = open_audio_stream(callback)
# stream.start_stream()
#
# while True:
#     data_from_queue = queue.get()
#     pxx = extract_features_spectrum(data_from_queue)  # extracting spectogram
#     preds = detect_trigger(pxx)
#     new_trigger = has_new_triggerword(preds, chunk_duration, feed_duration)
#     if new_trigger:
#         print("CARACOLA")
#
# stream.stop_stream()
# stream.close()
