import pyaudio
from queue import Queue
import numpy as np
import sys
from matplotlib import mlab
import argparse
from keras.models import load_model

# argparse para traer el valor de los parametros de entrada
ap = argparse.ArgumentParser(description="Main process in HotWord detection")
ap.add_argument("-t","--threshold", help="Introduce it to obtain threshold noise levels of the mic", action="store_true")
args = vars(ap.parse_args())

# instantiate PyAudio
p = pyaudio.PyAudio()
print(p.get_default_input_device_info())
# This way audio chunks get notified to our main thread
queue = Queue()
feed_duration=10
fs = 48000  # Sampling rate of the microphone
chunk_duration = 0.5  # In seconds, each read
chunk_samples = int(fs * chunk_duration)  # samples of each read, to extract features from
min_threshold = 850  # Configure with the mic

model = load_model('alex_test_model.h5')

# Callback function of for PyAudio async data input

def callback(in_data, frame_count, time_info, status):
    readed_data = np.frombuffer(in_data, dtype='int16')
    if args['threshold'] is not None and args['threshold']:
        print("Readed noise (4 callibrate threshold): " + str(np.abs(readed_data).mean()))

    if np.abs(readed_data).mean() < min_threshold:
        print('-', end='')
    else:
        print('.', end='')
        queue.put(readed_data)  # Appending the data to the queue
    return in_data, pyaudio.paContinue


# Creating the input audio stream
def open_audio_stream():
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
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    return predictions.reshape(-1)


def has_new_triggerword(predictions, chunk_duration, feed_duration, threshold=0.5):
    """
    Function to detect new trigger word in the latest chunk of input audio.
    It is looking for the rising edge of the predictions data belongs to the
    last/latest chunk.

    Argument:
    predictions -- predicted labels from model
    chunk_duration -- time in second of a chunk
    feed_duration -- time in second of the input to model
    threshold -- threshold for probability above a certain to be considered positive

    Returns:
    True if new trigger word detected in the latest chunk
    """
    predictions = predictions > threshold
    chunk_predictions_samples = int(len(predictions) * chunk_duration / feed_duration)
    chunk_predictions = predictions[-chunk_predictions_samples:]
    level = chunk_predictions[0]
    for pred in chunk_predictions:
        if pred > level:
            return True
        else:
            level = pred
    return False

def extract_features_spectrum(data):
    # The `specgram` method returns 4 objects. They are:
    # - Pxx: the periodogram
    # - freqs: the frequency vector
    # - bins: the centers of the time bins
    # - im: the matplotlib.image.AxesImage instance representing the data in the plot

    # different for different channel number
    if data.ndim == 2:
        return mlab.specgram(data[:, 0], NFFT=1024, Fs=8000, noverlap=900)
    elif data.ndim == 1:
        return mlab.specgram(data, NFFT=1024, Fs=8000, noverlap=900)


stream = open_audio_stream()
stream.start_stream()

try:
    while True:
        data_from_queue = queue.get()
        pxx = extract_features_spectrum(data_from_queue)  # extracting spectogram
        preds = detect_trigger(pxx)
        new_trigger = has_new_triggerword(preds, chunk_duration, feed_duration)
except Exception as e:
    print(str(e))
    quit()
