import numpy as np
import streamlit as st
import pydub
import wave
import math

#####################################

# Boost Pedal


def boost(inp_wav, gain, file_type):
    wav_file = wave.open(inp_wav, 'r')
    params = wav_file.getparams()
    n_frames = wav_file.getnframes()
    data = wav_file.readframes(n_frames)
    wav_file.close()

    audio_data = np.frombuffer(data, dtype=np.int16)
    audio_fft = np.fft.fft(audio_data)

    boosted_audio_fft = audio_fft * gain
    boosted_audio = np.fft.ifft(boosted_audio_fft).real.astype(np.int16)

    outfile = wave.open("boost.wav", 'wb')
    outfile.setparams(params)
    outfile.writeframes(boosted_audio)
    outfile.close()

    if file_type != 'wav':
        sound = pydub.AudioSegment.from_wav("boost.wav")
        sound.export("boost." + file_type, format=file_type)

#####################################

#  Distortion Pedal


def distort(inp_wav, gain, file_type):
    wav_file = wave.open(inp_wav, 'r')
    params = wav_file.getparams()
    n_frames = params.nframes
    data = wav_file.readframes(n_frames)
    wav_file.close()

    audio_data = np.frombuffer(data, dtype=np.int16)

    distorted = np.tanh((gain / 1000) * audio_data)

    distorted = (distorted * np.iinfo(np.int16).max).astype(np.int16)  # np.iinfo is the max amplitude

    outfile = wave.open('distorted.wav', 'wb')
    outfile.setparams(params)
    outfile.writeframes(distorted.tobytes())
    outfile.close()

    if file_type != 'wav':
        sound = pydub.AudioSegment.from_wav("distorted.wav")
        sound.export("distorted." + file_type, format=file_type)

#####################################

# Fuzz Pedal


def fuzz(inp_wav, distortion_factor, file_type):
    wav_file = wave.open(inp_wav, 'r')
    params = wav_file.getparams()
    n_frames = params.nframes
    data = wav_file.readframes(n_frames)

    audio_data = np.frombuffer(data, dtype=np.int16)

    normalized_audio = audio_data / 32767.0

    distorted_audio = np.sign(normalized_audio) * (1 - np.exp(-normalized_audio * distortion_factor))

    distorted_audio = np.clip(distorted_audio * 32767.0, -32768, 32767).astype(np.int16)

    fuzzed_frames = distorted_audio.tobytes()

    outfile = wave.open('fuzz.wav', 'wb')
    outfile.setparams(params)
    outfile.writeframes(fuzzed_frames)
    outfile.close()

    if file_type != 'wav':
        sound = pydub.AudioSegment.from_wav("fuzz.wav")
        sound.export("fuzz." + file_type, format=file_type)

#####################################

#  EQ Pedal


def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved=True):

    if sample_width == 1:
        dtype = np.uint8
    elif sample_width == 2:
        dtype = np.int16

    channels = np.fromstring(raw_bytes, dtype=dtype)

    if interleaved:
        channels.shape = (n_frames, n_channels)
        channels = channels.T
    else:
        channels.shape = (n_channels, n_frames)

    return channels


def eq(inp_wav, cut_off_frequency, file_type):
    wav_file = wave.open(inp_wav, 'r')
    sample_rate = wav_file.getframerate()
    sample_width = wav_file.getsampwidth()
    n_channels = wav_file.getnchannels()
    n_frames = wav_file.getnframes()

    data = wav_file.readframes(-1)  # Reading audio frames from file after 1000000 samples
    wav_file.close()

    channels = interpret_wav(data, n_frames, n_channels, sample_width, True)

    freq_ratio = cut_off_frequency / sample_rate

    window_size = int(math.sqrt(0.196201 + freq_ratio ** 2) / freq_ratio)
    cumulative_sum = np.cumsum(np.insert(channels[0], 0, 0))
    filtered = ((cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size).astype(channels.dtype)

    outfile = wave.open('eq.wav', "wb")
    outfile.setparams((1, sample_width, sample_rate, n_frames, 'NONE', 'not compressed'))
    outfile.writeframes(filtered.tobytes('C'))
    outfile.close()

    if file_type != 'wav':
        sound = pydub.AudioSegment.from_wav("eq.wav")
        # C:\\Users\\Simarbir G\\PycharmProjects\\DigitalSignalProcessing\\
        sound.export("eq." + file_type, format=file_type)

#####################################

#  EXPERIMENTAL REVERB IN PROGRESS NOT COMPLETED


def apply_comb_filter(signal, delay_samples, decay):
    output = np.zeros(len(signal))
    buffer = np.zeros(max(delay_samples))
    for i in range(len(signal)):
        for d in range(len(delay_samples)):
            if i - delay_samples[d] >= 0:
                output[i] += buffer[d] * decay[d]
        buffer = np.roll(buffer, 1)
        buffer[0] = signal[i]
    return output


def schroeder_reverb(audio_data, sample_rate):
    comb_delays = [int(sample_rate * d) for d in [0.03, 0.035, 0.04, 0.045]]
    comb_decays = [0.9, 0.8, 0.7, 0.6]

    comb_output = apply_comb_filter(audio_data, comb_delays, comb_decays)

    # Normalize the output
    comb_output /= np.max(np.abs(comb_output))

    return comb_output


def reverb(inp_wav):
    wav_file = wave.open(inp_wav, 'rb')
    sample_rate = wav_file.getframerate()
    sample_width = wav_file.getsampwidth()
    n_frames = wav_file.getnframes()
    data = wav_file.readframes(n_frames)
    audio_data = np.frombuffer(data, dtype=np.int16)
    wav_file.close()

    # Apply Schroeder Reverb effect
    reverbed = schroeder_reverb(audio_data, sample_rate)

    outfile = wave.open('reverb.wav', "wb")
    outfile.setparams((1, sample_width, sample_rate, n_frames, 'NONE', 'not compressed'))
    outfile.writeframes(reverbed.astype(np.int16).tobytes())
    outfile.close()

#####################################

#  Wave Generating ?

#####################################

# GUI


st.title('Morb Signal Processing')
st.subheader('Created by Anant Chary & Simarbir Gill to assist your signal processing needs.'
             '\nYou may use our service like a guitar amp, choosing different pedals to alter your sound!')

wav = st.file_uploader("Input a .wav file to get started", type=["wav"])

if wav is not None:
    ftype = st.selectbox(
        'Choose output file type: ',
        ('wav', 'mp3', 'flac', 'ogg'))

    st.text('Choose a pedal to alter your .wav file!')

    with st.container():
        gain_boost = st.slider('Gain boost', min_value=1.0, max_value=2.0, value=1.0, step=0.05)
        boost_button = st.button("Boost")

        gain_distort = st.slider('Gain Distort', min_value=0.1, max_value=2.0, value=1.0, step=0.01)
        distortion_button = st.button("Distortion")

        cutoff_eq = st.slider('Cutoff Frequency', min_value=400.0, max_value=5000.0, value=1000.0, step=100.0)
        eq_button = st.button("EQ")

        fuzz_factor = st.slider('Fuzz Factor',min_value=0.1, max_value=0.9, value=0.5, step=0.1)
        fuzz_button = st.button("Fuzz")

    if boost_button:
        boost(wav, gain_boost, ftype)

    if distortion_button:
        distort(wav, gain_distort, ftype)

    if eq_button:
        eq(wav, cutoff_eq, ftype)

    if fuzz_button:
        fuzz(wav, fuzz_factor, ftype)
