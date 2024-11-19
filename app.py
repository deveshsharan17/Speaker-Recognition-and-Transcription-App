import streamlit as st
import numpy as np
import os
from resemblyzer import VoiceEncoder, preprocess_wav
import joblib
import whisper
from scipy.io.wavfile import write
from pydub import AudioSegment
import tempfile
from sklearn.decomposition import PCA

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Initialize session state for audio storage
if 'audio_data' not in st.session_state:
    st.session_state['audio_data'] = None
def load_audio_files(folder_path, label):
    embeddings = []
    labels = []
    encoder = VoiceEncoder()
    
    files_count = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mp3') or file_name.endswith('.wav'):
            files_count += 1
            file_path = os.path.join(folder_path, file_name)
            wav = preprocess_wav(file_path)
            embed = encoder.embed_utterance(wav)
            embeddings.append(embed)
            labels.append(label)
    
    print(f"Loaded {files_count} files from {folder_path}")
    return embeddings, labels
# Load the trained speaker classification model
def prepare_audio_features(devesh_folder="./devesh", rakshit_folder="./rakshit", others_folder="./not_devesh"):
    # Load all three classes of audio files
    devesh_embeddings, devesh_labels = load_audio_files(devesh_folder, label=0)
    rakshit_embeddings, rakshit_labels = load_audio_files(rakshit_folder, label=1)
    others_embeddings, others_labels = load_audio_files(others_folder, label=2)

    # Combine all data
    X = np.vstack((devesh_embeddings, rakshit_embeddings, others_embeddings))
    
    return X

@st.cache_resource
def load_speaker_model(model_path="./speaker_classifier_model.pkl"):
    clf = joblib.load(model_path)
    n_components = 8  # You can adjust this number
    pca = PCA(n_components=n_components)
    pca=pca.fit(prepare_audio_features())
    print(f"Model loaded from {model_path}")
    return clf,pca

clf,pca = load_speaker_model()

# Load the Whisper model for transcription
@st.cache_resource
def load_whisper_model():
    model = whisper.load_model("base")  # Options: tiny, small, medium, large
    print("Whisper model loaded.")
    return model

whisper_model = load_whisper_model()

# Function to predict speaker
def predict_speaker(audio_file_path):
    """
    Predict if the speaker in the audio file is Devesh (0), Rakshit (1), or Other (2).

    Parameters:
        audio_file_path (str): Path to the audio file to predict.

    Returns:
        int: 0 if Devesh, 1 if Rakshit, 2 if Other
    """
    encoder = VoiceEncoder()
    wav = preprocess_wav(audio_file_path)
    embed = encoder.embed_utterance(wav)
    embed = embed.reshape(1, -1)
    embed_reduced = pca.transform(embed)
    prediction = clf.predict(embed_reduced)[0]
    return prediction

# Streamlit UI
st.title("🎤 Speaker Recognition and Transcription App")

# Add refresh button in the top right
if st.button("🔄 Refresh"):
    # Clear all session state
    st.session_state['audio_data'] = None
    st.rerun()

st.write("Click the **Record Audio** button below to start recording. Click **Stop** when you're done.")

# Audio Recorder using st.audio_input
audio_file = st.audio_input("Record Audio")

if audio_file is not None:
    # Display the audio player
    st.audio(audio_file, format='audio/wav')

    # Store the audio data in session state
    st.session_state['audio_data'] = audio_file

# Button to process the recorded audio
if st.button("Process Recording"):
    audio_data = st.session_state['audio_data']
    if audio_data is None:
        st.error("No audio data received. Please record audio first.")
    else:
        st.info("Processing audio...")

        try:
            # Save audio to a temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_data.read())
                audio_file_path = tmp.name

            # Ensure the audio is in the correct format (16kHz, mono)
            sound = AudioSegment.from_file(audio_file_path)
            sound = sound.set_frame_rate(16000).set_channels(1)
            sound.export(audio_file_path, format="wav")

            # Speaker prediction
            prediction = predict_speaker(audio_file_path)
            speaker_map = {0: "Devesh", 1: "Rakshit", 2: "Other"}
            speaker = speaker_map[prediction]
            st.success(f"The speaker in the audio file is: **{speaker}**")

            # Transcription for Devesh and Rakshit
            if speaker in ["Devesh", "Rakshit"]:
                result = whisper_model.transcribe(audio_file_path)
                transcription = result["text"]
                st.write("**Transcription:**")
                st.write(transcription)
            else:
                st.write("Speaker is not recognized. Transcription is not available.")

            # Optionally, provide a download link for the recorded audio
            with open(audio_file_path, "rb") as f:
                audio_bytes = f.read()
                st.download_button(
                    label="Download Recorded Audio",
                    data=audio_bytes,
                    file_name="recorded_audio.wav",
                    mime="audio/wav"
                )

            # Clean up the temporary file
            os.remove(audio_file_path)

            # Clear the session state after processing
            st.session_state['audio_data'] = None

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

# Optional: Display recorded audio
if st.session_state['audio_data'] is not None:
    st.write("### Recorded Audio")
    st.audio(st.session_state['audio_data'], format="audio/wav")