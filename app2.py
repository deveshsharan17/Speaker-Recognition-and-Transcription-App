import streamlit as st
import numpy as np
import os
from resemblyzer import VoiceEncoder, preprocess_wav
import whisper
from scipy.io.wavfile import write
from pydub import AudioSegment
import tempfile
from sklearn.metrics.pairwise import cosine_similarity

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Initialize session state for audio storage
if 'audio_data' not in st.session_state:
    st.session_state['audio_data'] = None

@st.cache_resource
def load_reference_embeddings(devesh_folder="./devesh", rakshit_folder="./rakshit"):
    encoder = VoiceEncoder()
    devesh_embeddings = []
    for file_name in os.listdir(devesh_folder):
        if file_name.endswith('.mp3') or file_name.endswith('.wav'):
            file_path = os.path.join(devesh_folder, file_name)
            wav = preprocess_wav(file_path)
            embed = encoder.embed_utterance(wav)
            devesh_embeddings.append(embed)
    devesh_embeddings = np.array(devesh_embeddings)

    rakshit_embeddings = []
    for file_name in os.listdir(rakshit_folder):
        if file_name.endswith('.mp3') or file_name.endswith('.wav'):
            file_path = os.path.join(rakshit_folder, file_name)
            wav = preprocess_wav(file_path)
            embed = encoder.embed_utterance(wav)
            rakshit_embeddings.append(embed)
    rakshit_embeddings = np.array(rakshit_embeddings)
    return devesh_embeddings, rakshit_embeddings

# Initialize speaker embeddings in session state
if 'speaker_embeddings' not in st.session_state:
    devesh_embeddings, rakshit_embeddings = load_reference_embeddings()
    st.session_state['speaker_embeddings'] = {
        'devesh': devesh_embeddings,
        'rakshit': rakshit_embeddings
    }

# Load the Whisper model for transcription
@st.cache_resource
def load_whisper_model():
    model = whisper.load_model("base")  # Options: tiny, small, medium, large
    print("Whisper model loaded.")
    return model

whisper_model = load_whisper_model()

# Function to predict speaker
def predict_speaker(audio_file_path, threshold=0.7):
    """
    Predict speaker from all known speakers in session state

    Parameters:
        audio_file_path (str): Path to the audio file to predict.
        threshold (float): Similarity threshold for classification.

    Returns:
        tuple: (predicted_speaker, scores_dict)
    """
    encoder = VoiceEncoder()
    wav = preprocess_wav(audio_file_path)
    embed = encoder.embed_utterance(wav).reshape(1, -1)
    
    # Compare with all known speakers
    scores = {}
    for speaker, embeddings in st.session_state['speaker_embeddings'].items():
        sims = cosine_similarity(embed, embeddings)
        scores[speaker] = np.max(sims)
    
    # Find the best match
    best_speaker = max(scores.items(), key=lambda x: x[1])
    if best_speaker[1] > threshold:
        return best_speaker[0], scores
    return 'Other', scores

# Streamlit UI
st.title("🎤 Speaker Recognition and Transcription App")

st.write("Adjust the similarity threshold if needed.")
threshold = st.slider("Set similarity threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.01)

# Add refresh button in the top right
if st.button("🔄 Refresh"):
    # Clear all session state
    st.session_state['audio_data'] = None
    st.rerun()

# Add new speaker section
st.write("### Add New Speaker")
new_speaker_name = st.text_input("Enter new speaker name").lower()
new_speaker_audio = st.audio_input("Record reference audio for new speaker")

if new_speaker_name and new_speaker_audio and st.button("Add New Speaker"):
    try:
        # Save and process new speaker's audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(new_speaker_audio.read())
            audio_file_path = tmp.name

        # Convert audio to correct format
        sound = AudioSegment.from_file(audio_file_path)
        sound = sound.set_frame_rate(16000).set_channels(1)
        sound.export(audio_file_path, format="wav")

        # Generate embedding
        encoder = VoiceEncoder()
        wav = preprocess_wav(audio_file_path)
        embed = encoder.embed_utterance(wav)
        
        # Add to session state
        st.session_state['speaker_embeddings'][new_speaker_name] = np.array([embed])
        st.success(f"Added {new_speaker_name} to known speakers!")
        
        # Cleanup
        os.remove(audio_file_path)
    except Exception as e:
        st.error(f"Error adding new speaker: {e}")

st.write("### Current Known Speakers")
st.write(", ".join(st.session_state['speaker_embeddings'].keys()))

st.write("### Record and Analyze Audio")
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
            speaker, scores = predict_speaker(audio_file_path, threshold)
            st.success(f"The speaker in the audio file is: **{speaker}**")
            
            # Display all similarity scores
            for speaker_name, score in scores.items():
                st.write(f"**{speaker_name.title()} similarity score:** {score:.4f}")

            # Transcription for known speakers
            if speaker != "Other":
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