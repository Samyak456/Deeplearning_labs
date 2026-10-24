# Assignment 10: Audio Classification with Speech-to-Text (STT) and Text-to-Speech (TTS)

## Objective
Build a deep-learning application that classifies audio commands and demonstrates Speech-to-Text (STT) and Text-to-Speech (TTS), evaluating the relevant tasks with suitable metrics.

## Dataset
The notebook uses the Mini Speech Commands dataset, which contains short recordings of spoken commands organised into class folders. Audio is divided into training and validation subsets.

## System Components

### 1. Audio Classification
- Load labelled audio recordings.
- Convert waveforms into spectrograms using the Short-Time Fourier Transform (STFT).
- Visualize a sample spectrogram.
- Train a CNN on the spectrograms to classify audio commands.
- Evaluate the classifier on held-out validation data.

### 2. Speech-to-Text (STT)
- Load a pre-trained Whisper speech-recognition model.
- Transcribe selected audio samples.
- Compare recognized text with reference text and calculate Word Error Rate (WER).

### 3. Text-to-Speech (TTS)
- Load pre-trained SpeechT5 and its HiFi-GAN vocoder.
- Load a compatible speaker embedding.
- Generate speech from text.
- Save the generated waveform as a WAV file and play it in the notebook.

## Workflow
1. Import libraries and select the available compute device.
2. Download and load Mini Speech Commands.
3. Create training and validation datasets.
4. Convert waveforms to spectrograms and visualize an example.
5. Build and train a CNN for audio-command classification.
6. Calculate accuracy, precision, recall, and F1-score on validation data.
7. Load Whisper and transcribe selected audio samples.
8. Normalize reference and predicted text and calculate WER.
9. Load SpeechT5, the vocoder, and a compatible speaker embedding.
10. Generate speech, save it as `generated_speech.wav`, and play it.
11. Display classification metrics, STT WER, and TTS output information.

## Main Technologies
- Python
- TensorFlow / Keras
- PyTorch
- Hugging Face Transformers and Datasets
- Whisper (`openai/whisper-tiny.en`, if configured in the notebook)
- SpeechT5 (`microsoft/speecht5_tts`)
- HiFi-GAN (`microsoft/speecht5_hifigan`)
- Scikit-learn, `jiwer`, `soundfile`, Matplotlib, and IPython audio playback

## Evaluation Metrics
### Audio Classification
- Accuracy
- Precision
- Recall
- F1-score
- Classification report and training/validation accuracy plot, if included

### Speech-to-Text
- **Word Error Rate (WER):** Measures word substitutions, deletions, and insertions relative to reference text. Lower is better.
- If folder names are used as references, WER is a small demonstration estimate rather than a comprehensive speech-recognition benchmark.

### Text-to-Speech
The notebook demonstrates speech generation by saving and playing the waveform. Unless additional objective or human evaluation is implemented, this is a functional demonstration rather than a quantitative TTS quality evaluation.

## Running the Notebook
1. Open the notebook in Google Colab.
2. Enable a GPU runtime if available.
3. Run package installation and dataset setup cells.
4. Ensure `data_dir` points to the directory containing the actual command-class folders, not hidden metadata folders such as `__MACOSX`.
5. Run audio loading, spectrogram generation, CNN training, and classification evaluation cells in order.
6. Run Whisper transcription and WER cells.
7. Run SpeechT5 TTS cells and verify that `generated_speech.wav` is created and can be played.

## Notes and Troubleshooting
- Dataset extraction paths may differ; confirm that `data_dir` exists and contains command-class folders.
- If audio loading reports a RIFF/header error, ensure non-audio metadata folders are not being read as audio.
- The `cmu-arctic-xvectors` dataset may require a compatible format or revision with the installed Hugging Face `datasets` version.
- SpeechT5 needs a valid compatible speaker embedding; an arbitrary zero vector is not an equivalent replacement.
- Record actual metrics from your run rather than assuming results.
