# Cf. https://huggingface.co/learn/audio-course/chapter7/voice-assistant
from transformers import pipeline
import torch
from transformers.pipelines.audio_utils import ffmpeg_microphone_live

device = "cuda:0" if torch.cuda.is_available() else "cpu"

classifier = pipeline(
    "audio-classification", model="MIT/ast-finetuned-speech-commands-v2",
    device=device
)

def launch_fn(
        wake_word="marvin",
        prob_threshold=0.5,
        chunk_length_s=2.0,
        stream_chunk_s=0.25,
        debug=False
):
    if wake_word not in classifier.model.config.label2id.keys():
        raise ValueError(
            f"Wake word '{wake_word}' is not in the set of valid class labels. Pick a wake word in the set {classifier.config.label2id.keys()}."
        )
    sampling_rate = classifier.feature_extractor.sampling_rate

    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s
    )

    print("Listening for wake word...")
    for prediction in classifier(mic):
        prediction = prediction[0]
        if debug:
            print(prediction)
        if prediction["label"] == wake_word:
            if prediction["score"] > prob_threshold:
                return True

launch_fn(debug=True)