# Bento
## Deploy niche models on HPE Machine Learning Inference Software (MLIS) 

This repository consists of prebuilt bento files which can be used by anyone for deploying niche models (non-NIMs, non-vLLM friendly models)

See folder 1.1.11 for working bento files & test notebooks to help you test inferencing & build your applications.

I have also included a folder showing the artifacts required to build your own bento images for version 1.1.11 and a notebook showing how to upload the resulting bento file into your local PCAI attached S3 which can be used as a repository in MLIS.

## Current models available

NLLB: Facebook's machine translation model supporting 200 languages including low-resource ones

Chatterbox: ResembleAI's multilingual text-to-speech model with emotion control and watermarking

ASL: Computer vision model for detecting American Sign Language alphabet gestures

Pyannote: Audio processing model for speaker diarization and voice activity detection

Whisper: OpenAI's automatic speech recognition model for transcribing audio to text

