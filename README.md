# Sign Language to Text Translator

## Description

Sign languages are natural languages, with their grammar and lexicon, expressed using visual-manual modality. Out of more than 150 sign languages worldwide, ASL is the most widely studied. The task of Sign Language translation is ongoing research.

This project focuses on Fingerspelling component of ASL.

There are three different startegies for Sign Language Translation:

- using specialized hand-tracking tools
- using depth map
- using Computer Vision

The main objective of this project is to develop an AI system capable of translating Sign Language without requiring any specialized hardware.

Training dataset has been taken from [Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet).

Unobserved signers dataset has been created using [this YouTube video](https://youtu.be/6_gXiBe9y9A).

## Tech Stack and concepts used

- Python
- Keras
- OpenCV
- Convolution Neural Network
- Transfer Learning
- Ensembling
- Bootstrap Aggregation

## Setup

- Download the trained model from [here](https://github.com/aniketsharma00411/sign-language-to-text-translator/tree/main/models).
- Download the [live_translate.py](https://github.com/aniketsharma00411/sign-language-to-text-translator/blob/main/live_translate.py) script to translate using Webcam or the [video_translate.py](https://github.com/aniketsharma00411/sign-language-to-text-translator/blob/main/video_translate.py) script to translate a recorded video.
- Run the script and choose the model to use to translate.

[This](https://youtu.be/TE6mQuVlylU) video demonstrates translation using Webcam.

## Results

On [observed signers](https://www.kaggle.com/grassknoted/asl-alphabet)

| Model                                        | Accuracy | Precision | Recall | F-Score |
| -------------------------------------------- | -------- | --------- | ------ | ------- |
| Basic CNN Model                              | 95.71%   | 0.958     | 0.957  | 0.957   |
| Transfer Learning CNN                        | 98.12%   | 0.982     | 0.981  | 0.981   |
| Basic CNN Model with Data Augmentation       | 95.72%   | 0.961     | 0.957  | 0.957   |
| Transfer Learning CNN with Data Augmentation | 94.95%   | 0.951     | 0.949  | 0.949   |
| Ensemble Model                               | 99.99%   | 0.999     | 0.999  | 0.999   |

On [unobserved signers](https://github.com/aniketsharma00411/sign-language-to-text-translator/tree/main/asl_alphabets)

| Model                                        | Accuracy | Precision | Recall | F-Score |
| -------------------------------------------- | -------- | --------- | ------ | ------- |
| Basic CNN Model                              | 37.02%   | 0.230     | 0.344  | 0.257   |
| Transfer Learning CNN                        | 42.31%   | 0.323     | 0.407  | 0.315   |
| Basic CNN Model with Data Augmentation       | 36.78%   | 0.259     | 0.368  | 0.269   |
| Transfer Learning CNN with Data Augmentation | 43.39%   | 0.422     | 0.434  | 0.380   |
| Ensemble Model                               | 44.11%   | 0.365     | 0.441  | 0.353   |
