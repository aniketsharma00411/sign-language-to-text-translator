"""
To Run:
python live_translate.py
"""

import cv2
import numpy as np
from scipy.stats import mode
import tensorflow as tf


CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
              'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing', 'space']


def prepare(frame, model_name):
    # making the frame square by cropping
    orig_width, orig_height, channel = frame.shape
    req_dim = min(orig_width, orig_height)
    frame = frame[orig_width//2-req_dim//2:orig_width//2+req_dim //
                  2, orig_height//2-req_dim//2:orig_height//2+req_dim//2]

    if model_name == 'basic' or model_name == 'basic_augmented' or model_name == 'ensemble':
        frame = cv2.resize(frame, (224, 224))/255
    if model_name == 'efficientnet_b0':
        frame = tf.keras.applications.efficientnet.preprocess_input(
            cv2.resize(frame, (224, 224))
        )
    if model_name == 'mobilenet':
        frame = tf.keras.applications.mobilenet.preprocess_input(
            cv2.resize(frame, (224, 224))
        )

    return frame.reshape(-1, 224, 224, 3)


def predict(model, frame, model_name):
    frame = prepare(frame, model_name)
    if model_name == 'ensemble':
        prediction = int(mode([np.argmax(m.predict([frame])[0])
                               for m in model])[0][0])
    else:
        prediction = int(np.argmax(model.predict([frame])[0]))

    return prediction


def main():
    models = ['basic',
              'basic_augmented',
              'efficientnet_b0',
              'mobilenet',
              'ensemble']

    choose_model = int(input(
        'Choose model to use:\n1. Basic\n2. Basic trained with Augmented data\n3. Transfer Learning\n4. Transfer Learning trained with Augmented data\n5. Ensemble model\nEnter: '))-1

    if choose_model == 4:
        model = [tf.keras.models.load_model(
            f'models/asl_basic_ensemble_{i}.h5') for i in range(5)]
    else:
        model = tf.keras.models.load_model(
            f'models/asl_{models[choose_model]}.h5')

    cap = cv2.VideoCapture(0)

    try:
        while True:
            _, frame = cap.read()
            prediction = predict(model, frame, models[choose_model])
            final = (CATEGORIES[prediction])

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, final, (50, 50), font,
                        1, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('Input', frame)

            c = cv2.waitKey(1)
            if c == 27:  # hit esc key to stop
                break
    except:
        pass

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
