import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow import keras
from keras.datasets import mnist
from PIL import Image

class MNISTDeepLearning:
    def __init__(self, model_path="dl_model.keras"):
        self.model_path = model_path
        
        print("\nDL 모델 로드 중...")
        if os.path.exists(self.model_path):
            try:
                self.model = keras.models.load_model(self.model_path)
                print("DL 모델 로드 완료.")
            except Exception as e:
                print(f"DL 모델 로드 실패: {e}. \n다시 학습을 진행합니다.")
                self._train_model()
        else:
            self._train_model()

    # 모델 학습
    def _train_model(self):
        print("DL 모델 학습 중...")

        (X_train, y_train), _ = mnist.load_data()
        X_resized = np.array([
            np.array(Image.fromarray(img).resize((28, 28), Image.LANCZOS))
            for img in X_train
        ])
        X_resized = X_resized / 255.0  # [0,1] 정규화

        self.model = keras.Sequential([
            keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model.fit(X_resized, y_train, epochs=10, batch_size=32, verbose=1)

        loss, acc = self.model.evaluate(X_resized, y_train, verbose=1)
        print(f"\nDL 학습 완료. 정확도: {acc:.4f}") # 정확도: 99.91%

        self.model.save(self.model_path)

    # 예측
    def predict_digit(self, input_array):
        input_array = input_array.reshape((1, 28, 28, 1))
        prediction = self.model.predict(input_array)
        return np.argmax(prediction)
