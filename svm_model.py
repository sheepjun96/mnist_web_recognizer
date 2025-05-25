import os
import numpy as np
from PIL import Image
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from keras.datasets import mnist
import joblib

class MNISTLinearSVM:
    def __init__(self, model_path="svm_model.joblib", scaler_path="scaler.joblib"):
        self.model_path = model_path
        self.scaler_path = scaler_path

        print("\nsvm 모델 로드 중...")
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                print("svm 모델 로드 완료.\n")
            except Exception as e:
                print(f"svm 모델 로드 실패: {e}. \n다시 학습을 진행합니다.")
                self._train_model()
        else:
            self._train_model()

    # 모델 학습
    def _train_model(self):
        print("svm 모델 학습 중...")

        # MNIST 데이터 로드
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_flattened = X_train.reshape((X_train.shape[0], -1))
        X_test_flattened = X_test.reshape((X_test.shape[0], -1))

        # 정규화
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_flattened)
        X_test_scaled = self.scaler.transform(X_test_flattened)

        best_C = 1

        '''
        # Grid Search
        param_grid = {'C': [0,1, 1, 10, 100]}
        grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=2)
        grid_search.fit(X_scaled, y_train)

        print("\nGrid Search 결과")
        for mean, std, params in zip(grid_search.cv_results_['mean_test_score'], 
                                     grid_search.cv_results_['std_test_score'], 
                                     grid_search.cv_results_['params']):
            print(f"   C={params['C']}: 평균 정확도={mean:.4f}, 표준편차={std:.4f}")

        best_C = grid_search.best_params_['C']
        print(f"\n최적의 C 값: {best_C}")
        '''

        # SVM 모델 학습
        self.model = svm.SVC(kernel='linear', C=best_C)
        self.model.fit(X_scaled, y_train)

        accuracy = self.model.score(X_test_scaled, y_test)
        print(f"\nsvm 모델 학습 완료. 정확도: {accuracy:.4f}\n") # 정확도: 92.93%

        # 모델 저장
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    # 예측
    def predict_digit(self, input_array):
        input_scaled = self.scaler.transform([input_array])
        prediction = self.model.predict(input_scaled)
        return int(prediction[0])
