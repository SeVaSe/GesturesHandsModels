import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Функция для чтения и обработки видео
def extract_frames(video_path, label):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Добавляем кадры и соответствующие им метки
        frames.append((frame, label))
    cap.release()
    return frames

# Путь к папке с видео
base_path = 'trainingData'

# Список для хранения кадров и меток
data = []

# Перебираем папки с видео
for folder in ['left_gesture', 'right_gesture']:
    folder_path = os.path.join(base_path, folder)
    label = 0 if folder == 'left_gesture' else 1  # 0 - жест "влево", 1 - жест "вправо"
    for filename in os.listdir(folder_path):
        video_path = os.path.join(folder_path, filename)
        # Извлекаем кадры из видео
        frames = extract_frames(video_path, label)
        data.extend(frames)

# Подготовка данных для обучения
X, y = [], []
for frame, label in data:
    # Преобразование кадров в градации серого и изменение размера до необходимых размеров
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (224, 224))  # Размер кадра (224x224), может быть другим
    X.append(frame)
    y.append(label)

# Преобразование данных в формат массивов numpy
X = np.array(X)
y = np.array(y)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели случайного леса
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train.reshape(len(X_train), -1), y_train)

# Предсказание на тестовых данных
y_pred = model.predict(X_test.reshape(len(X_test), -1))

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy * 100:.2f}%')

# Сохранение модели
if not os.path.exists('L_R_gesture_model.pkl'):
    joblib.dump(model, 'L_R_gesture_model.pkl')
else:
    print("Модель уже существует.")
