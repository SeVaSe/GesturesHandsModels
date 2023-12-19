import cv2
import joblib
import numpy as np

# Загрузка модели
model = joblib.load('L_R_gesture_model.pkl')

# Номер веб-камеры
cap = cv2.VideoCapture(0)  # Обычно 0 для встроенной веб-камеры, если у вас есть внешняя, номер может быть другим

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Обработка кадра, аналогично тому, как вы обрабатывали данные для обучения модели
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed_frame = cv2.resize(processed_frame, (224, 224))  # Размер кадра (224x224), может быть другим

    # Преобразование кадра в формат массива numpy
    X = np.array(processed_frame).reshape(1, -1)

    # Предсказание с использованием модели
    prediction = model.predict(X)

    # Вывод сообщения на основе предсказания
    if prediction == 0:
        print("Левый жест")
    else:
        print("Правый жест")

    cv2.imshow('Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
