qq55import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# ================= الصوت (اختياري) =================
VOICE_ENABLED = False
try:
    import boto3
    from playsound import playsound

    polly = boto3.client("polly", region_name="us-east-1")
except:
    polly = None


def speak(text):
    if not VOICE_ENABLED or polly is None:
        return
    response = polly.synthesize_speech(Text=text, OutputFormat="mp3", VoiceId="Joanna")
    with open("speech.mp3", "wb") as f:
        f.write(response["AudioStream"].read())
    playsound("speech.mp3")
    os.remove("speech.mp3")


# ================= Camera =================
def get_camera():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Using Camera {i}")
            return cap
    raise Exception("No camera found")


# ================= Mediapipe =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

DATASET = "dataset_en.csv"
MODEL = "model_en.sav"


# ================= مشاهدة البيانات =================
def view_data():
    if not os.path.isfile(DATASET):
        print("❌ No dataset found")
        return
    data = pd.read_csv(DATASET)
    print("\n📊 Dataset info:")
    print(data["label"].value_counts())


# ================= حذف تصنيف =================
def delete_label():
    label = input("Label to delete: ")
    if not os.path.isfile(DATASET):
        print("❌ Dataset not found")
        return
    data = pd.read_csv(DATASET)
    data = data[data["label"] != label]
    data.to_csv(DATASET, index=False)
    print(f"✅ Deleted label: {label}")


# ================= جمع البيانات =================
def collect_data():
    cap = get_camera()
    label = input("Enter sign label: ")

    exists = os.path.isfile(DATASET)
    with open(DATASET, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            header = (
                ["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
            )
            writer.writerow(header)

        count = 0
        print("Press Q to stop collecting")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                xs = [lm.x for lm in hand.landmark]
                ys = [lm.y for lm in hand.landmark]
                writer.writerow([label] + xs + ys)
                count += 1

            cv2.putText(
                frame,
                f"Samples: {count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Collecting Data", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


# ================= تدريب النموذج =================
def train_model():
    if not os.path.isfile(DATASET):
        print("❌ Dataset not found")
        return

    data = pd.read_csv(DATASET)
    if len(data["label"].unique()) < 2:
        print("❌ Need at least 2 labels to train")
        return

    X = data.drop("label", axis=1)
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = SVC(kernel="linear")
    model.fit(X_train, y_train)

    print("🎯 Accuracy:", model.score(X_test, y_test))
    pickle.dump(model, open(MODEL, "wb"))


# ================= التعرف + الكتابة + الصوت الاختياري =================
def recognize():
    if not os.path.isfile(MODEL):
        print("❌ Model not found, train first")
        return

    model = pickle.load(open(MODEL, "rb"))
    cap = get_camera()

    last_prediction = ""
    display_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            xs = [lm.x for lm in hand.landmark]
            ys = [lm.y for lm in hand.landmark]
            features = np.array(xs + ys).reshape(1, -1)

            prediction = model.predict(features)[0]

            if prediction != last_prediction:
                display_text = f"Sign: {prediction}"
                speak(prediction)
                last_prediction = prediction

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # كتابة النص على الشاشة
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        cv2.putText(
            frame, display_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3
        )

        cv2.imshow("Sign Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ================= القائمة الرئيسية =================
while True:
    print(f"""
1 - View dataset
2 - Add data
3 - Delete label
4 - Train model
5 - Recognize (Text)
6 - Toggle Voice (Current: {"ON" if VOICE_ENABLED else "OFF"})
0 - Exit
""")
    choice = input("Choose: ")

    if choice == "1":
        view_data()
    elif choice == "2":
        collect_data()
    elif choice == "3":
        delete_label()
    elif choice == "4":
        train_model()
    elif choice == "5":
        recognize()
    elif choice == "6":
        VOICE_ENABLED = not VOICE_ENABLED
        print("🔊 Voice:", "ON" if VOICE_ENABLED else "OFF")
    elif choice == "0":
        break
import tkinter as tk
from tkinter import messagebox
