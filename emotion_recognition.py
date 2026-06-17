from deepface import DeepFace
import cv2

emotion_emojis = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "surprise": "😲",
    "disgust": "🤢",
    "neutral": "😐"
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        data = result[0]

        dominant = data["dominant_emotion"]
        emotions = data["emotion"]

        emoji = emotion_emojis.get(dominant, "🙂")

        cv2.putText(
            frame,
            f"{dominant.upper()}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Emoji text (may appear as box on some systems)
        cv2.putText(
            frame,
            emoji,
            (300, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        y = 80
        for emotion, score in emotions.items():
            emo = emotion_emojis.get(emotion, "")
            text = f"{emo} {emotion}: {score:.1f}%"

            cv2.putText(
                frame,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            y += 30

    except Exception as e:
        print("Error:", e)

    cv2.imshow("Face Expression Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
