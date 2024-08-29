import cv2
import pyaudio
import wave

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, output=True)

alarm_wav_file = 'Tieng-coi-chong-trom-www_tiengdong_com.wav'
with wave.open(alarm_wav_file, 'rb') as wf:
    alarm_data = wf.readframes(wf.getnframes())

alarm_playing = False  
alarm_last_state = False  

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray)

    alarm_needed = False  

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        closed_eyes = sum(1 for (_, _, ew, eh) in eyes if ew * eh < 5000)

        if closed_eyes > 0:
            alarm_needed = True  
            break  

        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]

            if ew * eh < 5000:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
            else:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    if alarm_needed != alarm_last_state:
        if alarm_needed and not alarm_playing:
            stream.write(alarm_data)
            alarm_playing = True
            print("Started alarm sound")
        elif not alarm_needed and alarm_playing:
            stream.stop_stream()
            alarm_playing = False
            print("Stopped alarm sound")
        alarm_last_state = alarm_needed

    cv2.imshow('Camera - Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
stream.stop_stream()
stream.close()
audio.terminate()
