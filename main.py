import os
import threading

import cv2
import random
import face_recognition
from tkinter import *
from datetime import datetime
import playsound as ps
import pyttsx3 as sx
import numpy as np
import speech_recognition as sr

r = sr.Recognizer()

# === WINDOW INIT ===
window = Tk()
window.title('Superintendent Database')
window.configure(bg='#000000')
window.geometry("1000x800")

# === TTS INIT ===
engine = sx.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty('rate', 210)

# === IMAGE STORAGE ===
normal = PhotoImage(file='Emotions/normal.png')
happy = PhotoImage(file='Emotions/happy.png')
angry = PhotoImage(file='Emotions/angry.png')
sad = PhotoImage(file='Emotions/sad.png')
confused = PhotoImage(file='Emotions/confused.png')
irritated = PhotoImage(file='Emotions/irritated.png')

# === BUTTONS & LABELS ===
superFace = Label(window, image=normal, borderwidth=0, bg='#000000',
                  activebackground='#000000')
superFace.pack(pady=20)

initButton = Button(bg='black', fg='white', text='>Initialize', borderwidth=0, font='Courier 20',
                    activebackground='black', activeforeground='white', command=lambda: response(recordAudio()))
initButton.pack()

audioString = Label(window, height=10, width=150)
audioString.config(bg='black', fg='white', text='', font='Courier 14', borderwidth=0)
audioString.pack()


# === METHODS ===


def recordAudio():
    print("Listening Loop")
    with sr.Microphone() as source:
        audio = r.listen(source)
        voiceData = ''
        try:
            voiceData = r.recognize_google(audio)
        except sr.UnknownValueError:
            print('Didn\'t understand')
        except sr.RequestError:
            print("Service down")
        print("Finished")
        return voiceData


def speak(audioPlay):
    audioString.config(text=f'>Superintendent: {audioPlay}')
    rGen = random.randint(1, 20000000)
    audio_file = 'srAudio' + str(rGen) + '.mp3'
    engine.save_to_file("audio", audio_file)
    engine.say(text=audioPlay)
    engine.runAndWait()
    os.remove(audio_file)


def beep():
    ps.playsound(f'Sounds/beep.wav')


def response(voiceData):
    if 'sad' in voiceData:
        superFace.config(image=sad)
        beep()
        speak("Here is my sad face")

    if 'happy' in voiceData:
        superFace.config(image=happy)
        beep()
        speak("Here is my happy face")

    if 'angry' in voiceData:
        superFace.config(image=angry)
        beep()
        speak("Here is my angry face")

    if 'irritated' in voiceData:
        superFace.config(image=irritated)
        beep()
        speak("Here is my irritated face")

    if 'confused' in voiceData:
        superFace.config(image=confused)
        beep()
        speak("Here is my confused face")

    if 'normal' in voiceData:
        superFace.config(image=normal)
        beep()
        speak("Here is my normal face")

    if "facial recognition" in voiceData:
        beep()
        speak("Initializing facial recognition")
        faceRecInit()

    if "end facial recognition" in voiceData:
        beep()
        speak("Exiting facial recognition")
        quit(faceRecInit())


def init():
    while 1:
        response(recordAudio())


# === FACE REC ===
path = 'Database'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    temp = cv2.imread(f'{path}/{cl}')
    images.append(temp)
    classNames.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        return encodeList


def faceRecInit():
    global path
    global images
    global classNames
    global myList

    encodeListKnown = findEncodings(images)
    speak("Encoding complete")

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesTempFrame = face_recognition.face_locations(imgS)
        encodeTempFrame = face_recognition.face_encodings(imgS, facesTempFrame)

        for encodeFace, faceLoc in zip(encodeTempFrame, facesTempFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace, 0.45)
            faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
            faceDisPrint = (1.0 - np.round(faceDistance, 2)) * 100
            # print(str(faceDisPrint) + "% accurate")
            matchIndex = np.argmin(faceDistance)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                # print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 255, 255), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 0, 0), 1)

        cv2.imshow('Webcam', img)
        cv2.waitKey(1)


# =======================

# ===== MAIN METHOD =====
mainloop()
