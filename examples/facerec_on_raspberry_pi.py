# This is a demo of running face recognition on a Raspberry Pi.
# This program will print out the names of anyone it recognizes to the console.

# To run this, you need a Raspberry Pi 2 (or greater) with face_recognition and
# the picamera[array] module installed.
# You can follow this installation instructions to get your RPi set up:
# https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65

import face_recognition
import picamera
import numpy as np

# Get a reference to the Raspberry Pi camera.
# If this fails, make sure you have a camera connected to the RPi and that you
# enabled your camera in raspi-config and rebooted first.
camera = picamera.PiCamera()
camera.resolution = (320, 240)
output = np.empty((240, 320, 3), dtype=np.uint8)

# Load a sample picture and learn how to recognize it.
fnames = 'hongk.jpg,liming.jpeg,woman.jpeg,me.png,yyy.png'.split(',')
print("Loading known face image(s): %s" %fnames)

biden = []
dic = {}
for i, fname in enumerate(fnames):
    img = face_recognition.load_image_file(fname)
    biden.append(face_recognition.face_encodings(img)[0])
    dic[i] = fname
print(dic)


while True:
    print("Capturing image.")
    # Grab a single frame of video from the RPi camera as a numpy array
    camera.capture(output, format="rgb")

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(output)
    print("Found {} faces in image.".format(len(face_locations)))
    face_encodings = face_recognition.face_encodings(output, face_locations)

    # Loop over each face found in the frame to see if it's someone we know.
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(biden, face_encoding)
        print(match)
        name = "<Unknown Person>"
        for i, v in enumerate(match):
            if v:
                name = dic[i]
                break
        print("I see someone named {}!".format(name))
        break
