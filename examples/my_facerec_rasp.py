# -*- coding: utf-8 -*-

# To run this, you need a Raspberry Pi 2 (or greater) with face_recognition and
# the picamera[array] module installed.
# You can follow this installation instructions to get your RPi set up:
# https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65

import face_recognition
import numpy as np


def prepare_camera():
    import picamera
    # If this fails, make sure you have a camera connected to the RPi and that you
    # enabled your camera in raspi-config and rebooted first.
    camera = picamera.PiCamera()
    camera.resolution = (320, 240)
    output = np.empty((240, 320, 3), dtype=np.uint8)
    return camera, output


def known_faces(path, names):
    # Load a sample picture and learn how to recognize it.
    print("known face images size: %s" % len(names))
    bid_encodings = []
    fnames = []
    for i, fname in enumerate(names):
        # single face image
        print('\tload %d: %s' % (i, fname))
        img = face_recognition.load_image_file('%s/%s' %(path, fname))
        res = face_recognition.face_encodings(img)
        if len(res) < 1:
            continue
        bid_encodings.append(res[0])
        fnames.append(fname)
    return bid_encodings, fnames


def load_cache(roster):
    import os
    import pickle
    import sys
    res = None
    fpkl = roster + '.pkl'
    found = os.path.isfile(fpkl)
    if not found:
        fname = '%s.roster' % roster
        assert os.path.isfile(fname)
        with open(fname, 'r') as f:
            names = [s.strip() for s in f.readlines()]
            res = known_faces(roster, names)
            with open(fpkl, 'wb') as f:
                pickle.dump(res, f)
    else:
        with open(fpkl, 'rb') as f:
            if sys.version_info[0] >= 3:
                res = pickle.load(f, encoding='iso-8859-1')
            else:
                res = pickle.load(f)
    assert (res)
    return res


def download_img(url, roster):
    import BeautifulSoup
    import urllib
    with open(url) as f:
        txt = f.read()
        soup = BeautifulSoup.BeautifulSoup(txt)
        sps = soup.findAll('div', {'class': 'item'})
        print(len(sps))
        with open('%s.roster' % roster, 'w') as f1:
            for i in sps:
                url = i.a['href']
                fname = url.rsplit('/', 1)[1]
                if not fname[0].isalpha() or not fname.rsplit('.', 1)[0].isalnum():
                    continue
                print('download {}'.format(fname))
                urllib.urlretrieve(url, '%s/%s' % (roster, fname))
                f1.write('%s\n' % fname)


def main(camera, output, bid_encodings, bid_names):
    print(bid_names)
    while True:
        print("Capturing image.")
        # Grab a single frame of video from the RPi camera as a numpy array
        camera.capture(output, format="rgb")
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(output)
        print("Found {} faces in image.".format(len(face_locations)))
        face_encodings = face_recognition.face_encodings(
            output, face_locations)
        # Loop over each face found in the frame to see if it's someone we know.
        for k, face_encoding in enumerate(face_encodings):
            # See if the face is a match for the known face(s)
            res = face_recognition.face_distance(bid_encodings, face_encoding)
            most_like = np.argmin(res)
            if res[most_like] < 0.6:
                name = bid_names[most_like]
                print('most_like: %s, v: %s, name: %s' %(most_like, res[most_like], name))
            else:
                name = 'unkown'
            print("I see face {} named {}!".format(k, name))


if __name__ == "__main__":
    url = 'search_bing'
    roster = 'sample'
    roster = 'ewku.com'
    # ?q=%E6%98%8E%E6%98%9F+%E5%A4%B4%E5%83%8F+%E5%BA%93+ewku.com&qs=n&form=QBIRMH&qft=+filterui%3Aface-face+filterui%3Aimagesize-small&sp=-1&pq=%E6%98%8E%E6%98%9F+%E5%A4%B4%E5%83%8F+%E5%BA%93+ewku.com&sc=0-16&sk=&cvid=7842EFF9FA3445579A8FFAE7B25D6131'
    # download_img(url, roster)
    bids = load_cache(roster)
    cam, output = prepare_camera()
    main(cam, output, *bids)
