# -*- coding: utf-8 -*-

# To run this, you need a Raspberry Pi 2 (or greater) with face_recognition and
# the picamera[array] module installed.
# You can follow this installation instructions to get your RPi set up:
# https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65

import face_recognition
import numpy as np
import sys
if sys.version_info[0] >= 3:
    raw_input = input


def prepare_camera():
    import picamera
    # If this fails, make sure you have a camera connected to the RPi and that you
    # enabled your camera in raspi-config and rebooted first.
    camera = picamera.PiCamera()
    camera.resolution = (320, 240)
    return camera


def known_faces(path, names):
    # Load a sample picture and learn how to recognize it.
    print("known face images size: %s" % len(names))
    bid_encodings = []
    fnames = []
    for i, fname in enumerate(names):
        # single face image
        print('\tload %d: %s' % (i, fname))
        img = face_recognition.load_image_file('%s/%s' % (path, fname))
        res = face_recognition.face_encodings(img)
        if len(res) < 1:
            continue
        bid_encodings.append(res[0])
        fnames.append(fname)
    return bid_encodings, fnames


def load_cache(roster):
    import os
    import pickle
    res = [], []
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
                if not fname[0].isalpha() or not fname.rsplit('.',
                                                              1)[0].isalnum():
                    continue
                print('download {}'.format(fname))
                urllib.urlretrieve(url, '%s/%s' % (roster, fname))
                f1.write('%s\n' % fname)


def most_like(lst, target):
    min_v, argmin = 1e10, -1
    for i, a in enumerate(lst):
        v = np.linalg.norm(a - target)
        if v < 0.6 and v < min_v:
            min_v, argmin = v, i
    if argmin >= 0:
        print('most_like: %s, v: %s' % (argmin, min_v))
    return argmin


def main(camera, bid_encodings, bid_names):
    print(bid_names)
    output = np.empty((240, 320, 3), dtype=np.uint8)
    while True:
        print("Capturing image.")
        camera.capture(output, format="rgb")
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(output)
        print("Found {} faces in image.".format(len(face_locations)))
        face_encodings = face_recognition.face_encodings(output, face_locations)
        # Loop over each face found in the frame to see if it's someone we know.
        for k, face_encoding in enumerate(face_encodings):
            # See if the face is a match for the known face(s)
            name = 'unkown'
            most = most_like(bid_encodings, face_encoding)
            if most >= 0:
                name = bid_names[most]
            else:
                answer = raw_input('store new person? [y]|n\n')
                if answer != 'n':
                    name = raw_input('name?\n')
                    bid_encodings.append(face_encoding)
                    bid_names.append(name)
            print("I see face {} named {}!".format(k, name))


if __name__ == "__main__":
    url = 'search_bing'
    roster = 'sample'
    roster = 'ewku.com'
    # ?q=%E6%98%8E%E6%98%9F+%E5%A4%B4%E5%83%8F+%E5%BA%93+ewku.com&qs=n&form=QBIRMH&qft=+filterui%3Aface-face+filterui%3Aimagesize-small&sp=-1&pq=%E6%98%8E%E6%98%9F+%E5%A4%B4%E5%83%8F+%E5%BA%93+ewku.com&sc=0-16&sk=&cvid=7842EFF9FA3445579A8FFAE7B25D6131'
    # download_img(url, roster)
    main(prepare_camera(), *load_cache(roster))
