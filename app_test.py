import os
import requests
import base64


src_dir = r'data/ori'


def test_files():
    for root, dirs, files in os.walk(src_dir, topdown=True):
        for name in files:
            file = os.path.join(root, name)
            files = {"file_name": name.encode("utf-8"), "file": open(file, 'rb')}
            r = requests.post("http://192.168.8.45:5052/verification", files=files)
            print(r.json())


def test_json():
    for root, dirs, files in os.walk(src_dir, topdown=True):
        for name in files:
            file = os.path.join(root, name)
            base64_audio = base64.b64encode(open(file, "rb").read()).decode('utf8')
            data = {"file_name": name, "audio": base64_audio }
            r = requests.post("http://192.168.8.45:5052/verification", json=data)
            print(r.json())


if __name__ == '__main__':
    test_files()
    test_json()