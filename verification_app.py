from flask import Flask, request
import base64
import json
import logging
from io import BytesIO
import torch
import torchaudio
import os
import re
from logging.handlers import TimedRotatingFileHandler
from verification import enhance_and_embeding
import torchaudio.transforms as transforms
from pydub import AudioSegment


def setup_log(log_name):
    logger = logging.getLogger(log_name)
    log_path = os.path.join("/home/zxcl/zsm/project/Speaker-Verification/log/", log_name)
    logger.setLevel(logging.INFO)
    file_handler = TimedRotatingFileHandler(
        filename=log_path, when="MIDNIGHT", interval=1, backupCount=30
    )
    file_handler.suffix = "%Y-%m-%d.log"

    file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")

    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(process)d] [%(levelname)s] - %(module)s.%(funcName)s (%(filename)s:%(lineno)d) - %(message)s"
        )
    )
    logger.addHandler(file_handler)
    return logger


logHandler = TimedRotatingFileHandler(filename="/home/zxcl/zsm/project/Speaker-Verification/log/verification.log",
                                      when="midnight",
                                      interval=1, encoding='utf-8')
logHandler.suffix = "%Y-%m-%d_%H-%M-%S"
logFormatter = logging.Formatter('[%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s')
logHandler.setFormatter(logFormatter)
logger = logging.getLogger('MyLogger')
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

app = Flask(__name__)


@app.route("/status", methods=['GET'])
def status():
    rs = json.dumps({"Server_name": "声纹识别", "status": "ok"}, ensure_ascii=False)
    logger.info(rs)
    return rs


@app.route("/verification", methods=["POST"])
def verification():
    if request.files:
        logger.info("It is a file!")
        data = request.files
        file = data['file']
        logger.info("loading audio file!")
        audio_as = AudioSegment.from_file(file).export(format="wav", bitrate="256k").read()
        audio, sr = torchaudio.load(BytesIO(audio_as))
        audio = torch.mean(audio, dim=0, keepdim=True)
        if sr !=16000:
            transform = transforms.Resample(orig_freq=sr, new_freq=16000)
            audio = transform(audio)
        logger.info("embedding!")
        audio_tensor = enhance_and_embeding(noisy=audio).encoder()
        file_name = file.filename
        embedding = {"file_name": file_name, "tensor": audio_tensor.tolist()}
        logger.info(embedding)
        return embedding

    elif request.json:
        logger.info("It is a json!")
        data = request.json
        file_name = data["file_name"]
        audio_bs64 = data["audio"]
        logger.info("loading audio file!")
        audio_bytes = base64.b64decode(audio_bs64)
        audio_as = AudioSegment.from_file(BytesIO(audio_bytes)).export(format="wav", bitrate="256k").read()
        audio, sr = torchaudio.load(BytesIO(audio_as))
        audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != 16000:
            transform = transforms.Resample(orig_freq=sr, new_freq=16000)
            audio = transform(audio)
        logger.info("embedding!")
        audio_tensor = enhance_and_embeding(noisy=audio).encoder()
        embedding = {"file_name": file_name, "tensor": audio_tensor.tolist()}
        logger.info(embedding)
        return embedding


if __name__ == '__main__':
    try:
        audio_as = AudioSegment.from_file("data/noisy/1_1_1.wav").export(format="wav", bitrate="256k").read()
        audio, sr = torchaudio.load(BytesIO(audio_as))
        audio = torch.mean(audio, dim=0, keepdim=True)
        transform = transforms.Resample(orig_freq=sr, new_freq=16000)
        audio = transform(audio)

        audio_tensor = enhance_and_embeding(noisy=audio).encoder()
        print(audio_tensor)
        embedding = {"file_name": "1_1_1.wav", "tensor": audio_tensor}
        logger.info(embedding)
        logger.info("启动成功！")
        status()
        app.run(port=5052, host="0.0.0.0", threaded=True)
    except Exception as e:
        logger.info("启动失败！", e)
