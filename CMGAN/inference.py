import numpy as np
import io
import os
import torchaudio
import soundfile as sf
from pydub import AudioSegment
import math
from tqdm import tqdm
import sys

try:
    from models import generator
    from utils import *
    from tools.compute_metrics import compute_metrics
except:
    sys.path.append('%s/' % os.path.dirname(__file__))
    from models import generator
    from utils import *
    from tools.compute_metrics import compute_metrics

n_fft = 400
cut_len = 16000 * 16
hop = 100


class enhancement:
    def __init__(self, noisy_path, model_path='../../model/ckpt', save_tracks=False,
                 save_dir="./data/clean"):
        self.model_path = model_path
        self.noisy_dir = noisy_path
        self.save_tracks = save_tracks
        self.save_dir = save_dir

        self.model = generator.TSCNet(num_channel=64, num_features=n_fft // 2 + 1).cuda()
        self.model.load_state_dict((torch.load(model_path)))
        self.model.eval()

    @torch.no_grad()
    def enhance_one_tensor(self, model, audio_tensor):
        with torch.no_grad():
            noisy = audio_tensor.cuda()
            c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
            noisy = torch.transpose(noisy, 0, 1)
            noisy = torch.transpose(noisy * c, 0, 1)

            length = noisy.size(-1)
            frame_num = int(np.ceil(length / 100))
            padded_len = frame_num * 100
            padding_len = padded_len - length
            noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)

            if padded_len > cut_len:
                batch_size = int(np.ceil(padded_len / cut_len))
                while 100 % batch_size != 0:
                    batch_size += 1
                noisy = torch.reshape(noisy, (batch_size, -1))

            noisy_spec = torch.stft(noisy, n_fft, hop, window=torch.hamming_window(n_fft).cuda(), onesided=True,
                                    return_complex=False)

            torch.cuda.empty_cache()
            noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
            est_real, est_imag = model(noisy_spec)
            est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

            est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
            est_audio = torch.istft(
                est_spec_uncompress,
                n_fft,
                hop,
                window=torch.hamming_window(n_fft).cuda(),
                onesided=True
            )
            est_audio = est_audio / c
            est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
            return est_audio

    def enhance_split(self, song, lenth, model):
        n = math.ceil(lenth / 5)
        h = 0
        audio_list = []
        for j in range(0, n):
            m = 5 * 1000
            if j == n - 1:
                obj = song[h:]
            else:
                obj = song[h:h + m]
            AS = obj.export(format="wav").read()
            audio, sr = torchaudio.load(io.BytesIO(AS))
            est_audio = self.enhance_one_tensor(model, audio)
            audio_list.append(est_audio)
            h = h + m

        audio_tmp = torch.tensor(audio_list[0])
        for i in range(1, n):
            audio_tmp = torch.cat([audio_tmp, torch.tensor(audio_list[i])], 0)
        return audio_tmp

    def enhance(self):
        model = self.model
        noisy_dir = self.noisy_dir
        if type(noisy_dir) == torch.Tensor:
            audio_bytes = noisy_dir.numpy().tobytes()
            audio_segment = AudioSegment(audio_bytes, frame_rate=16000, channels=1,
                                         sample_width=noisy_dir.numpy().dtype.itemsize)
            lenth = math.ceil(audio_segment.duration_seconds)
            if lenth <= 5:
                est_audio = self.enhance_one_tensor(model, noisy_dir)
            else:
                est_audio = self.enhance_split(audio_segment, lenth, model)
            return torch.tensor(est_audio)
        if os.path.isdir(noisy_dir):
            est_audio_list = []
            for i in tqdm(os.listdir(noisy_dir)):
                song = AudioSegment.from_wav(os.path.join(noisy_dir, i))
                lenth = math.ceil(song.duration_seconds)
                if lenth > 5:
                    audio_tmp = self.enhance_split(song, lenth, model)
                    est_audio_list.append(torch.tensor(audio_tmp))
                    if self.save_tracks:
                        sf.write(os.path.join(self.save_dir, i), audio_tmp, samplerate=16000, format="wav")
                else:
                    audio, sr = torchaudio.load(os.path.join(noisy_dir, i))
                    est_audio = self.enhance_one_tensor(model, audio)
                    est_audio_list.append(torch.tensor(est_audio))
                    if self.save_tracks:
                        sf.write(os.path.join(self.save_dir, i), est_audio, samplerate=16000, format="wav")
            return est_audio_list

        elif os.path.isfile(noisy_dir):
            name = os.path.split(noisy_dir)[-1]
            song = AudioSegment.from_wav(noisy_dir)
            lenth = math.ceil(song.duration_seconds)
            if lenth > 5:
                audio_tmp = self.enhance_split(song, lenth, model)

                if self.save_tracks:
                    sf.write(os.path.join(self.save_dir, name), audio_tmp, samplerate=16000, format="wav")
                return torch.tensor(audio_tmp)
            else:
                audio, sr = torchaudio.load(noisy_dir)
                est_audio = self.enhance_one_tensor(model, audio)
                if self.save_tracks:
                    sf.write(os.path.join(self.save_dir, name), est_audio, samplerate=16000, format="wav")
                return torch.tensor(est_audio)


# if __name__ == "__main__":
    # est_audio = enhancement(noisy_path="../data/noisy", model_path="../model/ckpt", save_tracks=True).enhance()
    # print(len(est_audio))
    # est_audio = enhancement(noisy_path="../data/noisy/chenruoran_1_1.wav", model_path="../model/ckpt", save_tracks=False).enhance()
    # print(est_audio)
    # audio, sr = torchaudio.load("../data/noisy/huangxiaoliang_3_1.wav")
    # est_audio = enhancement(noisy_path=audio, model_path="../model/ckpt", save_tracks=False).enhance()
    # print(est_audio)
