from Speaker.speakerlab.bin.infer import embedding
from CMGAN.inference import enhancement
import torchaudio
import torch


class enhance_and_embeding:
    def __init__(self, noisy):
        self.noisy = noisy

    def encoder(self):
        est_audio = enhancement(noisy_path=self.noisy, model_path="./model/ckpt").enhance()
        torch.cuda.empty_cache()
        audio_tensor = embedding(wav_file=est_audio).compute_embedding()
        return torch.tensor(audio_tensor[-1])


# if __name__ == '__main__':
#     audio, _ = torchaudio.load("data/noisy/1_1_1.wav")
#     audio_tensor = enhance_and_embeding(noisy=audio).encoder()
#     print(type(audio_tensor))
#     print(audio_tensor.shape)
#     print(audio_tensor)
