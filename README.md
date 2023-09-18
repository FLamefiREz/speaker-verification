# <div align="center">声纹识别文档</div>
钟顺民 2023-09-18

<div align="center">
这是一个声纹识别功能的项目。

我先后测试了SpeechBrain、PaddleSpeech和阿里的声纹识别模型，最终选用阿里3dSpeaker开源的声纹识别模型——ERes2Net。

但在测试的过程发现，背景噪音对声纹识别有巨大影响。

因此又测试并使用了SpeechBrain集成的声音增强（环境降噪）模型和CMGAN来对音频去背景音。

CMGAN效果实测优于SpeechBrain提供的两个语音增强模型。
</div>

## 添加仓库

```
git clone https://git.webxtx.com/risk-manager/risk-manager-ai/speaker-verification.git
cd speaker-verification
```

## 获取声纹特征向量
修改对应`main`函数中的变量即可
```python
python verification.py
```

## 音频降噪模块
```python
python CMGAN/inference.py
```

## 声纹识别服务
```python
python verification_app.py
```
服务端口`5052`，也可以使用脚本启动，日志见`log/`。

服务接收参数可以是`{"file_name":"文件名","file":文件字节}`的形式，

也可以是`{"file_name":"文件名","audio":文件base64}`的形式。

返回参数：`{"file_name":"文件名","tensor":音频特征向量list}`
```shell
bash start.sh
```
服务单元测试
```python
python app_test.py
```
其中，分别测试两种传参的方式，一种是文件传输，一种是文件转base64的传输方式。

## 参考仓库地址
[3D-Speaker](https://github.com/alibaba-damo-academy/3D-Speaker)

[pydub](https://github.com/jiaaro/pydub)

[CMGAN](https://github.com/ruizhecao96/CMGAN)

[speechbrain](https://github.com/speechbrain/speechbrain)

[audioFlux](https://github.com/libAudioFlux/audioFlux)

## 此仓库仅供学习，不能用作商用
