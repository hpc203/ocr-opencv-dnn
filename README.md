# 极简主义OCR
在这个OCR程序中，文本检测用的是EAST，文本识别用的是CRNN，这套程序有Python和C++两种版本的实现。
这两个网络的前向推理依靠opencv的dnn模块实现的，整个程序的运行不依赖任何深度学习框架pytorch,tensorflow等等的。

Python版本的主程序是text_detect_recognition.py，C++版本的主程序是text_detect_recognition.cpp。
在运行程序前，要先下载模型文件放在同一目录下。
EAST模型的下载链接是：https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1

crnn的模型下载链接是：https://drive.google.com/drive/folders/1cTbQ3nuZG-EKWak6emD_s8_hHXWz7lAr

在编写这套程序时，我有一个疑惑，具体内容可参见我的CSDN博客文章：
https://blog.csdn.net/nihate/article/details/108754622

2021年，我发布了使用DBNet做文字检测的程序，包含C++和Python两个版本的。可以把上面的程序里的文件检测模块EAST网络替换成DBNet
