

首先是 从结果上来说，deeper is better ！
同样的参数下，多层网络就是比一层网络学出来的效果好。

然后就是，为什么 deeper 会 better？


## 1 Modularization

模组化， 举例：1 长头发男生样本很少的，分类问题。（有四个类别，长头发、短头发 * 男生、女生），先两两判断，再组合，模组化。比直接用4个分类器去判断，效果会更好，尤其是在长头发男生样本数量少的情况下。

**Basic Classifier Sharing by the following classifiers as module**

而且 The modularization is automatically learned from data


### 2 模组化 在图像上
回忆 CNN 的可视化每一层 output 出来的效果图，
> Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. In Computer Vision–ECCV 2014 (pp. 818-833)

#### Analogy
逻辑电路，组合逻辑电路 -- 多层神经网络之间的类比，

2 剪纸 -- 对折  
还做了一个实验，有意思的实验。神奇的实验，解释倒是还没有。


## End-to-end Learning
What each function should do is learned automatically

All functions are learned from data



## To learn more ...
* Deep Learning: Theoretical Motivations (Yoshua Bengio)
http://videolectures.net/deeplearning2015_bengio_the
oretical_motivations/
* Connections between physics and deep learning
https://www.youtube.com/watch?v=5MdSE-N0bxs
* Why Deep Learning Works: Perspectives from Theoretical
Chemistry
https://www.youtube.com/watch?v=kIbKHIPbxiU

http://rinuboney.github.io/2015/10/18/theoretical-motivations-deep-learning.html












