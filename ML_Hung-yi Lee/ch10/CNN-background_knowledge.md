


We immediately see that a Fourier transform contains a lot of information about the orientation of an object in an image. If an object is turned by, say, 37% degrees, it is difficult to tell that from the original pixel information, but very clear from the Fourier transformed values.

 （一篇文章包你弄懂傅里叶变换 - 博文可参考）

> 4，图像中傅立叶变换的物理意义
（1），图像的频率是表征图像中灰度变化剧烈程度的指标，是灰度在平面空间上的梯度。如：大面积的沙漠在图像中是一片灰度变化缓慢的区域，对应的频率值很低；而对于地表属性变换剧烈的边缘区域在图像中是一片灰度变化剧烈的区域，对应的频率值较高。傅立叶变换在实际中有非常明显的物理意义，设f是一个能量有限的模拟信号，则其傅立叶变换就表示f的谱。从纯粹的数学意义上看，傅立叶变换是将一个函数转换为一系列周期函数来处理的。从物理效果看，傅立叶变换是将图像从空间域转换到频率域，其逆变换是将图像从频率域转换到空间域。换句话说，傅立叶变换的物理意义是将图像的灰度分布函数变换为图像的频率分布函数，傅立叶逆变换是将图像的频率分布函数变换为灰度分布函数
> https://blog.csdn.net/ebowtang/article/details/39004979