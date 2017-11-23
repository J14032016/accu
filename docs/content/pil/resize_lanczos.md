# Lanczos 算法

在计算机图形学中, 图像缩放指的是通过增加或去掉像素来改变图片的尺寸. 由于要在效率和图像质量比如平滑度和清晰度之间做折衷, 图像缩放并不是个平庸的过程. 当图像尺寸增大的时候, 组成图像的像素也越来越大, 图像看上去就变"柔和"了. 而缩小图像的时候, 图像就变得平滑和清晰了. 本文介绍**Lanczos算法**.

Lanczos 算法是一种将对称矩阵通过正交相似变换变成对称三对角矩阵的算法, 以20世纪匈牙利数学家 [Cornelius Lanczos](https://en.wikipedia.org/wiki/Cornelius_Lanczos) 命名.

# 机理

Lanczos 算法与[双三次插值法](/content/pil/resize_bicubic/) 类似, 不同点在于权重函数不同. Lanczos 算法权重函数如下所示:

$$
L(x)={
    (1, if x=0),
    (asin((pix))sin((pixtext{/}a)) // pi^2x^2, if -a <= x < a and x != 0),
    (0, if text{otherwise})
:}
$$

通常, a 取值为 2 或 3. 当 a = 2 时适合图像缩小, a = 3 时适合图像放大.

编写以下代码, 画出权重函数图像

```py
import numpy as np
import matplotlib.pyplot as plt

lanczos_a = 2

def get_w(x):
    if x > -lanczos_a and x < lanczos_a:
        return np.sinc(x) * np.sinc(x / lanczos_a)
    return 0

x = np.linspace(-3, 3, num=100)
plt.plot(x, list(map(get_w, x)))
plt.plot(x, np.zeros(len(x)))
plt.show()
```

a = 2 时的函数图像:

![img](/img/pil/resize_lanczos/lanczos_a2.png)

a = 3 时的函数图像:

![img](/img/pil/resize_lanczos/lanczos_a3.png)

# 代码实现

拷贝双三次插值法的代码实现, 修改权重函数即可. 下述代码以图像缩小(a = 2)为例

```py
import numpy as np
import PIL.Image
import scipy.misc


def get_item(arr, *args):
    indexes = []
    for i, entry in enumerate(args):
        index = entry
        if index < 0:
            index = abs(index) - 1
        if index >= arr.shape[i]:
            index = arr.shape[i] - index % arr.shape[i] - 1
        indexes.append(index)
    r = arr
    for index in indexes:
        r = r[index]
    return r


lanczos_a = 2


def get_w(x):
    if x > -lanczos_a and x < lanczos_a:
        return np.sinc(x) * np.sinc(x / lanczos_a)
    return 0


im = PIL.Image.open('/img/jp.jpg')
im_mat = scipy.misc.fromimage(im)
im_mat_resized = np.empty((270, 480, im_mat.shape[2]), dtype=np.uint8)

for r in range(im_mat_resized.shape[0]):
    for c in range(im_mat_resized.shape[1]):
        rr = (r + 1) / im_mat_resized.shape[0] * im_mat.shape[0] - 1
        cc = (c + 1) / im_mat_resized.shape[1] * im_mat.shape[1] - 1

        rr_int = int(rr)
        cc_int = int(cc)

        sum_p = np.empty(im_mat.shape[2])
        for j in range(rr_int - lanczos_a + 1, rr_int + lanczos_a + 1):
            for i in range(cc_int - lanczos_a + 1, cc_int + lanczos_a + 1):
                w = get_w(rr - j) * get_w(cc - i)
                p = get_item(im_mat, j, i) * w
                sum_p += p

        for i, entry in enumerate(sum_p):
            sum_p[i] = min(max(entry, 0), 255)

        im_mat_resized[r][c] = sum_p

im_resized = PIL.Image.fromarray(im_mat_resized)
im_resized.show()
```
