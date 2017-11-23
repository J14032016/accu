# 盒取样法

在计算机图形学中, 图像缩放指的是通过增加或去掉像素来改变图片的尺寸. 由于要在效率和图像质量比如平滑度和清晰度之间做折衷, 图像缩放并不是个平庸的过程. 当图像尺寸增大的时候, 组成图像的像素也越来越大, 图像看上去就变"柔和"了. 而缩小图像的时候, 图像就变得平滑和清晰了. 本文介绍**盒取样法(Box sampling)**.

[双线性算法](/content/pil/resize_bilinear/)的一个缺点是它抽样了特定数量的像素. 当向下缩放到某一阈值以下时, 如缩放至原图一半以下,算法将采样相间隔的像素, 这会导致数据丢失, 并导致不精确的粗略结果.

这个问题的一个简单的解决方案是**盒取样**, 它考虑目标像素点对应原图中的一个框, 并采样所有框中的像素. 这可确保所有输入像素都对输出像素做出了贡献. 对于图像放大, 其等效于[最近邻插值法](/content/pil/resize_nearst/). 该算法的主要缺点是难以优化.

# 机理

假想以下一个 3x3 二维矩阵

$$
A = [[a_11,a_12, a_13],[a_21,a_22, a_23], [a_31, a_32, a_33]]
$$

如果将该矩阵缩放到 2x2, 那么目标矩阵第一个元素的值由一个$$a_11$$, 半个$$a_12$$, 半个$$a_21$$ 和 $$frac{1}{4}$$个$$a_22$$的归一化和构成. 因此 $$A^'$$ 的表达式为:

$$
A^' = frac{1}{1.5^2} * [[a_11 + frac{1}{2}a_12 + frac{1}{2}a_21 + frac{1}{4}a_22, frac{1}{2}a_12 + a_13 + frac{1}{4}a_22 + frac{1}{2}a_23], [frac{1}{2}a_21 + frac{1}{4}a_22 + a_31 + frac{1}{2}a_32, frac{1}{4}a_22 + frac{1}{2}a_23 + frac{1}{2}a_32 + a_33]]
$$

使用计算机语言描述, 有 $$a^'_11$$=sum(A[0:1.5, 0:1.5]) / (1.5 * 1.5), 即 $$a^'_11$$ 为矩阵 A 第 0 到第 1.5 行, 第 1 到第 1.5 列的和与其面积的商.

同时由于**线性可分性**, 对二维矩阵的操作可转换为两次对一维矩阵的操作. 下面的程序实现就使用了该技巧来优化计算速度.

# 代码实现
```py
import math

import numpy as np
import PIL.Image
import scipy.misc


def getsum_1d(arr, index):
    """一维矩阵求和函数: getsum_1d([...], [0, 1])"""
    intindex0 = int(index[0])
    intindex1 = int(index[1])
    sum_num = sum(arr[intindex0:intindex1])
    if index[0] > intindex0:
        sum_num -= arr[intindex0] * (index[0] - intindex0)
    if index[1] > intindex1:
        sum_num += arr[intindex1] * (index[1] - intindex1)
    return sum_num


def getsum_2d(arr, index):
    """二维矩阵求和函数: getsum_1d([...], [[0, 1], [0, 1]])"""
    row_sums = []
    for r in range(int(index[0][0]), math.ceil(index[0][1])):
        row = arr[r]
        row_sum = getsum_1d(row, index[1])
        row_sums.append(row_sum)
    return getsum_1d(row_sums, (index[0][0] - int(index[0][0]), index[0][1] - int(index[0][0])))


def convert_2d(r, size):
    du = r.shape[0] / size[0]
    dv = r.shape[1] / size[1]
    s = np.zeros(size, dtype=np.uint8)
    for sr in range(s.shape[0]):
        for sc in range(s.shape[1]):
            index = [[sr * du, (sr + 1) * du], [sc * dv, (sc + 1) * dv]]
            s[sr][sc] = int(getsum_2d(r, index) / (du * dv))
    return s


def convert_3d(r, size):
    s_dsplit = []
    for d in range(r.shape[2]):
        rr = r[:, :, d]
        ss = convert_2d(rr, size)
        s_dsplit.append(ss)
    s = np.dstack(s_dsplit)
    return s


im = PIL.Image.open('/img/jp.jpg')
im_mat = scipy.misc.fromimage(im)
im_converted_mat = convert_3d(im_mat, (270, 540))
img_resized = PIL.Image.fromarray(im_converted_mat)
img_resized.show()
```
