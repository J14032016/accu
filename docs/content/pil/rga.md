# 区域生长算法

区域生长算法(RGA)的基本思想是将有相似性质的像素点合并到一起. 对每一个区域要先指定一个种子点作为生长的起点, 然后将种子点周围领域的像素点和种子点进行对比, 将具有相似性质的点合并起来继续向外生长, 直到没有满足条件的像素被包括进来为止. 这样一个区域的生长就完成了. 这个过程中有几个关键的问题:

1. 选取种子点
2. 创建一个与原图像大小相等的 0 矩阵 mask, 并置种子点为 1
3. 创建一个空的 vect, 将种子点存入
4. 从 vect 中依次弹出种子点并判断种子点和周围 8 领域的关系(生长规则), 相似的点则作为下次生长的种子点重新存入 vect 并将 mask 中对应位置置 1. 已判断过的种子点不应再次判断(及 mask 对应位置为 1)
5. 重复 4, 直至 vect 为空

确定在生长过程中能将相邻像素包括进来的准则包括`灰度图像的差值`, `彩色图像的颜色`等等.

# 代码实现

```py
import numpy as np
import PIL.Image

im = PIL.Image.open('/tmp/github-mark.png')
im = im.convert('L')
im = np.array(im)

seed = (40, 300)
mask = np.zeros(im.shape[:2], dtype=np.uint8)
mask[seed] = 1
vect = [seed]
area = [im[seed]]

while vect:
    n = len(vect)
    mean = np.sum(np.array(area), axis=0) / len(area)
    for i in range(n):
        seed = vect[i]
        s0 = seed[0]
        s1 = seed[1]
        for p in [
            (s0 - 1, s1 - 1),
            (s0 - 1, s1),
            (s0 - 1, s1 + 1),
            (s0, s1 - 1),
            (s0, s1 + 1),
            (s0 + 1, s1 - 1),
            (s0 + 1, s1),
            (s0 + 1, s1 + 1)
        ]:
            if p[0] < 0 or p[0] >= im.shape[0] or p[1] < 0 or p[1] >= im.shape[1]:
                continue
            if mask[p] == 1:
                continue
            # 区域生长条件: 灰度值差值小于等于 5
            if abs(mean - im[p]) <= 5:
                mask[p] = 1
                vect.append(p)
                area.append(im[p])
    vect = vect[n:]


mask = (1 - mask) * 255
im = PIL.Image.fromarray(mask)
im.show()
```

原图:

![img](/img/pil/rga/github-mark.png)

以 (300, 40) 为种子进行区域生长:

![img](/img/pil/rga/grow.png)
