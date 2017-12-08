dlib 是一个包含机器学习算法和工具的 c++ 库.

# 安装

```sh
git clone --depth=1 https://github.com/davisking/dlib.git
cd dlib
mkdir build; cd build; cmake .. ; cmake --build .
# 安装 python API
python setup.py install

# 执行测试
cd dlib/test
mkdir build
cd build
cmake ..
cmake --build . --config Release
./dtest --runall
```

详细请至 [https://github.com/davisking/dlib](https://github.com/davisking/dlib) 阅读官方文档.

**记录一: dlib Python API 需要 boost.python 支持**

简而言之, 前往 [http://www.boost.org/](http://www.boost.org/) 下载 boost 后, 使用如下命令安装即可, 注意 `--with-python` 配置 python 可执行文件, 安装脚本会自动寻找 python 的安装目录.

```sh
# 默认情况下不需要设置环境变量
export CPLUS_INCLUDE_PATH=/usr/local/python/include/python3.6m

./bootstrap.sh --prefix=/usr/local/boost/prefix --with-python=python3 --with-libraries=python
./b2
./b2 install
```

安装完毕后在 ~/.bash_profile 中设置环境变量

```sh
export BOOST_INCLUDE=/usr/local/boost/include
export BOOST_LIB=/usr/local/boost/lib
export PATH=$PATH:$BOOST_INCLUDE:$BOOST_LIB
```

**记录二: 内存过小导致编译失败**

```
c++: internal compiler error: Killed (program cc1plus)
Please submit a full bug report,
with preprocessed source if appropriate.
See <http://bugzilla.redhat.com/bugzilla> for instructions.
gmake[2]: *** [CMakeFiles/dlib_.dir/src/vector.cpp.o] Error 4
gmake[1]: *** [CMakeFiles/dlib_.dir/all] Error 2
gmake: *** [all] Error 2
error: cmake build failed!
```

测试时 1G 内存导致编译失败, 使用额外的 1G swap 后重新编译解决问题:

```sh
dd if=/dev/zero of=/data/swap bs=64M count=16
chmod 0600 /data/swap
mkswap /data/swap
swapon /data/swap
```

# 测试 dlib 提供的人脸检测示例

dlib 自带人脸检测模块, 其 python 脚本位于 `/python_examples/face_detector.py`. 由于机器没有 GUI 界面, 因此我简单修改了下, 可以将检测结果保存在本地.

```py
import sys
import cv2
import dlib

load_name = sys.argv[1]
save_name = sys.argv[2]

detector = dlib.get_frontal_face_detector()

img = cv2.imread(load_name)
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))
for i, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        i, d.left(), d.top(), d.right(), d.bottom()))
    cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 255), 1)

cv2.imwrite(save_name, img)
```

```sh
# 执行脚本
python face_detector.py obama.jpg obama_face.jpg
```

原图:

![img](/img/ml_dlib/obama.jpg)

人脸:

![img](/img/ml_dlib/obama_face.jpg)


# 测试 dlib 提供的人脸标注示例

dlib 自带人脸标注模块, 其 python 脚本位于 `/python_examples/face_landmark_detection.py`.

```py
import sys

import cv2
import dlib
import numpy as np

predictor_path = sys.argv[1]
load_name = sys.argv[2]
save_name = sys.argv[3]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

img = cv2.imread(load_name)
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))
for i, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        i, d.left(), d.top(), d.right(), d.bottom()))
    cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 255), 1)
    shape = predictor(img, d)
    print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
    shape = [(p.x, p.y) for p in predictor(img, d).parts()]
    for i, pos in enumerate(shape):
        # 可以打上每个点的索引
        # cv2.putText(
        #     img, str(i), pos,
        #     fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        #     fontScale=0.4,
        #     color=(0, 0, 255)
        # )
        cv2.circle(img, pos, 2, color=(0, 255, 255))

cv2.imwrite(save_name, img)
```

```sh
# 在使用该脚本前, 需要先下载预训练权重
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
# 执行脚本, 保存结果至 obama_landmark.jpg
python face_landmark_detection.py shape_predictor_68_face_landmarks.dat obama.jpg obama_landmark.jpg
```

![img](/img/ml_dlib/obama_landmark.jpg)
