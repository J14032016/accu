# 散点图

```py
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

ax = plt.subplot()
x = np.linspace(-np.pi, np.pi, 16)
y = np.sin(x)

# s: 散点大小, 默认 20
# c: 颜色
# alpha: 透明度
ax.scatter(x, y, s=50, c='#FF0000', alpha=0.5)
plt.show()
```

![img](/img/py/plt_scatter/sample.png)

# 样式

```py
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

ax = plt.subplot()
x = np.linspace(-np.pi, np.pi, 16)
y = np.sin(x)

# marker: 散点样式. 全部可支持散点样式见 matplotlib.markers 模块
ax.scatter(x, y, s=50, c='#FF0000', marker='+', alpha=0.5)
plt.show()
```

![img](/img/py/plt_scatter/sample_mark.png)
