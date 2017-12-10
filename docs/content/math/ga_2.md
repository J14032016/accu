# 最优保留策略

最优保留策略指**将群体中最优的一部分个体不经过选择, 交叉和变异操作, 直接进入下一代**, 以避免优秀个体损失.

最优保留策略的执行过程如下:

1. 找出当前群体中适应度最高和最低的个体
2. 若当前群体中最优个体比历史最优个体适应度还高, 则以当前群体最优个体作为历史最优个体; 否则使用历史最优个体替换当前群体最差个体
3. 执行后续遗传算子(选择, 交叉, 变异等)

Python 实现(copy 上节代码, 稍微修改 evolve 函数即可):

```py
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn')


class GA:
    def __init__(self):
        self.pop_size = 80
        self.opt_size = 2
        self.max_iter = 20
        self.pc = 0.6
        self.pm = 0.001
        self.dna_size = 10
        self.x_bound = [0, 5]

    def f(self, x):
        return np.sin(10 * x) * x + np.cos(2 * x) * x

    def encode(self, per):
        a = per / (self.x_bound[1] - self.x_bound[0]) * (2 ** self.dna_size - 1)
        a = int(a)
        return np.array(list(np.binary_repr(a).zfill(self.dna_size))).astype(np.uint8)

    def decode(self, dna):
        return dna.dot(1 << np.arange(self.dna_size)[::-1]) / (2**self.dna_size - 1) * self.x_bound[1]

    def getfit(self, pop):
        x = self.decode(pop)
        return self.f(x)

    def evolve(self):
        pop = np.random.randint(2, size=(self.pop_size, self.dna_size))
        yield pop
        opt = pop[:self.opt_size]

        for _ in range(self.max_iter - 1):
            fit = self.getfit(pop)
            idx = np.argsort(fit)[::-1]

            opt_sum = np.vstack((opt, pop[idx[:self.opt_size]]))
            opt_fit = self.getfit(opt_sum)
            opt_fit_idx = np.argsort(opt_fit)[::-1]
            opt = opt_sum[opt_fit_idx[:self.opt_size]]

            pop[-self.opt_size:] = opt
            fit[-self.opt_size:] = opt_fit_idx[:self.opt_size]

            fit = fit - np.min(fit) + 0.001
            idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fit / fit.sum())
            pop = pop[idx]

            for i in range(0, self.pop_size, 2):
                if np.random.random() < self.pc:
                    a = pop[i]
                    b = pop[i + 1]
                    p = np.random.randint(1, self.dna_size)
                    a[p:], b[p:] = b[p:], a[p:]
                    pop[i] = a
                    pop[i + 1] = b

            mut = np.random.choice(np.array([0, 1]), pop.shape, p=[1 - self.pm, self.pm])
            pop = np.where(mut == 1, 1 - pop, pop)

            yield pop


ga = GA()
gaiter = ga.evolve()

fig, ax = plt.subplots()
ax.set_xlim(-0.2, 5.2)
ax.set_ylim(-10, 7.5)
x = np.linspace(*ga.x_bound, 200)
ax.plot(x, ga.f(x))
sca = ax.scatter([], [], s=200, c='#CF6FC1', alpha=0.5)


def update(*args):
    pop = next(gaiter)
    fx = ga.decode(pop)
    fv = ga.f(fx)
    sca.set_offsets(np.dstack((fx, fv)))


ani = matplotlib.animation.FuncAnimation(fig, update, interval=200, repeat=False)
plt.show()
```
