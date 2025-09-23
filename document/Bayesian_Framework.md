以下内容分两部分：  
1. 一般条件期望与条件协方差的定义（适用于任意随机变量）  
2. 多元正态分布情形下常用的“条件均值（条件期望）公式”和“条件协方差公式”及完整推导（含三种思路：配方/Schur 补、块矩阵逆、线性最小二乘投影）

--------------------------------
# 1. 一般定义（任意分布）

设随机向量 $ X \in \mathbb{R}^p, \; Y \in \mathbb{R}^q。$
1) 条件分布密度（若存在）：  
$$
f_{X|Y}(x|y)=\frac{f_{X,Y}(x,y)}{f_Y(y)}
$$

2) 条件期望（条件均值）：  
$$
\mathbb{E}[X|Y=y] = \int_{\mathbb{R}^p} x \, f_{X|Y}(x|y)\, dx
$$

3) 条件协方差（条件二阶矩减去均值项）：  
$$
\mathrm{Cov}(X|Y=y) = \mathbb{E}\Big[(X-\mathbb{E}[X|Y=y])(X-\mathbb{E}[X|Y=y])^{\top}\Big|Y=y\Big]
$$

4) 若不显式写 y，常记  
$$
\mathrm{Cov}(X|Y)=\mathbb{E}\big[(X-\mathbb{E}[X|Y])(X-\mathbb{E}[X|Y])^{\top} \mid Y\big]
$$

对一般分布，除非特殊结构，写出封闭形式往往困难；多元正态是最经典能得到显式公式的情形。

--------------------------------
# 2. 多元正态分布的条件均值与条件协方差

## 2.1 模型设定与符号

设联合向量  
$$
\begin{pmatrix} X \\ Y \end{pmatrix} \sim \mathcal{N}\left(
\begin{pmatrix}\mu_X \\ \mu_Y\end{pmatrix},
\begin{pmatrix}
\Sigma_{XX} & \Sigma_{XY} \\
\Sigma_{YX} & \Sigma_{YY}
\end{pmatrix}
\right)
$$
其中：  
- $X \in \mathbb{R}^p,\; Y \in \mathbb{R}^q$  
- $\Sigma_{XX} \in \mathbb{R}^{p\times p}, \Sigma_{YY} \in \mathbb{R}^{q\times q}$ 均对称正定  
- $\Sigma_{XY} = \Sigma_{YX}^{\top}$

则条件分布 X|Y=y 仍为多元正态：  
$$
X|Y=y \sim \mathcal{N}\big( \mu_{X|Y=y}, \;\Sigma_{X|Y} \big)
$$

其显式公式为：  
(1) 条件均值（条件期望）  
$$
\mu_{X|Y=y} = \mu_X + \Sigma_{XY} \Sigma_{YY}^{-1} (y - \mu_Y)
$$

(2) 条件协方差  
$$
\Sigma_{X|Y} = \Sigma_{XX} - \Sigma_{XY} \Sigma_{YY}^{-1} \Sigma_{YX}
$$

矩阵 $\Sigma_{XX} - \Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX}$ 即 $\Sigma$ 相对于 $\Sigma_{YY}$ 的 Schur 补，必为对称正定（或半正定，视相关性结构而定），保证条件协方差合法。

--------------------------------
## 2.2 推导方法一：配方（完成平方）与 Schur 补

联合密度（忽略常数因子）：  
$$
f_{X,Y}(x,y) \propto \exp\left\{-\tfrac12
\begin{pmatrix}x-\mu_X \\ y-\mu_Y\end{pmatrix}^{\!\top}
\Sigma^{-1}
\begin{pmatrix}x-\mu_X \\ y-\mu_Y\end{pmatrix}
\right\}
$$

记 $\Sigma^{-1} = \begin{pmatrix} A & B \\ B^{\top} & D \end{pmatrix}$。  
则指数展开为：  
$$
-\tfrac12\Big[(x-\mu_X)^{\top}A(x-\mu_X) + 2(x-\mu_X)^{\top}B(y-\mu_Y) + (y-\mu_Y)^{\top}D(y-\mu_Y)\Big]
$$

边缘密度 $f_Y(y)$ 是正态：  
$$
f_Y(y) \propto \exp\{-\tfrac12 (y-\mu_Y)^{\top}\Sigma_{YY}^{-1}(y-\mu_Y)\}
$$

由 $f_{X|Y}(x|y)=f_{X,Y}(x,y)/f_Y(y)$，只需把与 $x$ 相关部分抽出：  
$$
f_{X|Y}(x|y) \propto \exp\left\{-\tfrac12 \Big[
(x-\mu_X)^{\top}A(x-\mu_X) + 2(x-\mu_X)^{\top}B(y-\mu_Y) 
+ (y-\mu_Y)^{\top}\big(D-\Sigma_{YY}^{-1}\big)(y-\mu_Y)
\Big]\right\}
$$

把与 $x$ 无关的项并入归一化常数，只剩：  
$$
\propto \exp\left\{-\tfrac12 \Big[
(x-\mu_X)^{\top}A(x-\mu_X) + 2(x-\mu_X)^{\top}B(y-\mu_Y)
\Big]\right\}
$$

对 $x$ 完成平方。令  
$$
m(y)=\mu_X - A^{-1}B(y-\mu_Y)
$$
则  
$$
(x-\mu_X)^{\top}A(x-\mu_X)+2(x-\mu_X)^{\top}B(y-\mu_Y)
= (x-m(y))^{\top}A(x-m(y)) - (y-\mu_Y)^{\top}B^{\top}A^{-1}B(y-\mu_Y)
$$

去掉与 $x$ 无关的第二项后，得到条件分布是以 $m(y)$ 为均值、协方差矩阵 $A^{-1}$ 的正态：  
$$
X|Y=y \sim \mathcal{N}\big(m(y), A^{-1}\big)
$$

接下来需要把 $A^{-1}$ 和 $-A^{-1}B$ 用原始协方差分块表示。利用块矩阵逆及 Schur 补公式：  
$$
\Sigma^{-1} =
\begin{pmatrix}
(\Sigma_{XX} - \Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX})^{-1} & * \\
* & *
\end{pmatrix}
$$
更精确地（标准块逆公式）：
$$
\Sigma^{-1}=
\begin{pmatrix}
S^{-1} & -S^{-1}\Sigma_{XY}\Sigma_{YY}^{-1} \\
-\Sigma_{YY}^{-1}\Sigma_{YX}S^{-1} & \Sigma_{YY}^{-1} + \Sigma_{YY}^{-1}\Sigma_{YX}S^{-1}\Sigma_{XY}\Sigma_{YY}^{-1}
\end{pmatrix},
$$
其中 Schur 补 $S = \Sigma_{XX} - \Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX}$。

对比可得：  
$$
A = S^{-1},\quad B = -S^{-1}\Sigma_{XY}\Sigma_{YY}^{-1}
$$

于是：  
$$
A^{-1} = S = \Sigma_{XX} - \Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX}
$$
$$
-A^{-1}B = -S (-S^{-1}\Sigma_{XY}\Sigma_{YY}^{-1}) = \Sigma_{XY}\Sigma_{YY}^{-1}
$$

代回 $m(y)$：  
$$
m(y)=\mu_X + \Sigma_{XY}\Sigma_{YY}^{-1}(y-\mu_Y)
$$
这与前述公式一致，且协方差即 Schur 补。

这段内容详细推导了联合高斯分布的条件分布公式，以下是逐步解析和解释：

---

### **1. 联合密度的定义**
联合分布的概率密度函数（忽略归一化常数）为：
$$
f_{X,Y}(x,y) \propto \exp\left\{-\frac{1}{2}
\begin{pmatrix}x-\mu_X \\ y-\mu_Y\end{pmatrix}^{\top}
\Sigma^{-1}
\begin{pmatrix}x-\mu_X \\ y-\mu_Y\end{pmatrix}
\right\}
$$
其中：
- $\Sigma^{-1}$ 是联合协方差矩阵的逆，分块形式为：
  $$
  \Sigma^{-1} = \begin{pmatrix} A & B \\ B^{\top} & D \end{pmatrix}
  $$
  - $A$：与 $X$ 相关的部分。
  - $D$：与 $Y$ 相关的部分。
  - $B$：表示 $X$ 和 $Y$ 的交叉关系。

将指数部分展开，可以分离出 $x$ 和 $y$ 的相关项：
$$
-\frac{1}{2}\Big[(x-\mu_X)^{\top}A(x-\mu_X) + 2(x-\mu_X)^{\top}B(y-\mu_Y) + (y-\mu_Y)^{\top}D(y-\mu_Y)\Big]
$$

---

### **2. 边缘分布的计算**
边缘分布 $f_Y(y)$ 是关于 $y$ 的正态分布：
$$
f_Y(y) \propto \exp\{-\frac{1}{2} (y-\mu_Y)^{\top}\Sigma_{YY}^{-1}(y-\mu_Y)\}
$$
其中：
- $\Sigma_{YY}^{-1}$ 是 $Y$ 的协方差矩阵的逆。

---

### **3. 条件分布公式**
条件分布的定义为：
$$
f_{X|Y}(x|y) = \frac{f_{X,Y}(x,y)}{f_Y(y)}
$$
通过除去边缘分布 \(f_Y(y)\)，只需保留与 \(x\) 相关的部分即可：
$$
f_{X|Y}(x|y) \propto \exp\left\{-\frac{1}{2} \Big[
(x-\mu_X)^{\top}A(x-\mu_X) + 2(x-\mu_X)^{\top}B(y-\mu_Y)
\Big]\right\}
$$

---

### **4. 完成平方并提取条件均值**
对 $x$ 完成平方以提取条件均值。令：
$$
m(y) = \mu_X - A^{-1}B(y-\mu_Y)
$$
则：
$$
(x-\mu_X)^{\top}A(x-\mu_X) + 2(x-\mu_X)^{\top}B(y-\mu_Y)
= (x-m(y))^{\top}A(x-m(y)) - (y-\mu_Y)^{\top}B^{\top}A^{-1}B(y-\mu_Y)
$$

其中：
- 第一项 $(x-m(y))^{\top}A(x-m(y))$ 是关于 $x$ 的项，表示条件分布的核心部分。
- 第二项 $-(y-\mu_Y)^{\top}B^{\top}A^{-1}B(y-\mu_Y)$ 是与 $x$ 无关的项，可以并入归一化常数。

因此，条件分布是以 $m(y)$ 为均值、协方差矩阵 $A^{-1}$ 的正态分布：
$$
X|Y=y \sim \mathcal{N}\big(m(y), A^{-1}\big)
$$

---

### **5. 用原始协方差分块表示条件均值和协方差**
接下来，需要将 $$A^{-1}$ 和 $-A^{-1}B$ 用原始协方差矩阵表示。

#### （1）块矩阵的逆公式
利用块矩阵逆公式，联合协方差矩阵的逆可以表示为：
$$
\Sigma^{-1} =
\begin{pmatrix}
S^{-1} & -S^{-1}\Sigma_{XY}\Sigma_{YY}^{-1} \\
-\Sigma_{YY}^{-1}\Sigma_{YX}S^{-1} & \Sigma_{YY}^{-1} + \Sigma_{YY}^{-1}\Sigma_{YX}S^{-1}\Sigma_{XY}\Sigma_{YY}^{-1}
\end{pmatrix}
$$
其中：
- $$
S = \Sigma_{XX} - \Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX}\
$$ 是 Schur 补，表示条件协方差矩阵。

#### （2）对比分块公式
通过对比可得：
$$
A = S^{-1},\quad B = -S^{-1}\Sigma_{XY}\Sigma_{YY}^{-1}
$$

#### （3）条件协方差和均值的表达
- 条件协方差：
  $$
  A^{-1} = S = \Sigma_{XX} - \Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX}
  $$
- 条件均值的修正项：
  $$
  -A^{-1}B = -S(-S^{-1}\Sigma_{XY}\Sigma_{YY}^{-1}) = \Sigma_{XY}\Sigma_{YY}^{-1}
  $$

代入条件均值公式：
$$
m(y) = \mu_X + \Sigma_{XY}\Sigma_{YY}^{-1}(y-\mu_Y)
$$

---

### **6. 总结**
通过联合高斯分布的密度函数，结合矩阵分块逆公式，推导出条件分布的公式：
1. **条件均值**：
   $$
   m(y) = \mu_X + \Sigma_{XY}\Sigma_{YY}^{-1}(y-\mu_Y)
   $$
   表示在给定 $y$ 时，$x$ 的预测均值。

2. **条件协方差**：
   $$
   \text{Cov}(X|Y) = \Sigma_{XX} - \Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX}
   $$
   表示在给定 $y$ 时，$x$ 的不确定性。

这些公式广泛应用于机器学习中，例如高斯过程回归和多元正态分布的分析。