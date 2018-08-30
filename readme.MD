# Machine Learning course for internal use

## Lesson outline (还需要细化)
0. [CH.0 Overview](https://github.com/ouriris/ml-course/tree/master/chapter.0)
	- 课程背景简介
	- Data scientist roadmap
	- 机器学习实践流程
1. [CH.1 Python basic skills](https://github.com/ouriris/ml-course/tree/master/chapter.1) [(PPT)](https://github.com/ouriris/ml-course/tree/master/chapter.1/ppt)
	- 背景(优点、生态链)
	- 环境搭建
		- Anaconda使Python2/3共存(Conda的使用)
		- Jupyter Notebook(可分享的交互式编程环境)
		- PyCharm(IDE)
	- 语法特点(主要集中于Python与其它语言不同的特点上)
		- 数据类型(空值None、没有定义常量的办法、List和Tuple、集合运算)
		- 流程控制(没有花括号、靠缩进、作用域)
		- Dict & Set
		- 函数定义(默认参数、可变参数、命名参数、多返回值)
		- 高级特性(Slice、生成器)
		- 函数式编程(高阶函数)
		- OOP(如何定义class、构造函数、一般函数、私有变量、静态变量、继承、枚举)
		- 异常处理(try-except-finally、自定义异常)
		- Unittest
	- [Code Samples](https://github.com/yellowb/ml-sample)
2. [CH.2 ML basic models + basic math](https://github.com/ouriris/ml-course/tree/master/chapter.2)
	- 机器学习评价方法及性能度量
		- 留出法
		- 交叉验证法
		- 自助法
		- 错误率和精度
		- 查准率、查全率与F1
		- ROC与AUC
		- 代价敏感错误率与代价曲线
	- 监督学习模型(理论 + Demo, 引入sk-learn)
		- 回归
			- 线性回归
				- 应用场景
				- 算法原理 （预测函数， 成本函数， 梯度下降算法）
				- 算法示例： 线性归回算法拟合正弦函数
				- scikit-learn demo: 房价预测
		- 决策树
			- 决策树介绍
				- 分类问题
				- 分类方法
				- 决策树的优缺点
				- 决策树算法的代表
				- 随机森林
			- 如何构建决策树
				- 基本算法
				- 划分属性的选择策略
				- 剪枝
				- 连续属性值如何划分
				- 缺失值
				- 多变量决策树
		- 朴素贝叶斯
			- 背景（先验概率、后验概率）
			- 贝叶斯公式
			- 朴素贝叶斯的原理公式解释（结合例子）（为什么要假设特征之间独立？特征空间巨大、数据的稀疏性很容易导致统计到0）
			- 朴素贝叶斯Naive在哪
				- 优点：算法逻辑简单,易于实现、分类过程中时空开销小、可以应用在数据量大的情况
				- 缺点：朴素贝叶斯模型假设属性之间相互独立，这个假设在实际应用中往往是不成立的，在属性个数比较多或者属性之间相关性较大时，分类效果不好
			- 朴素贝叶斯的优点与缺点
			- 朴素贝叶斯与LR的区别
		- ...
	- 非监督学习模型(理论 + Demo, 引入sk-learn)
		- 聚类
			- Kmeans
			- Kmeans++
			- DBSCAN
				- DBSCAN 介绍（基于密度的聚类算法）
				- DBSCAN 算法关键属性（邻域，密度，最小密度阈值minpts，核心点，边界点，噪音点，直接密度可达，密度可达，密度相连）
				- 算法流程 & sklearn demo
				- 优缺点
3. [CH.3 Feature engineering]()
	- Introduction
		- 什么是特征
		- 什么是特征工程
		- 特征工程的重要性
	- Data cleaning
		- 清除异常数据
		- 采样
	- Data preprocessing and Feature extraction
		- 无量纲化
			- 标准化
				- z-score法
			- 归一化
				- 区间缩放法
		- 离散化
		- 定性特征转化成定量特征
			- oneHot编码
		- 缺失值处理
		- 数据转换
	- Feature selection
		- 特征选择标准
			- 特征是否发散
			- 特征与目标的相关性
		- 特征选择方法
			- 过滤法
			- 包装法
			- 嵌入法
	- Dimensionality reduction
		- 主成分分析法（PCA）
		- 线性判别分析（LDA）
4. [CH.4 Advanced tech]()
	- Bagging
	- Boosting
	- Stacking
	- 欠拟合和过拟合
5. [CH.5 Real Project sharing]()
	- // from ITA?
6. [CH.6 Deep learning(未确定)]()
	- What is DL
	- 神经网络
	- CNN/RNN/LSTM...
	- Demo

## Members
- [yellowb](https://github.com/yellowb)
- [qisibajie](https://github.com/qisibajie)
- [Fairylord](https://github.com/Fairylord)
- [TmacZhou](https://github.com/TmacZhou)
- [aaronmayo](https://github.com/aaronmayo)
- [JonasYuGit](https://github.com/JonasYuGit)

## 当前TODO List
1. 分组如下, 各组人员请细化自己负责那个item的outline(yellow@2018-6-22)
	- ~~Yellow + Tmac: CH.0 & CH.1~~ (Completed)
  - Diana + Jonas: CH.2回归模型
  - Will + Aaron: CH.2决策树模型
  - Yellow + Tmac: CH.2朴素贝叶斯

## Rules
1. 每个章节放在对应子目录下, 章节目录下需要有README.MD文件.
2. 更新了课件要一起更新README.MD.
3. 讨论请去Issues.
