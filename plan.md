Author: Yulun Wu

接下来的课程以jupyter notebook的形式发布

# 效果图
来自[kaggle.com](https://www.kaggle.com/learn/overview)  
![image.png](https://i.loli.net/2020/03/31/kiSKpExgebwdFDt.png)

# 工程需求

用户注册登录

架设jupyter notebook服务器 (别用jupyter lab), 参考[kaggle](https://www.kaggle.com/learn/overview)
* 给用户分配session, 开辟空间保存每个用户代码, 载入等等
* 注意设置权限, 因为 jupyter notebook 可以用 `!command` 调用bash命令  
* 管理计算资源 e.g. 自动关闭 inactive session
* 做一个体现主题的外观 可参考jupyter-theme, 或者用别的方法


# 课程制作
希望参与制作AI课程的你需要学习: (从会python开始约5h)
* Python, jupyter notebook
* 库: numpy, pytorch, matplotlib

每一个课程基于jupyter notebook制作, 参考[kaggle](https://www.kaggle.com/learn/overview)
## Feature
以代码填空的形式进行
```python
# create a variable called color with an appropriate value on the line below
# (Remember, strings in Python must be enclosed in 'single' or "double" quotes)
____

# Check your answer
q0.check()
```

自己写好用于check用户是否正确填写代码和展示solution的模块, 暂且称作`learning tool`
```python
from learntools.core import binder; binder.bind(globals()) # 这里
from learntools.python.ex1 import * # 这里
print("Setup complete! You're ready to start question 0.")

# create a variable called color with an appropriate value on the line below
# (Remember, strings in Python must be enclosed in 'single' or "double" quotes)
____

# Check your answer
q0.check()

#q0.hint()
#q0.solution()
```

## Topic
有自己的点子优先, 或者直接联系我我会给你一个topic  
模仿也可取! 参考coursera, Udemy, DataCamp, kaggle, google AI education, ...  
如果纯自己写有困难 更可以将已有代码挖空, 加上提示和`learning tool`(上面说的)即可. 可以从别人的代码或我的[kaggle notebook](https://www.kaggle.com/idiott/mnist-shallow-deep-and-cnn)开始着手

