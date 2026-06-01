## STEC model-diffusion based demo
这里的实验的基础是多星联合建模，也就是不再单星建模了。
- train中的训练比例划分，大概80%的ipp点来做trainset.然后，这其中的10-30%的点来做target，其余点为context。
- test中的测试比例划分，model_stations点全部作为target。
- 不同卫星导航系统分开建模，不共享权重
**用于服务器版本的代码更新**

