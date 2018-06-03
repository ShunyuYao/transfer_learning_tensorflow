# transfer_learning_tensorflow
A easy transfer learning program using tensorflow and tensornets  
用tensorflow和开源库[tensornets](https://github.com/taehoonlee/tensornets)实现迁移学习。  
### 程序主要针对百度点石竞赛商家招牌分类，也可迁移到其他任务。
## 功能简介：
- 支持35个模型
- 可设置随机图像增强
- 可选择finetune的变量数
- 可restore保存的ckpt接着训练
## 说明
- 更换模型只需替换`model.DenseNet201`为你想要transfer learn的model，具体支持的模型请移步[tensornets](https://github.com/taehoonlee/tensornets)  
- 请将数据文件放置于datasets文件夹下运行model/train.py即可开始训练
