# TorchServant

中文版本 | [English Version](https://github.com/QixuanAI/pytorch_AI_Engine/blob/master/README_en-US.md)

这是一个帮助[PyTorch](https://pytorch.org)用户快速进行深度学习开发的工具集。
它的设计思想是：将研究者从模型加载保存、数据记录与可视化等繁琐重复的工作中解放出来，
让用户只需专注于
**模型设计**、
**超参数选取**、
**损失函数与优化器设计**、
**训练模型**、
**验证模型**
这几个核心步骤，从而
**快速创建并调试基于[PyTorch](https://pytorch.org)的深度神经网络**。

它包含了许多有用的工具，比如
**自动化的模型保存/加载**、
**紧急中断恢复**、
**基于[Visdom](https://github.com/facebookresearch/visdom)或[TensorBoardX](https://github.com/lanpa/tensorboardX)的学习过程可视化**、
**模型迁移/复用帮助工具**（开发中）、
**模型可视化工具**（待开发）、
**权重可视化工具**（待开发）、
**个性化进度条**、
**可复现Dataloader**（待开发）等等。

目前该应用还处在设计开发阶段，后续还将视情况添加更多功能。欢迎有志于此的朋友们加入此项目。

## 安装

```bash
pip install torchservant
```

## 示例



### 使用内置模型

### 创建一个模型





## BasicConfig：默认配置

### 常用文件或目录的位置

* ```train_data```：训练数据位置

  可以是一个记录所有数据的文件，也可以指向一个目录，具体类型及用法取决于用户的Datasets相关代码。

* ```val_data```：验证数据位置

  同上。

* ```weight_load_path```：权重文件加载位置

  通常用于加载预训练的权重文件，或者加载上一次的权重文件用于继续训练。

* ```weight_save_path```：权重文件保存位置

  训练结束后保存权重文件的位置。可以使用格式化选项，可选的选项有
  * ```{time}```：程序开始时间；
  * ```{mode}```：模型模式（train或inference）
  * ```{epoch}```：权重保存时的迭代数；
  * ```{model}```：配置中指定的模型名称；
  * ```{loss}```：最终损失函数值；
  * ~~```{optim}```：优化器类型；~~
  * ```{score}```：验证分数，通常是准确率。

* ```log_root```：训练过程记录位置

  包括本次训练的配置，迭代数，时间，每次迭代时的损失函数、准确率等。一些临时文件也会存放在此处，包括用于紧急中断恢复的临时权重文件。
  
* ```history_root```：训练历史转存位置

  在当前训练正常结束后，会将相关训练记录转存到训练历史中，并清空当前训练记录保存位置，以防止与下一次的训练记录混淆。实际的转存目录会在给定目录下创建带有时间戳的子目录。如果为None则不转存。

### 常用参数

#### 数据相关

* ```classes_list```：一个表示所有类别的列表，其内容由用户定义。在指定了类别列表后，将会自动计算```self.num_classes```供网络模型获取类别数量。仅在分类问题时使用，在回归问题中可以忽略此参数。
* ```shuffle_train```：是否打乱训练数据的顺序，默认为```True```。
* ```shuffle_val```：是否打乱验证数据的顺序，默认为```False```。
* ``` drop_last_train```：当最后一批训练数据不够组成完整batch时，是否放弃这部分数据。默认为```True```。
* ``` drop_last_val```：当最后一批验证数据不够组成完整batch时，是否放弃这部分数据。默认为```False```。

#### 效率相关

* ``` use_gpu```：是否使用GPU加速，默认为```True```。如果设为```True```而系统内找不到可用的GPU，则会自动变为```False```。
* ``` num_data_workers```：用于读取数据的线程数，默认为1。
* ``` pin_memory```：是否固定数据在内存中的位置以加快训练速度，仅在系统内存充足的情况下设为```True```。默认为```False```。
* ``` time_out```：载入数据时的最大等待时间，0表示不限时。默认为0。
* ``` max_epoch```：训练时的最大迭代次数。达到这个次数后，会自动保存权重文件。
* ```batch_size```：数据批的大小。
* ```ckpt_freq```：更新显示进度并保存检查点的频率，数值表示经过的数据批。
* ```reproducible_record ```：完全可复现记录，当此项为```True```时，将会记录每次迭代的详细情况，以用作完全复现，但会占用大量资源。默认为```False```。

#### 模型相关

* ```model```：要使用的网络模型。可以指定的参数类型有：
  * 字符串：如果使用./models目录下预置的模型名，则可以直接加载相应预置模型；反之则仅作为表示模型名称的tag，不起任何其它作用；
  * 返回一个```torch.Module```的函数：会在准备阶段调用此函数以获取model。该函数的参数类型应当是```(config, **kwargs)```，其中```config```即为当前```BasicConfig```或任何继承的```BasicConfig```类的实例，```**kwgargs```可以在初始化```config```时指定；
  * ```None```：忽略该参数，默认为该项。
* ```mode```：网络模型的模式，可用选项有（不区分大小写）：
  * 训练模式：```"train"```, ```"training"```
  * 引用模式：```"inference"```, ```"validation"```, ```"val"```, ```"test"```, ```"evaluation"```, ```"eval"```
* ``use_batch_norm``：是否使用[L2正则化](https://pytorch.org/docs/stable/nn.html#batchnorm2d)。该参数需要网络模型的初始化参数的配合实现，默认为```True```。
* ``loss_type``：使用的损失函数类型，依据用户代码自由指定，主要用于存储训练记录与权重文件时的tag。
* ~~optimizer：优化器类型。可以指定的参数类型有：~~
  * ~~字符串：使用预设优化器，可选列表为：```["sgd", "adam", “”]```~~
  * ~~返回一个```torch.optim.Optimizer```的函数~~

#### 可视化相关

* ``visual_engine``：要使用的可视化引擎，可用选项有（不区分大小写）：
  * 使用[Visdom](https://github.com/facebookresearch/visdom)：```"visdom"```, ```"vis"```
  * 使用[TensorBoardX](https://github.com/lanpa/tensorboardX)：```"tensorboardx"```, ``"tensorboard"``, ``"tb"``

* ``port``：访问可视化网页的端口。
* ```visdom_env```：Visdom的默认环境名称。

#### 自动生成的参数

* init_time：一个表示配置文件初始化时间的字符串
* enable_grad：由网络模式决定是否启用梯度计算
* num_classes：由classes_list计算得到的类别数量
* num_gpu：系统发现的GPU数量，**注意：当batch_size不能被num_gpu整除时，系统将会报错**
* gpu_list：可用GPU列表
* map_location：设备分配方案
* vis_env_path：
