# CP-VITON-
经典CP-VITON的复现
大致流程
1. 加载VITON数据集并对其进行预处理，作为GMM-data
2. 训练GMM模型
3. 测试GMM模型
4. 基于GMM的结果生成TOM-data
5. 训练TOM模型
6. 测试TOM模型

代码结构说明
1. 数据加载和预处理部分：
   - 从data/train文件夹中加载数据集，数据集包括cloth,cloth-mask,image,image-parse,
pose五部分。
定义了一个名为 Dataset 的数据集类和一个名为 DataLoader 的数据加载器类，这些类用于处理衣服图像、人体图像和对应的遮罩，以及姿势关键点等信息，为了在几何匹配模块（Geometric Matching Module）中使用。
首先，Dataset 类是一个自定义的数据集类，它继承自 PyTorch 中的 data.Dataset 类。这个类用于加载并预处理衣服图像、人体图像、遮罩和姿势关键点等数据。它实现了 __init__、__getitem__ 和 __len__ 方法：
__init__ 方法：初始化数据集，设置数据集根目录、数据模式、阶段、数据列表等参数，并加载相关数据信息。
__getitem__ 方法：根据给定的索引获取数据集中特定索引位置的数据项，对衣服图像、人体图像、遮罩和姿势关键点等进行预处理。
__len__ 方法：返回数据集中图像的数量。
另外，DataLoader 类是一个数据加载器类，它用于批量加载数据集中的样本，并支持对数据进行迭代访问。它接受数据集对象并根据参数设置创建 PyTorch 的数据加载器。next_batch 方法用于获取数据加载器的下一个批次数据。
代码的最后部分用命令行参数设置了各种数据集相关的选项，创建了数据集对象和数据加载器对象，并展示了数据集和数据加载器的大小信息。最后，通过调用 IPython 库中的 embed() 方法，进入交互式的 IPython 环境，以便用户可以进一步探索和操作数据。
-在cp_dataset.py中加载数据和对数据进行预处理，最终会形成训练需要用到的如下数据。
    
2. 参数设置：分别在train和test文件开头通过get_opt方法设置好参数,下图是train文件设置的参数
   
3. 训练：在train文件中分别对gmm和tom进行训练。
开始训练：确认 train.py 中的 opt 参数正确配置后，运行 train_gmm() 函数，该函数会引用 networks.py 中定义的 GMM 类作为模型来开始训练。训练大约需要进行 200,000 个步骤。
保存权重：
训练结束后，会生成 finnal_gmm.pth 文件，这个文件将作为测试阶段的权重。在上述的参数设置的图里面有一个step参数，使其50000为一个循环，这样可以解决训练过程中途训练崩掉的情况。

4.	网络模型：networks文件给出了网络的代码，VITON由gmm和tom组成。

5. 测试：测试 GMM 模块
设置命令行参数，修改命令行参数以进行测试：python test.py --datamode test --stage GMM
运行测试：运行测试过程会在 result 文件夹内生成两个新的数据集 warp_cloth 和 warp_mask。将测试数据放入训练集中，将生成的 warp_cloth 和 warp_mask 文件放入 data/train 中，以便后续训练 TOM 模块时使用。TOM模块的训练同理   

