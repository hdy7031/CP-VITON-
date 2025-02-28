import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os.path as osp
import numpy as np
import json

class Dataset(data.Dataset):

    def __init__(self, opt):# 接受一个参数 opt，其中包含了数据集的各种选项和设置
        super(Dataset, self).__init__()
        # base setting
        self.opt = opt# 将传入的选项保存为对象的属性，以便在整个类中使用
        self.root = opt.dataroot# 设置数据集的根目录
        self.datamode = opt.datamode # train or test or self-defined# 设置数据模式，可以是训练、测试或者自定义的模式
        self.stage = opt.stage # 设置阶段，可能是GMM（Gaussian Mixture Model）或者TOM（Try-On Module）
        self.data_list = opt.data_list# 设置数据列表，指向数据的文件列表或索引
        self.fine_height = opt.fine_height# 设置图像的高度
        self.fine_width = opt.fine_width# 宽度
        self.radius = opt.radius# 设置半径
        self.data_path = osp.join(opt.dataroot, opt.datamode)# 构建数据集的路径，使用给定的根目录和数据模式
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#图像转换管道，将图像转换为张量（tensor）并进行归一化的操作
        self.transform_1d = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])
        self.transform_1d_x = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))])
        
        # load data list
        im_names = []# 存储图像文件名
        c_names = []# 存储掩码文件名
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()# strip()去除行尾换行符， split()分割行中的内容
                im_names.append(im_name)
                c_names.append(c_name)
        # 保存在数据集对象的属性中
        self.im_names = im_names
        self.c_names = c_names

    def name(self):
        return "Dataset"

    def __getitem__(self, index):# 根据给定的索引 index 获取数据集中特定索引位置的数据项
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        # cloth image & cloth mask 如果是 'GMM'，则加载 'cloth' 文件夹中的图像和掩码数据，否则加载 'warp-cloth' 文件夹中的图像和掩码数据
        if self.stage == 'GMM':
            c = Image.open(osp.join(self.data_path, 'cloth', c_name))
            cm = Image.open(osp.join(self.data_path, 'cloth-mask', c_name))
        else:
            c = Image.open(osp.join(self.data_path, 'warp-cloth', c_name))
            cm = Image.open(osp.join(self.data_path, 'warp-mask', c_name))
        c = self.transform(c)  # 将图像转换为张量，并进行归一化操作，使其值在 [-1, 1] 范围内
        cm_array = np.array(cm)# 将掩码 cm 转换为 NumPy 数组
        cm_array = (cm_array >= 128).astype(np.float32)# 掩码数组中大于等于 128 的像素值设为 1，其余像素值设为 0
        cm = torch.from_numpy(cm_array) # [0,1] 转换为 PyTorch 的张量
        cm.unsqueeze_(0)# 在张量的第0维（即通道维度）上增加一个维度

        # 加载 person image
        im = Image.open(osp.join(self.data_path, 'image', im_name))
        im = self.transform(im) # [-1,1]

        # 加载 parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(osp.join(self.data_path, 'image-parse', parse_name))
        # png color image is with colormap. 对分割图像进行图像处理
        parse_array = np.array(im_parse)
        parse_shape = (parse_array > 0).astype(np.float32)# 将非背景部分（像素值大于0）设为1，背景部分设为0，得到人物形状的二值掩码
        parse_head = (parse_array == 1).astype(np.float32) + \
                (parse_array == 2).astype(np.float32) + \
                (parse_array == 4).astype(np.float32) + \
                (parse_array == 13).astype(np.float32)# 根据特定的像素值提取头部部分的掩码信息
        parse_cloth = (parse_array == 5).astype(np.float32) + \
                (parse_array == 6).astype(np.float32) + \
                (parse_array == 7).astype(np.float32)# 根据特定的像素值提取衣物部分的掩码信息
       
        # 对之前提取的掩码信息进行了降采样和处理，以生成模型需要的输入数据
        parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8)) # 像素值范围从 [0, 1] 转换为 [0, 255] 的整数
        parse_shape = parse_shape.resize((self.fine_width//16, self.fine_height//16), Image.BILINEAR) # 下采样
        parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR) # 再上采样
        shape = self.transform_1d(parse_shape) # [-1,1]
        phead = torch.from_numpy(parse_head) # [0,1]
        pcm = torch.from_numpy(parse_cloth) # [0,1]

        # 图像遮罩
        im_c = im * pcm + (1 - pcm) # 保留衣物部分并且在其他区域填充值为 1，因为 (1 - pcm) 将非衣物部分设为 1
        im_h = im * phead - (1 - phead) # 保留头部部分并在其他区域填充值为 0，因为 (1 - phead) 将非头部部分设为 0

        #  加载姿势关键点，生成人体姿势关键点的热力图
        pose_name = im_name.replace('.jpg', '_keypoints.json')# 图像文件名创建了关键点的文件名
        with open(osp.join(self.data_path, 'pose', pose_name), 'r') as f:
            pose_label = json.load(f)# 获取关键点数据并重塑为 (-1, 3) 形状的 NumPy 数组
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1,3))

        point_num = pose_data.shape[0]# 获取关键点的数量
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)# 创建空的张量用于存储姿势热力图
        r = self.radius# 设置热力图的绘制半径
        im_pose = Image.new('L', (self.fine_width, self.fine_height))# 创建一个灰度图像，用于绘制热力图
        pose_draw = ImageDraw.Draw(im_pose)
        # 遍历关键点数据并生成热力图
        for i in range(point_num):
            # 创建一个空白的灰度图像
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            # 获取关键点的 x 和 y 坐标
            pointx = pose_data[i,0]
            pointy = pose_data[i,1]
            # 如果关键点在图像范围内，绘制热力图
            if pointx > 1 and pointy > 1:
                # 绘制一个矩形作为关键点的热力图
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            # 将图像转换为张量，并存储到 pose_map 中的对应位置
            one_map = self.transform_1d(one_map)# 一个关键点
            pose_map[i] = one_map[0]

        # 可视化关键点
        im_pose = self.transform_1d(im_pose)

        # 创建了一个“与服装无关的表示”（cloth-agnostic representation）
        # 将三个不同的张量拼接在一起，以形成一个新的张量
        agnostic = torch.cat([shape, im_h, pose_map], 0)
        '''
                将这些信息拼接在一起形成一个新的张量 agnostic，这个张量将包含多个通道（来自于拼接的不同张量），
                每个通道可能代表了不同的特征信息，这些特征信息与服装无关，而是关于人体的轮廓、头部信息和姿势关键点。
                这种“与服装无关的表示”可能用于神经网络模型的训练或其他相关任务，以便于模型学习和利用与服装无关的人体特征。
        '''
        if self.stage == 'GMM':
            # 加载并处理网格图像
            im_g = Image.open('grid.png')
            im_g = self.transform(im_g)
        else:
            im_g = ''

        result = {
            'c_name': c_name,  # 用于可视化
            'im_name': im_name,  # 用于可视化或实际数据
            'cloth': c,  # 输入数据
            'cloth_mask': cm,  # 输入数据
            'image': im,  # 用于可视化
            'agnostic': agnostic,  # 输入数据
            'parse_cloth': im_c,  # 实际数据
            'shape': shape,  # 用于可视化
            'head': im_h,  # 用于可视化
            'pose_image': im_pose,  # 用于可视化
            'grid_image': im_g,  # 用于可视化
            }

        return result

    # 返回了数据集中图像的数量
    def __len__(self):
        return len(self.im_names)

class DataLoader(object):
    def __init__(self, opt, dataset):
        super(DataLoader, self).__init__()
        # 根据 opt.shuffle 的值选择合适的数据采样器
        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None
        # 创建数据加载器，用于实际加载数据
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,# 批量大小
            shuffle=(train_sampler is None),# 是否进行数据打乱，如果使用了自定义的采样器，则不需要进行 shuffle
            num_workers=opt.workers, # 加载数据时使用的线程数
            pin_memory=True, # 是否将数据存储在固定的内存中，用于 GPU 加速
            sampler=train_sampler# 采样器，用于非随机采样数据
        )
        self.dataset = dataset# 存储数据集对象
        self.data_iter = self.data_loader.__iter__()# 创建数据加载器的迭代器，用于迭代访问数据
       
    def next_batch(self):
        try:
            # 尝试获取数据加载器的下一个批次数据
            batch = self.data_iter.__next__()
        except StopIteration:
            # 如果遍历完所有数据，重新初始化迭代器，再次获取数据的迭代器的下一个批次数据
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")
    # 使用了 argparse 库来解析命令行参数
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=1)
    '''
        dataroot：数据集根目录
        datamode：数据模式（例如，训练或测试）
        stage：阶段（可能是模型训练的不同阶段）
        data_list：数据列表文件名
        fine_width 和 fine_height：图像的宽度和高度
        radius：半径值
        shuffle：是否打乱输入数据
        -b 或 --batch-size：批量大小
        -j 或 --workers：工作线程数
    '''
    # 解析命令行参数后存储在 opt 对象
    opt = parser.parse_args()
    dataset = Dataset(opt)
    data_loader = DataLoader(opt, dataset)
    # 打印数据集和数据加载器的大小（数量）
    print('Size of the dataset: %05d, dataloader: %04d'
            % (len(dataset), len(data_loader.data_loader)))
    # 获取数据集中的第一个样本和数据加载器中的第一个批次数据
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()
    # 进入交互式的IPython环境，允许用户在这个环境中进一步探索和操作数据
    from IPython import embed; embed()

'''
CPDataset 的实例通常负责定义数据集的结构、数据读取方式和数据处理方法等，可以被数据加载器（DataLoader）用于数据的加载和组织
CPDataLoader 的实例通常接收一个数据集对象（例如 CPDataset 的实例），并负责对数据集中的数据进行分批次加载，以便于模型的训练或推断。
它可以定义数据加载的顺序、批次大小、是否打乱数据等参数。
'''
