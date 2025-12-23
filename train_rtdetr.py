import sys

# 添加'ultralytics'目录到系统路径，确保可以正确导入RTDETR模块
sys.path.append("ultralytics")
from ultralytics import RTDETR

if __name__ == '__main__':
    # 1. 加载模型
    # 加载预训练的RT-DETR模型权重。'rtdetr-x.pt'是模型文件路径，根据你的文件位置可能需要调整。
    model = RTDETR('rtdetr-l.pt')
    # model = RTDETR(r"Z:\网盘下载\\ultralytics-main\\ultralytics\cfg\models\\rt-detr\\rtdetr-x.yaml")
    # 2. 开始模型训练
    model.train(
        # 2.1 基础数据与训练配置
        data=r'Z:\网盘下载\RT-DETR-main\rtdetr_pytorch\demo\A_my_data.yaml',  # 数据集配置文件路径，YAML格式，定义了数据路径、类别数等。
        epochs=1000,
        imgsz=640,  # 输入模型的图像尺寸，会被自动调整为最接近的32的倍数（如640）
        batch=4,  # 每个批次的图像数量（实际批量大小）。根据GPU显存调整，太小可能不稳定。
        patience=50,  # 早停耐心值。如果验证集指标在连续50个epoch内未提升，则提前终止训练。
        save=True,  # 是否保存训练过程中的最佳模型和最后一个epoch的模型
        save_period=-1,  # 保存中间检查点的周期（每N个epoch保存一次）。-1表示不按周期保存，仅保存最佳和最后模型。
        cache=False,  # 是否将数据集缓存到内存或磁盘以加速训练。True可加速但占用大量内存/磁盘。
        device='',  # 训练设备。留空（''）表示自动选择；可指定如 '0'（第1块GPU）、'0,1'（多卡）、'cpu'。
        workers=2,  # 数据加载的子进程数。Windows系统下多进程可能有问题，常设为0；Linux下可增加以加速数据读取。
        project='runs/train',  # 训练日志和结果的保存根目录
        name='exp',  # 实验名称，本次训练的结果会保存在 `project/name` 目录下
        exist_ok=False,  # 如果 `project/name` 目录已存在，是否允许覆盖。False会创建新的递增目录（如exp2）。
        pretrained=True,  # 是否从预训练权重开始训练。当前模型已加载‘rtdetr-x.pt’，此参数可能被覆盖。
        resume=False,  # 是否从上次的检查点（如 `project/name` 目录下的 `last.pt`）恢复训练。

        # 2.2 优化器与训练策略
        optimizer='Adam',  # 优化器类型。可选 'SGD', 'Adam', 'AdamW' 等。
        cos_lr=True,  # 是否使用余弦退火（Cosine Annealing）学习率调度器，使学习率呈余弦曲线下降。
        lr0=0.0001,  # 初始学习率
        lrf=0.0001,  # 最终学习率因子（final learning rate factor）。最终学习率 = lr0 * lrf。
        momentum=0.937,  # SGD优化器的动量参数，Adam优化器不使用此参数。
        weight_decay=0.0001,  # 权重衰减系数，用于L2正则化，防止过拟合。
        warmup_epochs=5.0,  # 学习率预热的轮数，在开始时从低学习率逐渐升至lr0，有助于稳定训练。
        warmup_momentum=0.8,  # 预热阶段使用的动量值。
        warmup_bias_lr=0.1,  # 预热阶段偏置（bias）参数的学习率。
        label_smoothing=0.0,  # 标签平滑系数。通过软化标签（如0.95而非1）来减轻模型过拟合和过度自信。
        amp=True,  # 是否启用自动混合精度（Automatic Mixed Precision）训练，可节省显存并可能加速训练。
        deterministic=False,  # 是否启用确定性模式，使训练可完全复现（可能降低训练速度）。
        seed=0,  # 全局随机种子，用于固定数据洗牌、初始化等，确保结果可复现。

        # 2.3 损失函数相关（需要特别注意参数名）
        pose=12.0,  # 【注意：此参数名可能非标准】在目标检测中，这很可能指代边界框回归损失的权重（应为`box`损失）。
        # 官方YOLOv8中分类、检测、分割任务的损失权重参数通常是 `cls`， `box`， `dfl`。
        # 建议查阅RT-DETR源码或尝试修改为 `box_loss=12.0` 或 `cls_loss=12.0`。
        kobj=1.0,  # 【注意：此参数名可能非标准】可能表示关键点或目标存在性损失的权重。

        # 2.4 批处理与归一化
        nbs=64,  # 名义批量大小（Nominal Batch Size）。当实际 `batch` 较小时，通过梯度累积模拟此大批量进行归一化。
        single_cls=False,  # 是否将所有类别视为单一类别进行训练。适用于仅检测“是否有物体”的场景。
        rect=False,  # 是否使用矩形训练（Rectangular Training）。将批次内的图像统一为相同长宽比，减少填充，加速训练。

        # 2.5 数据增强（像素级与空间级）
        hsv_h=0.015,  # HSV色彩空间增强：色调（Hue）扰动幅度（比例）
        hsv_s=0.7,  # HSV色彩空间增强：饱和度（Saturation）扰动幅度（比例）
        hsv_v=0.4,  # HSV色彩空间增强：明度（Value）扰动幅度（比例）
        degrees=0.0,  # 图像随机旋转的最大角度（正负范围）
        translate=0.1,  # 图像随机水平/垂直平移的最大比例（相对于图像尺寸）
        scale=0.5,  # 图像随机缩放的范围（例如0.5表示在0.5倍到1.5倍之间缩放）
        shear=0.0,  # 图像随机剪切变换的最大角度（正负范围）
        perspective=0.0,  # 图像随机透视变换的强度（0-1范围）
        flipud=0.0,  # 图像随机上下翻转的概率
        fliplr=0.5,  # 图像随机左右翻转的概率（目标检测中常用0.5）

        # 2.6 高级数据增强（会显著增加计算负载）
        mosaic=0.0,  # 使用马赛克增强的概率（将4张图拼接为1张）。设为0.0表示关闭，常用于训练初期或小数据集。
        mixup=0.0,  # 使用MixUp增强的概率（混合两张图像和标签）。同样会显著增加计算复杂度。
        copy_paste=0.0,  # 使用复制-粘贴增强的概率（主要应用于实例分割任务，复制一个物体粘贴到图像中）。
        close_mosaic=0,  # 在训练结束前 N 个epoch关闭mosaic增强。0通常表示全程不关闭或未启用。

        # 2.7 其他
        fraction=1.0,  # 训练时使用的数据集比例。1.0表示使用全部数据，可用于快速测试。
        profile=False,  # 是否在训练时进行性能分析（profiling），用于调试速度瓶颈。
        verbose=True,  # 是否在控制台打印详细的训练进度和日志信息。
    )
