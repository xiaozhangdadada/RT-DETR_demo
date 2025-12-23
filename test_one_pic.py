import sys

# 添加ultralytics到系统路径，确保可以导入RTDETR模块
sys.path.append("ultralytics")
from ultralytics import RTDETR

if __name__ == '__main__':
    # 加载预训练的RT-DETR模型，'rtdetr-x.pt'是模型权重文件
    model = RTDETR('rtdetr-x.pt')

    # 使用模型进行预测
    model.predict(
        source=r'test.jpg',  # 输入源：测试图片路径，可以是单张图片、视频、文件夹或URL
        save=True,  # 是否保存预测结果（图片/视频）
        imgsz=640,  # 输入图片大小，可以是整数（如640）或元组（如(640, 480)）
        conf=0.80,  # 目标检测置信度阈值，高于此值的检测框才会被保留（范围0-1）
        iou=0.80,  # NMS（非极大值抑制）的IoU阈值，用于去除重叠的检测框（范围0-1）
        show=False,  # 是否实时显示预测结果（GUI显示）
        project='runs/predict',  # 保存预测结果的根目录
        name='exp',  # 实验名称，结果会保存在project/name目录下
        save_txt=False,  # 是否将结果保存为YOLO格式的.txt文件（每行：类别 x_center y_center width height）
        save_conf=True,  # 在保存的.txt文件中是否包含置信度分数
        save_crop=False,  # 是否将检测到的目标裁剪保存为单独的图片
        show_labels=True,  # 在结果图片中是否显示类别标签
        show_conf=True,  # 在结果图片中是否显示置信度分数
        vid_stride=1,  # 视频预测时的帧采样间隔（每隔多少帧处理一帧）
        line_width=3,  # 预测框的线条宽度（像素）
        visualize=False,  # 是否可视化模型特征图（用于调试）
        augment=False,  # 是否使用数据增强进行推理（TTA：测试时增强）
        agnostic_nms=False,  # 是否使用类别无关的NMS（将所有类别一起处理）
        retina_masks=False,  # 是否使用高分辨率的分割掩码（仅适用于分割任务）
        show_boxes=True  # 是否显示边界框
    )