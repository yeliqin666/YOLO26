import sys
import torch
from ultralytics import YOLO

# 确保 YOLO26 库在路径中 (根据你的环境保留)
sys.path.insert(0, '/root/autodl-tmp/yolov12')

if __name__ == '__main__':
    # 1. 加载模型
    # 推荐: 使用 YOLO26n.pt 进行迁移学习，它会自动包含架构和预训练权重
    # 如果想从零训练，改用 'yolo26.yaml'
    print("🚀 Loading YOLO26 model (End-to-End, NMS-Free)...")
    model = YOLO("yolo26n.pt")  

    # 2. 开始训练
    print("\n🎯 Starting Training with MuSGD...")
    model.train(
        # 基础配置
        data='/root/autodl-tmp/mytd.yaml',
        project='runs/detect',
        name='yolo26_train_v1',
        epochs=1000,
        patience=100,
        batch=100,
        imgsz=800,
        device=[0, 1],
        workers=12,
        
        # YOLO26 核心策略
        optimizer='MuSGD',   # YOLO26 专属优化器
        lr0=0.01,            # 配合 MuSGD 的初始学习率
        progloss=True,       # 启用渐进式 Loss 平衡
        amp=True,            # 混合精度

        # 数据增强 (保留原有配置)
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        erasing=0.4,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        
        # 验证与保存
        close_mosaic=20,     # 最后20轮关闭马赛克增强
        save=True,
        plots=True
    )
    
    # 3. 导出模型 (推荐 ONNX 或 TensorRT 用于部署)
    print("\n📦 Exporting model...")
    path = model.export(format="onnx", dynamic=True)
