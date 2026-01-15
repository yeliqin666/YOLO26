import sys
import os
import torch
from ultralytics import YOLO

# ============================================================================
# 1. 环境配置 (指向你的 YOLO26 代码库)
# ============================================================================
# 确保这个路径指向你 clone 下来的文件夹根目录
yolo_path = '/root/autodl-tmp/YOLO26'
if yolo_path not in sys.path:
    sys.path.insert(0, yolo_path)

# ============================================================================
# 2. 主训练流程
# ============================================================================
if __name__ == '__main__':
    # 配置文件路径
    yaml_path = '/root/autodl-tmp/YOLO26/ultralytics/cfg/models/26/myyolo26-p2.yaml'
    
    # 检查文件是否存在
    if not os.path.exists(yaml_path):
        print(f"❌ Error: 找不到配置文件 {yaml_path}")
        exit()

    print(f"🚀 Loading custom YOLO26 model: {yaml_path}")
    
    # 使用 YAML 从头构建模型
    model = YOLO(yaml_path)

    print("\n🎯 Starting Training...")
    
    try:
        model.train(
            data='/root/autodl-tmp/mytd.yaml',
            project='runs/detect',
            name='yolo26_p2_rephms_v1',
            
            # 训练参数
            epochs=1000,
            patience=100,
            batch=64,       # P2架构显存占用较大，如果 OOM 请调小到 32 或 16
            imgsz=1024,     # 🚀 小目标检测建议用大图 (1024 或 1280)
            device=[0],     # 只有一个GPU就写[0]，两个写[0,1]
            workers=8,
            
            # 优化器
            # 如果 'MuSGD' 报错 (KeyError/AttributeError)，请改为 'auto' 或 'AdamW'
            optimizer='MuSGD', 
            lr0=0.01,
            
            # ❌ 已移除 'progloss=True' 以修复 SyntaxError
            
            # 数据增强 (针对小目标微调)
            mosaic=1.0,
            mixup=0.1,      # 小目标不宜过高
            scale=0.5,      # 缩放很重要
            erasing=0.4,
            
            # 系统设置
            amp=True,
            save=True,
            plots=True,
            close_mosaic=20
        )
        
        print("\n✅ Training Finished.")
        
        # 尝试导出
        print("📦 Exporting ONNX...")
        model.export(format='onnx', dynamic=True)
        
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误:\n{e}")