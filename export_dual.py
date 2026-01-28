from ultralytics.nn.modules import YOLOESegment
from utils.dual_branch_model import DualBranchModel

# 定义类别文本描述
with open("en_4586.txt", "r", encoding="utf-8") as f:
    names = [line.strip() for line in f.readlines()]
names = [name for name in names if name]

# 创建模型
model = DualBranchModel(
    model="trt_engines/yoloe-11l-merged-seg.yaml",
    task="segment",
    nc=len(names)
)

# 先加载权重
model._load_dual_weights(
    general_weights="trt_engines/temp/yoloe-11l-seg.pt",
    gripper_weights="trt_engines/temp/yoloe-11l-gripper-seg.pt"
)

# 详细验证权重是否已加载
model.model.verify_weights_loaded_detailed()

# 设置默认文本描述
general_texts = names[:-1]  # 通用类别文本描述
gripper_texts = names[-1:]  # 夹爪类别文本描述

model.model.set_default_texts(general_texts, gripper_texts)
model.model.eval()

model.model.general_tpe = model.model.get_text_pe(general_texts)
model.model.gripper_tpe = model.model.get_text_pe(gripper_texts)

# 在融合前确保所有参数都是叶节点变量
for param in model.model.parameters():
    if not param.is_leaf:
        param.detach_()

# 融合文本嵌入到模型头部
model.model.fuse()

# 设置模型为导出模式
model.model.export = True

# 导出模型
model.export(format='engine', half=True, device=0, imgsz=640)
