from copy import deepcopy

import torch

from ultralytics.nn.modules import YOLOESegment
from ultralytics.nn.tasks import YOLOESegModel, yaml_model_load, parse_model
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import feature_visualization


class DualBranchYOLOESegModel(YOLOESegModel):
    """
    双分支YOLOE分割模型，在两个专门的检测头之间共享骨干网络。

    该模型将两个预训练的YOLOE分割模型合并，分别用于通用目标检测（379个类别）和夹爪专用检测（1个类别）。
    骨干网络在两个分支之间共享，同时保持独立的检测头。
    """

    def __init__(self, cfg="zoo_tools/yoloe-11l-merged-seg.yaml", ch=3, nc=380, verbose=True):
        """
        初始化双分支YOLOE分割模型。

        Args:
            cfg (str): 模型配置文件路径。
            ch (int): 输入通道数。
            nc (int): 总类别数。
            verbose (bool): 是否显示模型信息。
        """
        # 存储类别数
        self.nc = int(nc)  # 确保nc是整数
        self.nc_general = self.nc - 1
        self.nc_gripper = 1
        self.general_tpe = None
        self.gripper_tpe = None

        # 更新模型名称
        self.names = {i: f"general_class_{i}" for i in range(self.nc_general)}
        self.names.update({i + self.nc_general: f"gripper_class_{i}" for i in range(self.nc_gripper)})
        self._names = self.names  # 兼容性属性

        # 直接调用BaseModel的初始化，跳过YOLOESegModel的初始化
        from ultralytics.nn.tasks import BaseModel
        BaseModel.__init__(self)

        # 加载模型配置
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # 定义模型
        self.yaml["channels"] = ch  # save channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)

        # 硬编码stride值，避免在初始化时进行前向传播
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        if hasattr(self.model[-1], 'stride'):
            self.model[-1].stride = self.stride
        self.model[-1].inplace = self.inplace

        # 为YOLOESegment层设置stride和strides属性
        for i, layer in enumerate(self.model):
            if isinstance(layer, YOLOESegment):
                # 设置stride属性
                layer.stride = self.stride
                # 设置strides属性（通常是stride的副本）
                if not hasattr(layer, 'strides'):
                    layer.strides = self.stride.clone()

                # 如果需要，可以为每个检测头设置特定的stride值
                if i == 23:  # 第一个检测头
                    layer.stride = self.stride
                    layer.strides = self.stride.clone()
                elif i == 36:  # 第二个检测头
                    layer.stride = self.stride
                    layer.strides = self.stride.clone()

        # 修复YOLOESegment层的nc参数以匹配实际类别数
        # 第一个YOLOESegment层 (索引23) 应该有nc_general个类别
        if len(self.model) > 23 and isinstance(self.model[23], YOLOESegment):
            self.model[23].nc = self.nc_general
            # 更新YOLOESegment层中的no参数 (no = nc + reg_max * 4 + 32)
            if hasattr(self.model[23], 'reg_max'):
                self.model[23].no = self.nc_general + self.model[23].reg_max * 4 + 32  # 32是分割系数
            else:
                self.model[23].no = self.nc_general + 16 * 4 + 32  # 默认reg_max为16

        # 第二个YOLOESegment层 (索引36) 应该有nc_gripper个类别
        if len(self.model) > 36 and isinstance(self.model[36], YOLOESegment):
            self.model[36].nc = self.nc_gripper
            # 更新YOLOESegment层中的no参数 (no = nc + reg_max * 4 + 32)
            if hasattr(self.model[36], 'reg_max'):
                self.model[36].no = self.nc_gripper + self.model[36].reg_max * 4 + 32  # 32是分割系数
            else:
                self.model[36].no = self.nc_gripper + 16 * 4 + 32  # 默认reg_max为16

        # 初始化权重和偏置
        from ultralytics.utils.torch_utils import initialize_weights
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def predict(
            self, x, profile=False, visualize=False, general_tpe=None, gripper_tpe=None, augment=False, embed=None,
            vpe=None, return_vpe=False, return_feature_indices=False  # 添加新参数
    ):
        """
        执行双分支模型的前向传播。

        Args:
            x (torch.Tensor): 输入张量。
            profile (bool): 是否分析每层的计算时间。
            visualize (bool): 是否保存特征图用于可视化。
            general_tpe (torch.Tensor, optional): 通用类别文本位置嵌入。
            gripper_tpe (torch.Tensor, optional): 夹爪类别文本位置嵌入。
            augment (bool): 是否在推理期间执行数据增强。
            embed (list, optional): 要返回的特征向量/嵌入列表。
            vpe (torch.Tensor, optional): 视觉位置嵌入。
            return_vpe (bool): 是否返回视觉位置嵌入。
            return_feature_indices (bool): 是否返回特征索引信息。

        Returns:
            (torch.Tensor): 模型输出张量。
        """
        y, dt, embeddings = [], [], []  # outputs
        b = x.shape[0]

        # 存储检测头输出
        detection_outputs = []
        # 存储特征
        head_features = []

        for m_idx, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)

            if isinstance(m, YOLOESegment):
                # 分别为两个YOLOESegment层提供相应的文本嵌入
                if m.i == 23:  # 第一个检测头，处理通用类别
                    layer_features = m.get_features_before_fuse(x)
                    head_features.append(layer_features)

                    vpe = m.get_vpe(x, vpe) if vpe is not None else None
                    if return_vpe:
                        assert vpe is not None
                        assert not self.training
                        return vpe
                    # 使用通用类别的文本嵌入
                    if general_tpe is not None:
                        cls_pe = general_tpe
                        if cls_pe.shape[0] != b or m.export:
                            cls_pe = cls_pe.expand(b, -1, -1)
                        x = m(x, cls_pe)
                    else:
                        # 如果没有文本嵌入，使用随机嵌入
                        random_text = torch.randn(b, self.nc_general, 512, device=x[0].device, dtype=x[0].dtype)
                        cls_pe = self.get_cls_pe(m.get_tpe(random_text), vpe).to(device=x[0].device, dtype=x[0].dtype)
                        if cls_pe.shape[0] != b or m.export:
                            cls_pe = cls_pe.expand(b, -1, -1)
                        x = m(x, cls_pe)
                    detection_outputs.append(x)  # 保存第一个检测头的输出
                elif m.i == 36:  # 第二个检测头，处理夹爪类别
                    vpe = m.get_vpe(x, vpe) if vpe is not None else None
                    if return_vpe:
                        assert vpe is not None
                        assert not self.training
                        return vpe
                    # 使用夹爪类别的文本嵌入
                    if gripper_tpe is not None:
                        cls_pe = gripper_tpe
                        if cls_pe.shape[0] != b or m.export:
                            cls_pe = cls_pe.expand(b, -1, -1)
                        x = m(x, cls_pe)
                    else:
                        # 如果没有文本嵌入，使用随机嵌入
                        random_text = torch.randn(b, self.nc_gripper, 512, device=x[0].device, dtype=x[0].dtype)
                        cls_pe = self.get_cls_pe(m.get_tpe(random_text), vpe).to(device=x[0].device, dtype=x[0].dtype)
                        if cls_pe.shape[0] != b or m.export:
                            cls_pe = cls_pe.expand(b, -1, -1)
                        x = m(x, cls_pe)
                    detection_outputs.append(x)  # 保存第二个检测头的输出
                else:
                    # 其他YOLOESegment层使用默认处理
                    x = m(x)  # 使用默认处理
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)

        # 将head特征添加到检测输出中，替代原来的embeddings
        if head_features:
            # 将head_features扁平化并添加到detection_outputs中
            flattened_features = []
            for branch_features in head_features:
                for layer_feature in branch_features:
                    flattened_features.append(layer_feature)
            detection_outputs.append(flattened_features)

        # 在导出模式下，直接返回所有输出
        if getattr(self, 'export', False):
            # 返回检测输出和嵌入特征
            if len(detection_outputs) >= 2:
                # 如果有嵌入特征，将其展平到主输出列表中
                outputs = []
                # 添加检测头输出
                for output in detection_outputs[:-1]:
                    if isinstance(output, (list, tuple)):
                        outputs.extend(output)
                    else:
                        outputs.append(output)
                # 添加嵌入特征
                if isinstance(detection_outputs[-1], list):
                    outputs.extend(detection_outputs[-1])
                else:
                    outputs.append(detection_outputs[-1])

                # 确保所有输出都是独立的张量，避免共享存储
                detached_outputs = []
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        detached_outputs.append(output.contiguous())
                    else:
                        detached_outputs.append(output)

                return tuple(detached_outputs) if len(detached_outputs) > 1 else detached_outputs[0]

        # 在推理模式下，也返回两个分支的独立输出和嵌入特征
        if len(detection_outputs) >= 2:
            # 返回两个分支的独立输出和嵌入特征
            return tuple(detection_outputs)

        # 默认返回
        return x

    def set_default_texts(self, general_texts=None, gripper_texts=None):
        """
        设置默认的文本描述，用于标准predict方法。

        Args:
            general_texts (list, optional): 通用类别的文本描述列表。
            gripper_texts (list, optional): 夹爪类别的文本描述列表。
        """
        self.default_general_texts = general_texts
        self.default_gripper_texts = gripper_texts

    def forward(self, x, tpe=None, augment=False, visualize=False, embed=None, vpe=None, return_vpe=False):
        """
        通过模型的前向传播。

        Args:
            x (torch.Tensor): 输入张量。
            tpe (torch.Tensor, optional): 文本位置嵌入。
            augment (bool): 是否使用augmentation进行推理。
            visualize (bool): 是否可视化特征图。
            embed (list[int], optional): 要返回的嵌入层索引。
            vpe (torch.Tensor, optional): 视觉位置嵌入。
            return_vpe (bool): 是否返回视觉位置嵌入。

        Returns:
            (tuple): 模型输出。
        """
        general_tpe = self.general_tpe
        gripper_tpe = self.gripper_tpe

        return self.predict(x, profile=False, visualize=visualize, general_tpe=general_tpe, gripper_tpe=gripper_tpe,
                            augment=augment, embed=embed, vpe=vpe, return_vpe=return_vpe)

    def load_dual_weights(self, general_model_path, gripper_model_path, verbose=True):
        """
        从两个独立的预训练模型加载权重。

        Args:
            general_model_path (str): 通用模型权重文件路径。
            gripper_model_path (str): 夹爪模型权重文件路径。
            verbose (bool): 是否显示加载进度信息。
        """
        from ultralytics.nn.tasks import attempt_load_one_weight

        # 加载通用模型
        try:
            general_model, _ = attempt_load_one_weight(general_model_path)
            general_csd = general_model.float().state_dict()  # checkpoint state_dict as FP32
        except Exception as e:
            LOGGER.error(f"Error loading general model: {e}")
            return False

        # 加载夹爪模型
        try:
            gripper_model, _ = attempt_load_one_weight(gripper_model_path)
            gripper_csd = gripper_model.float().state_dict()  # checkpoint state_dict as FP32
        except Exception as e:
            LOGGER.error(f"Error loading gripper model: {e}")
            return False

        # 获取当前模型状态字典
        model_state_dict = self.model.state_dict()

        # 创建新的状态字典用于加载
        new_state_dict = model_state_dict.copy()

        # 处理权重字典键，移除可能的"model."前缀
        def strip_model_prefix(state_dict):
            """移除state_dict键中的'model.'前缀"""
            stripped_dict = {}
            for key, value in state_dict.items():
                new_key = key[6:] if key.startswith('model.') else key
                stripped_dict[new_key] = value
            return stripped_dict

        general_csd = strip_model_prefix(general_csd)
        gripper_csd = strip_model_prefix(gripper_csd)

        # 加载骨干网络权重 (共享层，索引0-10)
        backbone_layers = list(range(11))
        backbone_loaded = 0
        for key in model_state_dict.keys():
            layer_index = int(key.split('.')[0]) if key.split('.')[0].isdigit() else -1
            if layer_index in backbone_layers:
                # 优先从通用模型加载骨干网络权重
                if key in general_csd and general_csd[key].shape == model_state_dict[key].shape:
                    new_state_dict[key] = general_csd[key]
                    backbone_loaded += 1
                # 如果通用模型中没有，则尝试从夹爪模型加载
                elif key in gripper_csd and gripper_csd[key].shape == model_state_dict[key].shape:
                    new_state_dict[key] = gripper_csd[key]
                    backbone_loaded += 1
                elif verbose:
                    LOGGER.warning(f"Backbone layer {key} not found in either model or shape mismatch")

        if verbose:
            LOGGER.info(f"成功加载 {backbone_loaded} 个骨干网络层权重")

        # 加载通用检测头权重 (层11-23)
        general_head_layers = list(range(11, 24))
        general_loaded = 0
        for key in model_state_dict.keys():
            layer_index = int(key.split('.')[0]) if key.split('.')[0].isdigit() else -1
            if layer_index in general_head_layers:
                # 特殊处理YOLOESegment层的cv6参数加载
                # 在加载通用检测头权重部分替换原有的cv6处理代码
                if isinstance(self.model[layer_index], YOLOESegment) and '.cv6.' in key:
                    # 更安全地解析键结构
                    parts = key.split('.')
                    try:
                        # 确保键格式正确
                        if len(parts) >= 3 and parts[1] == 'cv6':
                            # 构造对应的cv3键
                            cv3_parts = parts[:1] + ['cv3'] + parts[2:]
                            cv3_key = '.'.join(cv3_parts)

                            if cv3_key in general_csd and general_csd[cv3_key].shape == model_state_dict[key].shape:
                                new_state_dict[key] = general_csd[cv3_key]
                                general_loaded += 1
                                continue

                            # 处理BN层参数映射
                            if len(parts) >= 3 and parts[2].startswith('3.'):  # BN层参数
                                # 映射到cv4.bn.norm
                                bn_part = parts[2][2:]  # 去掉'3.'前缀
                                cv4_key = f"{parts[0]}.cv4.norm.{bn_part}"

                                if cv4_key in general_csd and general_csd[cv4_key].shape == model_state_dict[key].shape:
                                    new_state_dict[key] = general_csd[cv4_key]
                                    general_loaded += 1
                                    continue
                    except (ValueError, IndexError) as e:
                        if verbose:
                            LOGGER.warning(f"Error parsing cv6 key {key}: {e}")
                        pass

                # 普通参数加载
                if key in general_csd and general_csd[key].shape == model_state_dict[key].shape:
                    new_state_dict[key] = general_csd[key]
                    general_loaded += 1
                elif verbose:
                    LOGGER.warning(f"General head layer {key} not found in general model or shape mismatch")

        if verbose:
            LOGGER.info(f"成功加载 {general_loaded} 个通用检测头层权重")

        # 加载夹爪检测头权重 (层24-36)
        gripper_head_layers = list(range(24, 37))
        gripper_loaded = 0
        for key in model_state_dict.keys():
            layer_index = int(key.split('.')[0]) if key.split('.')[0].isdigit() else -1
            if layer_index in gripper_head_layers:
                # 将夹爪模型的权重映射到对应的层
                source_key = key
                # 将层索引从24-36映射到11-23
                parts = key.split('.')
                if parts[0].isdigit():
                    layer_idx = int(parts[0])
                    if 24 <= layer_idx <= 36:
                        source_layer = layer_idx - 13  # 映射到11-23
                        source_key = f"{source_layer}.{'.'.join(parts[1:])}"

                # 检查直接映射是否匹配
                if source_key in gripper_csd and gripper_csd[source_key].shape == model_state_dict[key].shape:
                    new_state_dict[key] = gripper_csd[source_key]
                    gripper_loaded += 1
                else:
                    # 查找具有相似结构的键
                    found = False
                    for gripper_key, gripper_value in gripper_csd.items():
                        # 检查键是否相似（去掉层索引前缀后比较）
                        if (gripper_key.endswith('.'.join(parts[2:])) and
                                gripper_value.shape == model_state_dict[key].shape):
                            new_state_dict[key] = gripper_value
                            gripper_loaded += 1
                            found = True
                            if verbose:
                                LOGGER.info(f"Found alternative mapping: {key} <- {gripper_key}")
                            break

                    # 如果仍然找不到匹配项，对于分割相关层，尝试使用通用模型的权重
                    if not found and any(
                            keyword in key for keyword in ['segment', 'savpe', 'proto', 'cv2', 'cv3', 'cv4', 'cv5']):
                        # 对于分割层，如果夹爪模型中没有对应的权重，使用通用模型的权重
                        if key in general_csd and general_csd[key].shape == model_state_dict[key].shape:
                            new_state_dict[key] = general_csd[key]
                            gripper_loaded += 1
                            found = True
                            if verbose:
                                LOGGER.info(f"Using general model weights for segmentation layer: {key}")

                    if not found and verbose:
                        LOGGER.warning(
                            f"Gripper head layer {key} (mapped from {source_key}) not found in gripper model or shape mismatch")

        if verbose:
            LOGGER.info(f"成功加载 {gripper_loaded} 个夹爪检测头层权重")

        # 加载更新后的状态字典
        try:
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            if verbose:
                LOGGER.info(f"成功从 {general_model_path} 和 {gripper_model_path} 加载双权重")
                if missing_keys:
                    LOGGER.info(f"Missing keys: {len(missing_keys)}")
                    # 打印部分缺失的键以供调试
                    for i, key in enumerate(missing_keys):
                        if i < 10:  # 只打印前10个
                            LOGGER.info(f"  Missing key: {key}")
                        else:
                            LOGGER.info(f"  ... and {len(missing_keys) - 10} more")
                            break
                if unexpected_keys:
                    LOGGER.info(f"Unexpected keys: {len(unexpected_keys)}")

            # 同步所有YOLOESegment层中的cv6权重
            for m in self.model.modules():
                if isinstance(m, YOLOESegment):
                    m.sync_cv6_weights()

            return True
        except Exception as e:
            LOGGER.error(f"加载权重时出错: {e}")
            # 尝试非严格模式加载
            try:
                self.model.load_state_dict(new_state_dict, strict=False)
                if verbose:
                    LOGGER.info("使用非严格模式加载权重")

                # 同步所有YOLOESegment层中的cv6权重
                for m in self.model.modules():
                    if isinstance(m, YOLOESegment):
                        m.sync_cv6_weights()

                return True
            except Exception as e2:
                LOGGER.error(f"加载权重失败: {e2}")
                return False

    def fuse(self, verbose=True):
        """
        Fuse prompt embeddings to model heads to enable prompt-free inference.

        Args:
            verbose (bool): Whether to print fusion information.

        Returns:
            (DualBranchYOLOESegModel): The fused model.
        """
        super().fuse(verbose=verbose)

        # 获取默认文本
        general_texts = getattr(self, 'default_general_texts', None)
        gripper_texts = getattr(self, 'default_gripper_texts', None)

        # 为两个分支生成文本嵌入
        if general_texts is not None:
            general_tpe = self.get_text_pe(general_texts)
        else:
            # 使用默认的通用类别文本
            default_general_texts = [f"object_{i}" for i in range(self.nc_general)]
            general_tpe = self.get_text_pe(default_general_texts)

        if gripper_texts is not None:
            gripper_tpe = self.get_text_pe(gripper_texts)
        else:
            # 使用默认的夹爪类别文本
            default_gripper_texts = ["gripper"]
            gripper_tpe = self.get_text_pe(default_gripper_texts)

        # 为每个YOLOESegment头融合文本嵌入
        for i, m in enumerate(self.model):
            if isinstance(m, YOLOESegment):
                if i == 23:  # 第一个检测头，处理通用类别
                    # 融合通用类别文本嵌入
                    if hasattr(m, 'fuse'):
                        m.fuse(general_tpe)
                elif i == 36:  # 第二个检测头，处理夹爪类别
                    # 融合夹爪类别文本嵌入
                    if hasattr(m, 'fuse'):
                        m.fuse(gripper_tpe)

        if verbose:
            LOGGER.info("Fused prompt embeddings to model heads.")

        return self

    def warmup(self, imgsz) -> None:
        """
        Warm up the model by running one forward pass with a dummy input.

        Args:
            imgsz (tuple): The shape of the dummy input tensor in the format (batch_size, channels, height, width)
        """
        import torchvision  # noqa (import here so torchvision import time not recorded in postprocess time)

        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton, self.nn_module
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):
                self.forward(im)  # warmup

    def verify_weights_loaded_detailed(self):
        """
        详细验证模型权重是否已正确加载，特别是特殊模块
        """
        LOGGER.info("=== 详细权重验证 ===")

        # 检查关键层的权重
        key_layers = {
            "骨干网络层0 (Conv)": 0,
            "骨干网络层3 (Conv)": 3,
            "骨干网络层10 (C2PSA)": 10,
            "通用检测头层13 (C3k2)": 13,
            "通用检测头层23 (YOLOESegment)": 23,
            "夹爪检测头层26 (C3k2)": 26,
            "夹爪检测头层36 (YOLOESegment)": 36
        }

        all_zero = True
        for layer_name, layer_idx in key_layers.items():
            if layer_idx < len(self.model):
                layer = self.model[layer_idx]

                # 检查各种可能的权重
                weight_sum = 0
                weight_count = 0

                # 特殊处理YOLOESegment层
                if isinstance(layer, YOLOESegment):
                    # 检查cv3模块
                    if hasattr(layer, 'cv3'):
                        for p in layer.cv3.parameters():
                            weight_sum += torch.sum(torch.abs(p)).item()
                            weight_count += 1

                    # 检查cv4模块
                    if hasattr(layer, 'cv4'):
                        for p in layer.cv4.parameters():
                            weight_sum += torch.sum(torch.abs(p)).item()
                            weight_count += 1

                    # 检查cv6模块
                    if hasattr(layer, 'cv6'):
                        for p in layer.cv6.parameters():
                            weight_sum += torch.sum(torch.abs(p)).item()
                            weight_count += 1

                # 检查卷积层权重
                if hasattr(layer, 'conv') and hasattr(layer.conv, 'weight'):
                    weight_sum += torch.sum(torch.abs(layer.conv.weight)).item()
                    weight_count += 1
                elif hasattr(layer, 'cv1') and hasattr(layer.cv1, 'conv') and hasattr(layer.cv1.conv, 'weight'):
                    weight_sum += torch.sum(torch.abs(layer.cv1.conv.weight)).item()
                    weight_count += 1
                elif hasattr(layer, 'm') and hasattr(layer.m, '0') and hasattr(layer.m[0], 'cv1'):
                    # 检查模块列表中的权重
                    try:
                        for submodule in layer.m:
                            if hasattr(submodule, 'cv1') and hasattr(submodule.cv1, 'conv') and hasattr(
                                    submodule.cv1.conv, 'weight'):
                                weight_sum += torch.sum(torch.abs(submodule.cv1.conv.weight)).item()
                                weight_count += 1
                    except:
                        pass

                # 检查其他可能的权重
                if hasattr(layer, 'weight') and layer.weight is not None:
                    weight_sum += torch.sum(torch.abs(layer.weight)).item()
                    weight_count += 1

                if weight_count > 0 and weight_sum > 0:
                    all_zero = False
                    LOGGER.info(f"{layer_name} 权重已加载，权重和: {weight_sum:.4f}")
                elif weight_count > 0:
                    LOGGER.info(f"{layer_name} 权重为零")
                else:
                    LOGGER.info(f"{layer_name} 未找到可检查的权重")

        if all_zero:
            LOGGER.warning("警告: 所有检查的层权重都为零，可能权重未正确加载")
        else:
            LOGGER.info("模型权重已成功加载")

        return not all_zero
