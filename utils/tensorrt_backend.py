import json
import logging
import threading
from collections import namedtuple, OrderedDict
from pathlib import Path
from typing import Union, List, Optional

import numpy as np
import tensorrt as trt
import torch

LOGGER = logging.getLogger(__name__)


class TensorRTBackend(torch.nn.Module):
    """
    Ultralytics 风格的 TensorRT 后端，支持同步/异步 CUDA 推理。
    适用于 Jetson AGX / x86 + GPU 平台，仅支持 TensorRT 10+。
    """

    @torch.no_grad()
    def __init__(
            self,
            weights: Union[str, List[str], torch.nn.Module] = "yolo11n.pt",
            device: torch.device = torch.device("cpu"),
            dnn: bool = False,
            data: Optional[Union[str, Path]] = None,
            fp16: bool = False,
            batch: int = 1,
            fuse: bool = True,
            verbose: bool = True,
            asynchronous: bool = True  # 是否使用异步推理
    ):
        """
        初始化用于推理的 AutoBackend。

        参数:
            weights (str | List[str] | torch.nn.Module): 模型权重文件路径或模块实例。
            device (torch.device): 运行模型的设备。
            dnn (bool): 使用 OpenCV DNN 模块进行 ONNX 推理。
            data (str | Path | optional): 包含类别名称的额外 data.yaml 文件路径。
            fp16 (bool): 启用半精度推理。仅特定后端支持。
            batch (int): 假设的推理批处理大小。
            fuse (bool): 融合 Conv2D + BatchNorm 层以优化。
            verbose (bool): 启用详细日志记录。
            asynchronous (bool): 是否使用异步推理，默认为True。
        """
        super().__init__()
        self.stream = torch.cuda.Stream() if asynchronous else None
        self.lock = threading.Lock()
        self.asynchronous = asynchronous

        w = str(weights[0] if isinstance(weights, list) else weights)
        nn_module = isinstance(weights, torch.nn.Module)

        stride, ch = 32, 3  # 默认步长和通道数
        end2end, dynamic = False, False
        model, metadata, task = None, None, None

        LOGGER.info(f"加载 {w} 用于 TensorRT 推理...")

        if device.type == "cpu":
            device = torch.device("cuda:0")
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        logger = trt.Logger(trt.Logger.INFO)
        # 读取文件
        with open(w, "rb") as f, trt.Runtime(logger) as runtime:
            try:
                meta_len = int.from_bytes(f.read(4), byteorder="little")  # 读取元数据长度
                metadata = json.loads(f.read(meta_len).decode("utf-8"))  # 读取元数据
            except UnicodeDecodeError:
                f.seek(0)  # 引擎文件可能缺少嵌入的 Ultralytics 元数据
            dla = metadata.get("dla", None)
            if dla is not None:
                runtime.DLA_core = int(dla)
            model = runtime.deserialize_cuda_engine(f.read())  # 读取引擎

        # 模型上下文
        try:
            context = model.create_execution_context()
        except Exception as e:  # 模型为 None
            LOGGER.error(f"TensorRT 模型导出使用的版本与 {trt.__version__} 不同\n")
            raise e

        bindings = OrderedDict()
        output_names = []
        fp16 = False  # 默认值将在下方更新
        dynamic = False

        # 仅支持 TensorRT 10+，所以不需要检查版本
        is_trt10 = True
        num = range(model.num_io_tensors)
        for i in num:
            name = model.get_tensor_name(i)
            dtype = trt.nptype(model.get_tensor_dtype(name))
            is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            if is_input:
                if -1 in tuple(model.get_tensor_shape(name)):
                    dynamic = True
                    context.set_input_shape(name, tuple(model.get_tensor_profile_shape(name, 0)[1]))
                if dtype == np.float16:
                    fp16 = True
            else:
                output_names.append(name)
            shape = tuple(context.get_tensor_shape(name))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        batch_size = bindings["images"].shape[0]  # 如果动态，这实际上是最大批处理大小

        if metadata and isinstance(metadata, dict):
            for k, v in metadata.items():
                if k in {"stride", "batch", "channels"}:
                    metadata[k] = int(v)
                elif k in {"imgsz", "names", "kpt_shape", "args"} and isinstance(v, str):
                    metadata[k] = eval(v)
            stride = metadata["stride"]
            task = metadata["task"]
            batch = metadata["batch"]
            imgsz = metadata["imgsz"]
            names = metadata["names"]
            kpt_shape = metadata.get("kpt_shape")
            end2end = metadata.get("args", {}).get("nms", False)
            dynamic = metadata.get("args", {}).get("dynamic", dynamic)
            ch = metadata.get("channels", 3)

        self.__dict__.update(locals())  # 将所有变量赋值给 self

    def forward(self, im: torch.Tensor):
        with self.lock:
            try:
                # 检查输入是否有效
                if im is None or im.numel() == 0:
                    LOGGER.warning("输入张量为空")
                    return [torch.empty(0, device=self.device) for _ in self.output_names]

                if self.dynamic and im.shape != self.bindings["images"].shape:
                    self.context.set_input_shape("images", im.shape)
                    self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                    for name in self.output_names:
                        new_shape = tuple(self.context.get_tensor_shape(name))
                        if new_shape != self.bindings[name].shape:
                            self.bindings[name].data.resize_(new_shape)

                s = self.bindings["images"].shape
                if im.shape != s:
                    LOGGER.warning(f"输入大小 {im.shape} 与模型期望大小 {s} 不匹配")
                    return [torch.empty(0, device=self.device) for _ in self.output_names]

                # 更新输入输出张量地址（TensorRT 10+ 必须）
                self.context.set_tensor_address("images", im.data_ptr())
                for name in self.output_names:
                    self.context.set_tensor_address(name, self.bindings[name].data.data_ptr())

                if self.asynchronous and self.stream is not None:
                    # 异步执行推理并检查结果
                    result = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)

                    if not result:
                        LOGGER.warning("TensorRT 异步推理执行失败")
                        return [torch.empty(0, device=self.device) for _ in self.output_names]

                    # 同步等待推理完成
                    self.stream.synchronize()
                else:
                    # 同步执行推理
                    result = self.context.execute_v2([int(im.data_ptr())] +
                                                     [int(self.bindings[name].data.data_ptr()) for name in
                                                      self.output_names])

                    if not result:
                        LOGGER.warning("TensorRT 同步推理执行失败")
                        return [torch.empty(0, device=self.device) for _ in self.output_names]

                y = [self.bindings[x].data for x in sorted(self.output_names)]

                # 验证输出结果
                if any(x.numel() == 0 for x in y):
                    LOGGER.warning("TensorRT 推理返回空结果")
                    return [torch.empty(0, device=self.device) for _ in y]

                # 确保输出在正确的设备上
                y = [x.to(self.device) for x in y]

                # 特别处理双分支模型的输出，根据形状识别输出类型
                if len(y) >= 4:
                    # 双分支模型有4个或更多输出（检测输出 + 分割输出 + 嵌入特征）
                    outputs = self._organize_dual_branch_outputs_with_embeddings(y)
                    return self.from_numpy(outputs)
                elif len(y) == 4:
                    # 双分支模型有4个输出，需要根据形状进行分类和排序
                    outputs = self._organize_dual_branch_outputs(y)
                    return self.from_numpy(outputs)
                elif isinstance(y, (list, tuple)):
                    if len(self.names) == 999 and (self.task == "segment" or len(y) == 2):
                        nc = y[0].shape[1] - y[1].shape[1] - 4
                        self.names = {i: f"class{i}" for i in range(nc)}
                    return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
                else:
                    return self.from_numpy(y)

            except Exception as e:
                LOGGER.error(f"TensorRT 推理过程中发生异常: {e}", exc_info=True)
                return [torch.empty(0, device=self.device) for _ in self.output_names]

    def _organize_dual_branch_outputs_with_embeddings(self, outputs):
        """
        根据输出形状组织双分支模型的输出，包括嵌入特征
        双分支模型应该有7个输出:
        - 2个检测输出: (1, 399, 8400) 和 (1, 37, 8400)
        - 2个分割输出: (1, 32, 160, 160) 和 (1, 32, 160, 160)
        - 3个嵌入特征:
          * (1, 512, 80, 80) - 层16特征
          * (1, 512, 40, 40) - 层19特征
          * (1, 512, 20, 20) - 层22特征

        返回格式: [general_detect, general_segment, gripper_detect, gripper_segment, embeddings]
        """
        detection_outputs = []
        segmentation_outputs = []
        embedding_outputs = []

        # 根据形状分类输出
        for output in outputs:
            if len(output.shape) == 3:  # 检测输出 (B, C, N)
                detection_outputs.append((output.shape[1], output))  # (通道数, 张量)
            elif len(output.shape) == 4:  # 分割输出或嵌入特征 (B, C, H, W)
                # 根据通道数区分分割输出和嵌入特征
                if output.shape[1] == 32:  # 分割输出有32个通道
                    segmentation_outputs.append(output)
                else:  # 嵌入特征有不同的通道数
                    embedding_outputs.append(output)

        # 按通道数排序检测输出（通道数多的为通用分支）
        detection_outputs.sort(key=lambda x: x[0], reverse=True)

        # 按空间尺寸排序嵌入特征（从小到大）
        embedding_outputs.sort(key=lambda x: x.shape[2] * x.shape[3])  # 按H*W排序

        # 重新组织输出顺序
        organized_outputs = []

        # 添加检测输出（按通道数排序）
        for _, output in detection_outputs:
            organized_outputs.append(output)

        # 添加分割输出（保持原有顺序）
        organized_outputs.extend(segmentation_outputs)

        # 添加嵌入特征
        organized_outputs.append(embedding_outputs)

        # 确保输出顺序为: [general_detect, general_segment, gripper_detect, gripper_segment, embeddings]
        if len(organized_outputs) >= 5:
            general_detect = organized_outputs[0]  # (1, 399, 8400) 通道数多的检测输出
            gripper_detect = organized_outputs[1]  # (1, 37, 8400) 通道数少的检测输出
            general_segment = organized_outputs[3]  # (1, 32, 160, 160) 第一个分割输出
            gripper_segment = organized_outputs[2]  # (1, 32, 160, 160) 第二个分割输出
            embeddings = organized_outputs[4]  # 嵌入特征列表 [(1, 512, 80, 80), (1, 512, 40, 40), (1, 512, 20, 20)]

            return [general_detect, general_segment, gripper_detect, gripper_segment, embeddings]

        return organized_outputs

    def _organize_dual_branch_outputs(self, outputs):
        """
        根据输出形状组织双分支模型的输出
        双分支模型应该有4个输出:
        - 2个检测输出: (1, 399, 8400) 和 (1, 37, 8400)
        - 2个分割输出: (1, 32, 160, 160) 和 (1, 32, 160, 160)

        返回格式: [general_detect, general_segment, gripper_detect, gripper_segment]
        """
        detection_outputs = []
        segmentation_outputs = []

        # 根据形状分类输出
        for output in outputs:
            if len(output.shape) == 3:  # 检测输出 (B, C, N)
                detection_outputs.append((output.shape[1], output))  # (通道数, 张量)
            elif len(output.shape) == 4:  # 分割输出 (B, C, H, W)
                segmentation_outputs.append(output)

        # 按通道数排序检测输出（通道数多的为通用分支）
        detection_outputs.sort(key=lambda x: x[0], reverse=True)

        # 重新组织输出顺序
        organized_outputs = []

        # 添加检测输出（按通道数排序）
        for _, output in detection_outputs:
            organized_outputs.append(output)

        # 添加分割输出（保持原有顺序）
        organized_outputs.extend(segmentation_outputs)

        # 确保输出顺序为: [general_detect, general_segment, gripper_detect, gripper_segment]
        if len(organized_outputs) == 4:
            general_detect = organized_outputs[0]  # 第一个检测输出
            gripper_detect = organized_outputs[1]  # 第二个检测输出
            general_segment = organized_outputs[3]  # 第一个分割输出
            gripper_segment = organized_outputs[2]  # 第二个分割输出

            # 注意：由于两个分割输出形状相同，我们无法区分哪个属于哪个分支
            # 在实际应用中，我们需要在后处理阶段同时使用两个分割输出
            return [general_detect, general_segment, gripper_detect, gripper_segment]

        return organized_outputs

    def from_numpy(self, x):
        """
        将 numpy 数组转换为张量。

        参数:
            x (np.ndarray): 要转换的数组。

        返回:
            (torch.Tensor): 转换后的张量
        """
        try:
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).to(self.device)
            elif isinstance(x, torch.Tensor):
                return x.to(self.device)
            else:
                return x
        except Exception as e:
            LOGGER.warning(f"从 numpy 转换张量时出错: {e}")
            return torch.empty(0, device=self.device)

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        通过运行一次前向传递来预热模型。

        参数:
            imgsz (tuple): 输入张量的形状，格式为 (batch_size, channels, height, width)
        """
        import torchvision  # noqa (在此导入以避免在后处理时间中记录 torchvision 导入时间)

        im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # 输入
        for _ in range(2):  # 多次预热
            self.forward(im)  # 预热
