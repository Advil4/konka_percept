# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.engine.model import Model
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, yaml_model_load
from utils.custom_model import DualBranchYOLOESegModel
from utils.dual_branch_predictor import DualBranchPredictor


class DualBranchModel(Model):
    """
    双分支YOLOE分割模型，与Ultralytics框架兼容。
    """

    def __init__(self, model="zoo_tools/yoloe-11l-merged-seg.yaml", task=None, nc=364) -> None:
        """
        初始化双分支模型。

        Args:
            model (str): 模型配置文件路径
            task (str): 任务类型
            nc (int): 类别数
        """
        self.nc = nc
        super().__init__(model, task)

    @property
    def task_map(self):
        """
        Map head to model, trainer, validator, and predictor classes
        """
        return {
            "segment": {
                "model": DualBranchYOLOESegModel,
                "trainer": None,  # 可以添加自定义训练器
                "validator": None,  # 可以添加自定义验证器
                "predictor": DualBranchPredictor,  # 使用我们的自定义预测器
            }
        }

    def _new(self, cfg: str, task=None, model=None, verbose=True) -> None:
        """
        初始化新模型并从模型定义中推断任务类型。
        """
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)

        # 创建双分支模型而不是默认模型
        self.model = DualBranchYOLOESegModel(
            cfg=cfg,
            nc=self.nc,
            verbose=verbose
        )

        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task

        # Below added to allow export from YAMLs
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine default and model args (prefer model args)
        self.model.task = self.task
        self.model_name = cfg

    def _load(self, weights: str, task: str = None):
        """
        Load a model from weights file or create a new one.

        Args:
            weights (str): Path to the weights file or model name.
            task (str, optional): Task type. Defaults to None.
        """
        if weights.lower().startswith(("https://", "http://")):
            weights = attempt_load_one_weight(weights)[0]  # download and return weight filename

        # Try to load as a YOLO model first
        try:
            super()._load(weights, task)
            return
        except Exception:
            pass

        # If that fails, create a new dual branch model
        self.model = self._new(weights, task)
        self.model.args = {}
        self.model.task = self.task

    def _load_dual_weights(self, general_weights: str, gripper_weights: str):
        """
        加载双分支权重。

        Args:
            general_weights (str): 通用模型权重路径
            gripper_weights (str): 夹爪模型权重路径
        """
        if hasattr(self.model, 'load_dual_weights'):
            self.model.load_dual_weights(general_weights, gripper_weights)
        else:
            print("模型不支持加载双分支权重")

    @property
    def names(self):
        """
        返回类别名称。
        """
        return getattr(self.model, 'names', None) or [f"class_{i}" for i in range(self.nc)]

    @staticmethod
    def _reset_ckpt_args(args):
        """
        Reset arguments when loading a PyTorch model.
        """
        args.pop("device", None)
        args.pop("task", None)
        args.pop("model", None)
        args.pop("imgsz", None)
        args.pop("half", None)
        args.pop("augment", None)
        args.pop("verbose", None)
        args.pop("nc", None)
        args.pop("nc_general", None)
        args.pop("nc_gripper", None)
        args.pop("names", None)
        return args
