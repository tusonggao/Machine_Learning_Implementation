class Base:
    signature = "Base"

    """
        初始化结构
        name: 模型的名字
        model_param_settings: 管理“模型超参数”的字典
        model_structure_settings: 管理“结构超参数”的字典
    """

    def __init__(self, name=None, model_param_settings=None,
                 model_structure_settings=None):
        self.log = {}
        self._name = name