import tensorflow as tf

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
        self._name_appendix = ""
        self._setttings_initialized = False

        self._generator_base = Generator
        self._train_generator = self._test_generator = None
        self._sample_weights = self._tf_sample_weights = None
        self.n_dim = self.n_class = None
        self.n_random_train_subset = self.n_random_train_subset = None

        if model_param_settings is None:
            self.model_param_settings = {}
        else:
            assert_msg = "model_param_settings should be a dictionary"
            assert isinstance(model_param_settings, dict), assert_msg
            self.model_param_settings = model_param_settings

        self.lr = None
        self._loss = self._loss_name = self._metric_name = None
        self._optimizer_name = self._optimizer = None
        self.n_epoch = self.max_epoch = self.n_iter = self.batch_size = None

        if model_structure_settings is None:
            self.model_structure_settings = {}
        else:
            assert_msg = "model_structure_settings should be a dictionary"
            assert isinstance(model_structure_settings, dict), assert_msg
            self.model_structure_settings = model_structure_settings

        self._model_built = False
        self.py_collections = self.tf_collections = None
        self._define_py_collections()
        self._define_tf_collections()

        self._ws, self._bs = [], []
        self._is_training = tf.placeholder(tf.bool, name='is_training')
        self._loss = self._train_step = None
        self._tfx = self._tfy = self._output = self._prob_output = None

        self._sess = None
        self._graph = tf.Graph()
        self._sess_config = self.model_param_settings.pop('sess_config', None)
