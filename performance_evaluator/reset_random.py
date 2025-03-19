def reset_random():
    import os
    seed = 1
    os.environ['PYTHONHASHSEED'] = str(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import warnings
    warnings.filterwarnings('ignore', category=Warning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    try:
        import tensorflow as tf
        tf.compat.v1.random.set_random_seed(seed)
        tf.compat.v1.set_random_seed(seed)
        tf.compat.v1.disable_eager_execution()
    except ImportError:
        pass
