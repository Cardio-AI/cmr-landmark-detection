import os


def show_available_gpus():
    """
    Prints all GPUs and their currently BytesInUse
    Returns the first GPU instance with less than 1280 bytes in use
    From our tests that indicates that this Gpu is available
    Usage:
    gpu = get_available_gpu()
    with tf.device(gpu):
       results = model.fit()
       ...
       
    """

    # get a list of all visible GPUs

    import os
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.python.client import device_lib

    gpus = [x for x in device_lib.list_local_devices() if x.device_type == 'GPU']

    [print('available GPUs: no:{} --> {}'.format(i, gpu.physical_device_desc)) for i, gpu in enumerate(gpus)]
    print('-----')
    return gpus


def show_free_gpus(gpus):
    import os
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.python.client import device_lib
    from tensorflow.contrib.memory_stats import BytesInUse


    bytes_used = {}

    for gpu in gpus:
        with tf.device(gpu.name):
            bytes_in_use = BytesInUse()
            bytes_used[gpu.name] = K.get_session().run(bytes_in_use)

    [print('available GPUs: no:{} --> {}, current load: {}'.format(i, gpu.physical_device_desc, bytes_used[gpu.name]))
     for i, gpu in enumerate(gpus)]
    # filter GPUs with more RAM usage than 1280 bytes
    available_gpus = [gpu for gpu in gpus if bytes_used[gpu.name] <= 1280]

    if len(available_gpus) >= 1:
        selected_gpu = available_gpus[0]
        print('Selected GPU: {}, {}'.format(selected_gpu.name, selected_gpu.physical_device_desc))
        # os.environ["CUDA_VISIBLE_DEVICES"]=str(selected_gpu.name[-1])
        return '/device:GPU:0'
    else:
        print('No GPU available!!!')


def choose_gpu_by_id(gpu_id='0'):
    """
    define the visible GPUs returns the current GPU
     which could be used for "with.tf.device(current_gpu)
    :param gpu_id: GPU id provided by the system as string e.g. '0'
    :return: current_gpu (str)
    """

    from tensorflow.python.client import device_lib

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    gpu_list = ["/gpu:%d" % i for i in range(len(gpu_id.split(',')))]
    # print(device_lib.list_local_devices())
    return gpu_list
