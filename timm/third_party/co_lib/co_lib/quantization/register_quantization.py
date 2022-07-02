QUANTIZATION_ZOOM = []


def register_quantization(priority):

    def insert(_cls):
        QUANTIZATION_ZOOM.append([_cls, priority])
        return _cls

    return insert
