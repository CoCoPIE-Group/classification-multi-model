COMPRESSION_ZOOM = []


def register_optimizer(priority):

    def insert(_cls):
        if _cls is not None:
            COMPRESSION_ZOOM.append([_cls, priority])
        return _cls

    return insert
