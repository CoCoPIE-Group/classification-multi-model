PRUNING_ZOOM = []


def register_pruning(priority):

    def insert(pruning_cls):
        PRUNING_ZOOM.append([pruning_cls, priority])
        return pruning_cls

    return insert
