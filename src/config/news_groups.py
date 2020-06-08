from yacs.config import CfgNode as CN

_C = CN()

_C.output_dir = '../outputs'

_C.data = CN()
_C.data.path = "[path to]/data/news_groups"
_C.data.name = "news_groups"
_C.data.split = 'trainval'


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
