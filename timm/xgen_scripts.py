"""
Step 1: Please follow the Xgen ready prepare doc to modify your training script.
Step 2: Send this script to the directory of train_script_main.py
Step 3: Run it

Example : Python xgen.py --xgen-config-path =xxxx/xxx/xx.json --xgen-mode='auto' --xgen-pretrained-model-path=xxx/xxx/xxx.pth
"""
import sys
from xgen_tools import xgen
from xgen_tools.xgen_main import args_parser
from train_script_main import training_main

COMPILER_INSTALL_PATH = '/home/tmp00047/test2/cocogen/'
sys.path.append(COMPILER_INSTALL_PATH)
from cocogen import run
#
# def run(onnx_path, quantized, pruning, output_path, **kwargs):
#     import random
#     res = {}
#     # for simulation
#     pr = kwargs['sp_prune_ratios']
#     num_blocks = kwargs.get('num_blocks', None)
#     res['output_dir'] = output_path
#     if quantized:
#         res['latency'] = 50
#     else:
#         if num_blocks is not None:
#             res['latency'] = 10 * (num_blocks) * (num_blocks)
#         else:
#             res['latency'] = 100 - (pr * 10) * (pr * 10) - random.uniform(0, 10)
#     return res

if __name__ == '__main__':
    # xgen(training_main, run, xgen_config_path="./configs/dense_effnetb4/dense_xgen.json", xgen_mode='scaling')
    xgen(training_main, run, xgen_config_path="./configs/dense_resnet50/dense_xgen.json", xgen_mode='pruning')

