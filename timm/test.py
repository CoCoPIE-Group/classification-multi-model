if __name__ == '__main__':
    import json
    from xgen.xgen_run import xgen
    from train_script_main import training_main

    json_path = 'efficientnet_config/xgen.json'

    #if you are using new config
    from xgen.utils.args_ai_map import get_old_config
    old_json_path = 'efficientnet_config/xgen.json'
    with open(json_path) as f:
        new = json.load(f)

    old = get_old_config(new)
    with open(old_json_path, 'w') as f:
        json.dump(old, f)
    # using old patn instead of the new version
    json_path = old_json_path

    # json_path = 'args_ai_template_sgpu.json'

    def run(onnx_path, quantized, pruning, output_path, **kwargs):
        import random
        res = {}
        # for simulation
        pr = kwargs['sp_prune_ratios']
        res['output_dir'] = output_path

        res['latency'] = 50

        return res

    xgen(training_main, run, xgen_config_path=json_path, xgen_mode='customization')