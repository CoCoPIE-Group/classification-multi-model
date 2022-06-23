import yaml

dic = {}
filename = "prune_config_effnetb0_nu0.6.yaml"
with open(filename) as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in configs['prune_ratios'].items():
        name = k.split(',')[0]
        sparsity = float(1 - list(v.values())[0])
        dic[name] = sparsity

ans = {'prune_ratios': dic}

with open(filename, 'w') as outfile:
    yaml.dump(ans, outfile, default_flow_style=False)