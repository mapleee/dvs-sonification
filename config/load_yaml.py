import yaml

configs_name = 'insight'

with open(f'config/{configs_name}.yaml') as f:
    configs = yaml.safe_load(f)
