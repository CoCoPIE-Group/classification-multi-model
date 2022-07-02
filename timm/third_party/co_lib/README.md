# Full installation

```bash
git clone -b feature-shell-scripts git@bitbucket.org:cocopie/toolchain.git  # the newest
cd scripts
chmod +x aimet_install_py37.sh
./aimet_install_py37.sh co-lib-py37
```

or

```bash
chmod +x aimet_install_py37.sh  # script in current folder
./aimet_install_py37.sh co-lib-py37
```
# Partial installation (without install AIMET) 
Partial installation will only support pruning function, quantization config will be ignored
```bash
git clone git@bitbucket.org:cocopie/co_lib.git
cd co_lib
python setup.py install
# or
python setup.py develop

```

# Examples:

```bash
import argparse
import torch
import json

# Cocopie Lib 1 add prune_lib
from co_lib import Co_Lib as CL
# Cocopie end


def training_main(args_ai):
    # user original args generator
    user_args = user_args_parser()

    # Cocopie pruning 0:parsing args; ****************************************************************************************************************
    parser = argparse.ArgumentParser(description='Co lib')
    CL.argument_parser(parser)
    args_ai = parser.parse_args()
    # Or you can load args from json *****************************************************************************************************************
    # from third_party.model_train.toolchain.model_train.model_train_tools import *
    json_path = r'simple_task.json'
    args_ai = json.load(open(json_path, 'r'))
    # Cocopie end

    # define dataloader
    trainDataLoader = torch.utils.data.DataLoader(...)
    testDataLoader = torch.utils.data.DataLoader(...)

    # define DNN model
    model = ...
    model = torch.nn.DataParallel(model).cuda()  # if using multiple GPU

    # define loss function
    criterion = ...

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)

    # Cocopie pruning 1:add init function*************************************************************************************************************
    CL.init(args=args_ai, model=model, optimizer=optimizer, data_loader=trainDataLoader)
    # Cocopie end

    for epoch in range(start_epoch, args.epoch):

        # Cocopie pruning 2: add prune_update ********************************************************************************************************
        CL.before_each_train_epoch(epoch=epoch)
        # Cocopie end

        scheduler.step()

        # Cocopie pruning 3: add prune_update_learning_rate ******************************************************************************************
        CL.after_scheduler_step(epoch=epoch)
        # Cocopie end

        for batch_id, data in enumerate(trainDataLoader):
            model.train()

            output = model(input)

            loss = criterion(...)  # regular loss, i.e., cross-entropy, mse, ...

            # Cocopie pruning 4: add prune_update_loss ***********************************************************************************************
            loss = CL.update_loss(loss)
            # Cocopie end

            loss.backward()
            optimizer.step()
            # Cocopie end

        if epoch % user_args.eval_epochs == 0:
            accuracy = eval(model)
            # save the model
            save_path = 'path_name.pth'
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            # if you want to export quantization model ***********************************************************************************************
            # aimet_export_onnx(model,'runs/','co_lib_qat',dummy_input=dummy_input)
            # else you can save your model in your code **********************************************************************************************
            # torch.save(state, save_path)

        # Cocopie end
```

# Minimum necessary code insertion for different functions

```bash
# magnitude pruning/quantization *********************************************************************************************************************
from third_party.co_lib.co_lib import Co_Lib as CL

json_path = r'simple_task.json'
args_ai = json.load(open(json_path, 'r'))
CL.init(args=args_ai, model=model, optimizer=optimizer, data_loader=qat_data)

# admm pruning ***************************************************************************************************************************************
from third_party.co_lib.co_lib import Co_Lib as CL

json_path = r'simple_task.json'
args_ai = json.load(open(json_path, 'r'))
CL.init(args=args_ai, model=model, optimizer=optimizer, data_loader=qat_data)
CL.before_each_train_epoch(epoch=epoch)
CL.after_scheduler_step(epoch=epoch)
loss = CL.update_loss(loss)
```

# Args json define

```bash
Example config could be found at /example/example_config_json/

args_json={
  #prune key args
  "prune" : {
        "sp_admm": false, # whether using admm pruning
        "sp_retrain": false, # whether using magnitude pruning
        "sparsity_type": "block_punched", # sparsity_type block_punched/irregular
        "sp_admm_lr": 0.01, # admm special learning rate
        "sp_admm_block": "(4,4)", # admm block size, a hyperparameter when we choose block-punched
        "sp_admm_update_epoch": 5, #how frequently we update z and u in admm pruning
        "sp_admm_rho": 0.001, # ( a hyperparameter in admm loss function)
        "sp_prune_ratios": 0.5, # the sparsity ratios for uniform pruning
        "sp_config_file": null, # a configure file, the user could define which layer they want to prune(either json str or a file path for a yaml file)
                                # if it is null, will auto generate a basic uniform or global_weight config
        "sp_prune_threshold": -1.0, # user could define the customized threshold for pruning
        "sp_global_weight_sparsity": false, # if true will using global_weight to prune instead of per layer
  },

  #quantization key args
  "quantization": {
        "qt_aimet": false, # whether using aimet quantization, false will disable all args
        "qat": true,  # whether using QAT, it will auto disable amp, and replace pytorch weights norm
        "fold_layers": true, #whether using fold_layers, include conv + batch norm fusion.
        "cross_layer_equalization": true, # whether using cross_layer_equalization, it will force do fold layers
        "bias_correction": true, # whether using bias_correction
        "rounding_mode": "nearest", # rounding_mode nearest/stochastic
        "num_quant_samples": 1000, # samples for quantization from data_loader
        "num_bias_correct_samples": 1000, # samples for bias_correct from data_loader
        "weight_bw": 8, # weights bit width
        "act_bw": 8, # activation bit width
        "quant_scheme": "tf_enhanced", # quantization scheme in Aimet
        "layers_to_ignore": [], # any layer ignore in bias_correction
        "auto_add_bias": true, # auto add bias to each module, will force do it when apply bias_correction
        "perform_only_empirical_bias_corr": true # whether  perform_only_empirical_bias_corr
  }
}
```

# sp_config_file

```bash

# json_string formate example:
prune = {
        ......
        "sp_config_file": {"prune_ratios": {"layer1.0.conv1.weight": 0.5, "layer1.0.conv2.weight": 0.5, "layer1.1.conv1.weight": 0.5, "layer1.1.conv2.weight": 0.5, "layer2.0.conv1.weight": 0.5, "layer2.0.conv2.weight": 0.5, "layer2.0.shortcut.0.weight": 0.5, "layer2.1.conv1.weight": 0.5, "layer2.1.conv2.weight": 0.5, "layer3.0.conv1.weight": 0.5, "layer3.0.conv2.weight": 0.5, "layer3.0.shortcut.0.weight": 0.5, "layer3.1.conv1.weight": 0.5, "layer3.1.conv2.weight": 0.5, "layer4.0.conv1.weight": 0.5, "layer4.0.conv2.weight": 0.5, "layer4.0.shortcut.0.weight": 0.5, "layer4.1.conv1.weight": 0.5, "layer4.1.conv2.weight": 0.5}}
        ......
        }
# yaml file example:
prune = {
        ......
        "sp_config_file": resnet-18.yaml
        ......
        }
resnet-18.yaml:
prune_ratios:
  layer1.0.conv1.weight: 0.5
  layer1.0.conv2.weight: 0.5
  layer1.1.conv1.weight: 0.5
  layer1.1.conv2.weight: 0.5
  layer2.0.conv1.weight: 0.5
  layer2.0.conv2.weight: 0.5
  layer2.0.shortcut.0.weight: 0.5
  layer2.1.conv1.weight: 0.5
  layer2.1.conv2.weight: 0.5
  layer3.0.conv1.weight: 0.5
  layer3.0.conv2.weight: 0.5
  layer3.0.shortcut.0.weight: 0.5
  layer3.1.conv1.weight: 0.5
  layer3.1.conv2.weight: 0.5
  layer4.0.conv1.weight: 0.5
  layer4.0.conv2.weight: 0.5
  layer4.0.shortcut.0.weight: 0.5
  layer4.1.conv1.weight: 0.5
  layer4.1.conv2.weight: 0.5

```
# Use cases
```bash
#Magnitude auto uniform pruning, pruning_ratios = 0.5 ****************************************
args_json={
      "prune": {
      "sp_retrain": true,
      "sp_admm": false,
      "sp_config_file": null,
      "sp_admm_update_epoch": 5,
      "sp_admm_rho": 0.001,
      "sparsity_type": "block_punched",
      "sp_admm_lr": 0.01,
      "sp_global_weight_sparsity": false,
      "sp_admm_block": "(8,4)",
      "sp_prune_ratios": 0.5,
      "sp_prune_threshold": -1.0
    },
    "quantization": {
      "qt_aimet": false,
    }
}

#Magnitude auto global nonuniform pruning, pruning_ratios = 0.5 ****************************************
args_json={
      "prune": {
      "sp_retrain": true,
      "sp_admm": false,
      "sp_config_file": null,
      "sp_admm_update_epoch": 5,
      "sp_admm_rho": 0.001,
      "sparsity_type": "block_punched",
      "sp_admm_lr": 0.01,
      "sp_global_weight_sparsity": true,  # changed
      "sp_admm_block": "(8,4)",
      "sp_prune_ratios": 0.5,
      "sp_prune_threshold": -1.0
    },
    "quantization": {
      "qt_aimet": false,
    }
}
#Magnitude uniform pruning with customized layer, pruning_ratios = 0.5 ****************************************
args_json={
      "prune": {
      "sp_retrain": true,
      "sp_admm": false,
      "sp_config_file": "resnet18.yml",  #changed
      "sp_admm_update_epoch": 5,
      "sp_admm_rho": 0.001,
      "sparsity_type": "block_punched",
      "sp_admm_lr": 0.01,
      "sp_global_weight_sparsity": false ,  
      "sp_admm_block": "(8,4)",
      "sp_prune_ratios": 0.5,
      "sp_prune_threshold": -1.0
    },
    "quantization": {
      "qt_aimet": false,
    }
}
resnet18.yml
prune_ratios:
  layer1.0.conv1.weight: 0.5 #  Any value will be overwritten by sp_prune_ratios 
  layer1.0.conv2.weight: 0.5
  layer1.1.conv1.weight: 0.5
  layer1.1.conv2.weight: 0.5
  layer2.0.conv1.weight: 0.5
  layer2.0.conv2.weight: 0.5
  layer2.0.shortcut.0.weight: 0.5
  layer2.1.conv1.weight: 0.8
  layer2.1.conv2.weight: 0.8
  layer3.0.conv1.weight: 0.8
  layer3.0.conv2.weight: 0.8
  layer3.0.shortcut.0.weight: 0.5
  layer3.1.conv1.weight: 0.9
  layer3.1.conv2.weight: 0.9
  layer4.0.conv1.weight: 0.9
  layer4.0.conv2.weight: 0.9
  layer4.0.shortcut.0.weight: 0.5
  layer4.1.conv1.weight: 0.9
  layer4.1.conv2.weight: 0.9

#Magnitude nonuniform pruning with customized layer and pruning_ratios ****************************************
args_json={
      "prune": {
      "sp_retrain": true,
      "sp_admm": false,
      "sp_config_file": "resnet18.yml",
      "sp_admm_update_epoch": 5,
      "sp_admm_rho": 0.001,
      "sparsity_type": "block_punched",
      "sp_admm_lr": 0.01,
      "sp_global_weight_sparsity": false, 
      "sp_admm_block": "(8,4)",
      "sp_prune_ratios": null,  #changed
      "sp_prune_threshold": -1.0
    },
    "quantization": {
      "qt_aimet": false,
    }
}
resnet18.yml
prune_ratios:
  layer1.0.conv1.weight: 0.5 #  Any value will be overwritten by sp_prune_ratios
  layer1.0.conv2.weight: 0.5
  layer1.1.conv1.weight: 0.5
  layer1.1.conv2.weight: 0.5
  layer2.0.conv1.weight: 0.5
  layer2.0.conv2.weight: 0.5
  layer2.0.shortcut.0.weight: 0.5
  layer2.1.conv1.weight: 0.8
  layer2.1.conv2.weight: 0.8
  layer3.0.conv1.weight: 0.8
  layer3.0.conv2.weight: 0.8
  layer3.0.shortcut.0.weight: 0.5
  layer3.1.conv1.weight: 0.9
  layer3.1.conv2.weight: 0.9
  layer4.0.conv1.weight: 0.9
  layer4.0.conv2.weight: 0.9
  layer4.0.shortcut.0.weight: 0.5
  layer4.1.conv1.weight: 0.9
  layer4.1.conv2.weight: 0.9

#admm pruning auto uniform pruning, pruning_ratios = 0.5 ****************************************
args_json={
      "prune": {
      "sp_retrain": false,
      "sp_admm": true,
      "sp_config_file": null,
      "sp_admm_update_epoch": 5,
      "sp_admm_rho": 0.001,
      "sparsity_type": "block_punched",
      "sp_admm_lr": 0.01,
      "sp_global_weight_sparsity": false,
      "sp_admm_block": "(8,4)",
      "sp_prune_ratios": 0.5,
      "sp_prune_threshold": -1.0
    },
    "quantization": {
      "qt_aimet": false,
    }
}
# after admm pruning, we need do retrain again for a sparse model
args_json={
      "prune": {
      "sp_retrain": true,
      "sp_admm": false ,
      "sp_config_file": null,
      "sp_admm_update_epoch": 5,
      "sp_admm_rho": 0.001,
      "sparsity_type": "block_punched",
      "sp_admm_lr": 0.01,
      "sp_global_weight_sparsity": false,
      "sp_admm_block": "(8,4)",
      "sp_prune_ratios": 0.5,
      "sp_prune_threshold": -1.0
    },
    "quantization": {
      "qt_aimet": false,
    }
}

```

# Known Limitations of Pruning

## Block_punched pruning

- Only support weights:
  - shape = weights.shape
  - shape2d = [shape[0], np.prod(shape[1:])]
  - block_rows, block_cols = eval(args.sp_admm_block) # usually sp_admm_block is (8,4)
  - kernel_s1d = shape[2] \* shape[3]
  - length_x = kernel_s1d \* block_cols
  - Constrain: shape2d[0] % block_rows == 0 and shape2d[1] % length_x == 0

# Known Limitations of Quantization

## Data_loader

- Constrain:
  - we use data loader for sampling data init our quantization module:
  - Please make sure your data_loader work for this function
  ```bash
  for (data_in_one_batch, _) in data_loader_n_samples_quant:
        forward_pass(model, data_in_one_batch)
  ```

# Code structure:

```bash
Running logic:
                                 -> admm
                               /
           -> PruneOptimizer -----> magnitude
         /                     \
CoLib -->                        -> ...
       | \
       |   -> QuantizationOptimizer --> aimet_qt
       |                            \
       |----> ...                    -> ...



class logic:
OptimizerBase: --> CoLib
              \
                -> SpOptimizerBase  --> PruneOptimizer
                                    \
                                     -> QuantizationOptimizer

CompressionBase: --> PruneBase ---> Magnitude
                 |            \
                 |              -> Admm
                 |-> QuantizationBase -> AimetQt



call logic:
Current support call logic:(could add more in the future):
- init(args=None, model=None, optimizer=None, logger=None,data_loader=None)
- before_each_train_epoch(epoch)
- after_scheduler_step(epoch)
- update_loss(loss)
```
