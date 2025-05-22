# NeuroTrails: Training with Dynamic Sparse Heads as the Key to Effective Ensembling

NeuroTrails is a framework for training sparse ensemble networks within a single neural network architecture. 
This approach offers the benefits of ensemble methods while maintaining computational efficiency.


## Installation

Tested on Linux.
```bash
conda create -n neurotrails_vision python=3.11
conda activate neurotrails_vision
pip install -r requirements.txt
```

## Usage

### To train on CIFAR-100: 

NeuroTrails:
```python
python main_cifar.py --model WideResNet --data cifar100 --blocks_in_head 8 --num_ensemble 3 --density 0.2 --sparse_init Multi_Output
```

Single Dense model:
```python
python main_cifar.py --model WideResNet --data cifar100
```

Full Ensemble:
```python
python main_cifar.py --model WideResNet --data cifar100 --baseline_ensemble 3
```

TreeNet:
```python
python main_cifar.py --model WideResNet --data cifar100 --blocks_in_head 8 --num_ensemble 3 --density 1
```

### Multi-GPU training on ImageNet

```python
CUDA_VISIBLE_DEVICES=1,2 python multiproc.py --nnodes 1 --nproc_per_node 2 main_distributed_imagenet.py --data imagenet --model ResNet50 --gpu 1,2 --distributed True --mst_prt 25000 --batch_size 32  --blocks_in_head 10 --num_ensemble 3 --density 0.2 --sparse_init Multi_Output
```

### Pruning baseline training

Train a single dense model using the `main_cifar.py` file, then continue training from the saved checkpoint using `main_pruning.py` with `args.sparse_init == 'pruning'`. 

### Evaluation

To evaluate trained models:
```python
python evaluate_ece_nll.py --model WideResNet --data cifar100 --num_ensemble 3 --blocks_in_head 8
```
