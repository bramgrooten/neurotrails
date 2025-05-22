# NeuroTrails: Training with Dynamic Sparse Heads as the Key to Effective Ensembling


## Install

Tested on Linux. 

```bash
conda create -n neurotrails_llm python=3.9
conda activate neurotrails_llm
pip install -r requirements.txt
```

## Downloading the C4 Dataset

When running locally on your laptop, this is not necessary. 
Just make sure that `args.data_dir = None`. (i.e. don't set `--data_dir` in the command line, it is `None` by default).
This will use 'online streaming mode'.

If you are running on a server, we recommend to download the C4 dataset. This can be done with the script:
```bash
scripts/download_c4.sh
```


## Run

To train **NeuroTrails**, run the following script:
```bash
bash scripts/train_llm.sh
```

which uses these default parameters for the 130M model:
```bash
--num_ensemble 3 \
--blocks_in_head 8 \
--density 0.9 \
```

If you run into OOM (out-of-memory) errors, lower the batch size or try a smaller model size. 
Note that we use gradient accumulation, so the total batch size stays the same (512 by default). Only the speed will be affected.

To train a **single dense model**, use:
```bash
--num_ensemble 1 \
--density 1 \
```

To train a **full ensemble**, use:
```bash
--num_ensemble 3 \
--full_ensemble \
```

To train **TreeNet**, use:
```bash
--num_ensemble 3 \
--blocks_in_head 8 \
--density 1 \
```


## Acknowledgement
This repository is built upon the [MixLN](https://github.com/pixeli99/MixLN/tree/main) repo, 
which is based on [GaLore](https://github.com/jiaweizzhao/GaLore). Thanks for their great work!
