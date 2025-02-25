# Public Repo for "Towards Understanding Link Predictor Generalizability Under Distribution Shifts""

# Results Replication

## Code run within a [Miniconda Virtual Environment](https://docs.anaconda.com/miniconda/install/) built from the environment.yml file
## Additional Dataset Details Included in the [Article Appendix](https://openreview.net/pdf?id=UqV89N2a0p#appendix.Ap)

## Generate dataset splits
`
bash gen_synth.sh
`

## Run GCN baseline
`
bash gcn.sh
`

# Minimal Project Installation
```
# Necessary Packages for Minimal Result Replication
# All packages installed via Conda unless specified
Python 3.9
CUDA 11.6
PyTorch 1.13.1
PyTorch Geometric  2.5.2
Torch Scatter 2.1.0
Torch Sparse 0.6.15
Torch Cluster 1.6.0
OGB 1.3.6 # pip
```


# Dataset Naming Scheme
* After gen_synth.py finishes running: 'forward' splits of the shifted dataset will then be available in the dataset/ folder under the name:
<pre>
{data_name}_{split_type}_0_{valid_rat}_{test_rat}_seed1Dataset
</pre>
* 'backward' splits swap 'test_rat' and 'valid_rat' parameters:
<pre>
{data_name}_{split_type}_{test_rat}_{valid_rat}_0_seed1Dataset
</pre>


# Dataset Loading
* LPShift datasets follow the [OGB](https://ogb.stanford.edu/docs/linkprop/#pytorch-geometric-loader) format for positive samples and [HeaRT](https://github.com/Juanhui28/HeaRT/blob/master/heart_negatives/create_heart_negatives.py) for negative valid and testing samples.
* We advise running different size batches for training, validation, and testing to ensure efficient run time.
```
from synth_dataset import SynthDataset

data = SynthDataset(dataset_name="ogbl-collab_CN_2_1_0_seed1").get()  # PyG graph object for training adjacency matrix         
split_edge = SynthDataset(dataset_name="ogbl-collab_CN_2_1_0_seed1").get_edge_split()

pos_train_edge = split_edge['train']['edge']
pos_valid_edge = split_edge['valid']['edge']
pos_test_edge = split_edge['test']['edge']

with open(f'dataset/{dataset_name}Dataset/heart_valid_samples.npy', "rb") as f:
    neg_valid_edge = np.load(f)
    neg_valid_edge = torch.from_numpy(neg_valid_edge)
with open(f'dataset/{dataset_name}Dataset/heart_test_samples.npy', "rb") as f:
    neg_test_edge = np.load(f)
    neg_test_edge = torch.from_numpy(neg_test_edge)

```

# If you use this code or find the article helpful, please cite:
```
@article{revolinsky2024understanding,
  title={Understanding the Generalizability of Link Predictors Under Distribution Shifts on Graphs},
  author={Revolinsky, Jay and Shomer, Harry and Tang, Jiliang},
  journal={arXiv preprint arXiv:2406.08788},
  year={2024}
}
```
