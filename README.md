
# AEGNN_reproduction
Here follow some brief instructions to be able to arive at the same evaluations as in the blog post.
To obtain the data for N-Caltech101 one should run
```
wget "https://filesender.surf.nl/download.php?token=d1cc4a6d-9a9d-45bb-84eb-2791e53a2a33&files_ids=13325018"
```

## Reproduction
1. Download the N-Calltech101 dataset
2. Create environment from \AEGNN_reproduction\new_data\environment.yml
3. The code that was used to run the evaluation of the reproduction can be found in the new_data folder under the name "ReproductionEvaluation.ipynb". Be sure to include the N-Calltech101 data in a folder called data in that directory.
## New Data
1. Download the N-Cars dataset from [https://drive.google.com/file/d/1vlByGVjqmyYvbzLSIzZzNLfcjfTJijyz/view?usp=sharing]
2. Create environment from \AEGNN_reproduction\new_data\environment.yml
3. The code that was used to run the evaluation of the reproduction can be found in the new_data folder under the name "EvaluateNCARS.ipynb". Be sure to include the N-Cars data in a folder called data/storage/N-Cars_parsed in the new_data directory.
## Hyper Parameters Check
1.	Create environment from \AEGNN_reproduction\hyper-parameter-check\environment.yml
2.	Change sample_sizes [] to an list of desired sample sizes one would like to test
For changing the model network change network_variations to an array with values in range 0-2. Linked to the models defined in models.py
3.	run \AEGNN_reproduction\hyper-pararm-check\execution.ipynb
## New algorithm variant
1. create environment from `\AEGNN_reproduction\new_algorithm_variant\dl_proj2_environment.yml`
2. on linux or MacOS, run the following command to download the dataset: `wget "https://filesender.surf.nl/download.php?token=d1cc4a6d-9a9d-45bb-84eb-2791e53a2a33&files_ids=13325018"` or on windows simply visit the address: `https://filesender.surf.nl/download.php?token=d1cc4a6d-9a9d-45bb-84eb-2791e53a2a33&files_ids=13325018`
3. unzip the data.zip file in the `new_algorithm_variant` folder
4. for the simple graph convolution experiment, run the `execution_usingsimple.ipynb` notebook
5. for the network with all transformer convolutions, to run the hyperparameter search, run `execution_transformer_param_search.ipynb`
6. for the network with the last two layers being transformer convolutions, run `execution_transformer.ipynb`