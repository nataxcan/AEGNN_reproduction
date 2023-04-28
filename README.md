Download dataset 101ncaltech

New variant reproduction
1)	create environment from \AEGNN_reproduction\new_algorithm_variant\dl_proj2_environment.yml

Hyper-param-reproduction
1)	Create environment from \AEGNN_reproduction\hyper-parameter-check\environment.yml
2)	Change sample_sizes [] to an list of desired sample sizes one would like to test
For changing the model network change network_variations to an array with values in range 0-2. Linked to the models defined in models.py
3)	run \AEGNN_reproduction\hyper-pararm-check\execution.ipynb

New data reproduction
1)	create env from \AEGNN_reproductionenvironment.yml
2)	run \AEGNN_reproduction\ReproductionEvaluation.ipynb
