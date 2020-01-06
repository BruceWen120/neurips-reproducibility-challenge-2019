# Ablation study for Neurips 2019 reproducibility challenge

Paper we studied: `Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation`[(Arxiv:1905.12926)](https://arxiv.org/abs/1905.12926). Our report can be found [here](https://openreview.net/forum?id=XibbOqrT4q).

## About the paper
In this work, we present a controllable unsupervised text attribute transfer framework, which can edit the entangled latent representation instead of modeling attribute and content separately. Specifically, we first propose a Transformer-based autoencoder to learn an entangled latent representation for a discrete text, then we transform the attribute transfer task to an optimization problem and propose the Fast-Gradient-Iterative-Modification algorithm to edit the latent representation until conforming to the target attribute. To the best of our knowledge, this is the first one that can not only control the degree of transfer freely but also perform sentiment transfer over multiple aspects at the same time. 

![Model architecture](/file/model.png)

## Documents

### Dependencies
	Python 3.7
	PyTorch 1.2.0

### Directory description

<pre><code>Root
├─data/*        Store datasets
├─method/*      Store the source code and saved models
├─fasttext/*	Fasttext classifier for evaluation
├─results/*	    Generated sentences and evaluation results
├─srilm/*	    Language model for evaluation
└─outputs/*	    Store evaluation scripts
</code></pre>

###  Data Preprocessing
In the data directory, run:

	python preprocessed_data.py 


### Run main model

To train the model, run in the method directory:

	python main.py 

To use trained models, you can specify the check point directory in `main.py`,

	args.if_load_from_checkpoint = True
	args.checkpoint_name = "xxx"

and then run `python main.py` .

To use modified version of the original model, specify corresponding parameters after `python main.py`.

For example, to use pretrained model with 512-dimensional latent space, use `python main.py --latent_size 512 --transformer_model_size 512`.

### Pretrained models indexes

Yelp dataset:

- original parameters: 1576344354
- 1 Transformer block: 1576119110
- 512-dimensional latent space: 1576109322
- 128-dimensional latent space: 1576097025

Caption dataset:

- original parameters: 1576006739
- 1 Transformer block: 1576083260
- 512-dimensional latent space: 1576011793
- 128-dimensional latent space: 1576010830

## LICENSE

[MIT](./LICENSE)
