# Using this code
The data intended for use is from the 2021 NFL Big Data Bowl, [found on Kaggle](https://www.kaggle.com/c/nfl-big-data-bowl-2021/data). The files `preprocessing.py`, `timesteps.py`, and `event_frames.py` can be used to convert the data into a form usable by the network.

The files `visualize.py`, `visualize_predicted.py`, `visualize_events.py` and `visualize_tsne.py` are used to create visualizations at various stages of processing.

The `NETS` directory contains the code for the netowrk. `NFL_pretraining.py` is used to train the trajectory predictor, `predict.py` is used to generate trajectory predictions, `finetuning.py` is used to train the classifier, and `tsne_code_snippets.py` is used to produce a visualization of the clusters. There are several files within this directory that aren't used by this project. The original NETS project is hosted [here](https://github.com/S-Hauri/NETS)