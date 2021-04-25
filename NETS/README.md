# NETS
Neural Embeddings in Team Sports and Application to Basketball Play Analysis

This is the code base for the paper "Neural Embeddings in Team Sports and Application to Basketball Play Analysis", currently under review for KDD 2021. It is based on pytorch with a single GPU, programmed and tested on Windows (check requirements.txt for further dependencies). To modify the hyper-parameters, modify the yaml files.

We provide the data for only one game, but the entire dataset can be found at https://github.com/sealneaward/nba-movement-data.

The entire pipeline can be executed by running main.py, which pretraines a model based on trajectory prediction (self-supervised), extracts weak labels for pick-and-roll and handoffs, and then finetunes the model based on the 3-way classification task pick-and-roll vs. handoff vs. other.

The model is a transformer with LSTM embedding based on the following architecture:
![image](https://user-images.githubusercontent.com/51958221/107989165-a2fe7380-6f9f-11eb-92d4-cef303e5b98a.png)

To create NETS embeddings that are invariant to permutations within a team, we pool the player embeddings per team by summing them together like this:
![image](https://user-images.githubusercontent.com/51958221/107989191-bc9fbb00-6f9f-11eb-9078-7dde6dfc5b12.png)

visual_analytics.py contains data with embeddings generated for one game and provides code to automatically detect regions of interest, where the teams can be linearly separated (with at least 80% accuacy) by which team is on offense.
![image](https://user-images.githubusercontent.com/51958221/107990527-a2b3a780-6fa2-11eb-93d2-03a5a9e89728.png)
