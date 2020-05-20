## Applied Deep Learning - Student Project 2019/2020
### University of Bristol, Dept. of Computer Science

This is a reimplementation of the paper: "Yu Su, Ke Zhang, Jingyu Wang, Kurosh Madani, **Environment Sound Classification Using a Two-Stream CNN Based on Decision-Level Fusion**, Sensors 2019" [link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6479959/)

For two stream fusion we simply average the predictions of LMC and MC networks.

The hyperparameters that we used for training are:

* learning rate: 0.001
* batch  size: 32
* dropout: 0.5
* weight decay: 0.0005
* optimiser: Adam

### Training

To train the LMC model, run:

```
python train.py LMC --train_pickle path/to/UrbanSound8K_train.pkl --test_pickle path/to/UrbanSound8K_test.pkl 
```
To train MC annd MLMC models, simply replace the first argument in the line above with *MC* and *MLMC* respectively.

### Testing

To test the LMC model, run:

```
python test.py LMC --weights_dir models/mode\=LMC/date/ --scores_output scores/LMC.pkl --test_pickle path/to/UrbanSound8K_test.pkl --mapping class_mapping.pkl --average
```

To test MC annd MLMC models, simply replace every instance of *LMC* in the line above with *MC* and *MLMC* respectively.

To test the 2 stream fusion model (assuming that you have already trained and tested LMC and MC), run:

```
python test.py LMC+MC --scores_input scores/LMC.pkl scores/MC.pkl --scores_output scores/LMC_MC_fusion.pkl --mapping class_mapping.pkl --average
```
