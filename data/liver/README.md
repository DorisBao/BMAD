# Liver Tumor Segmentation preprocessing

## Data Download
### Altas
Visit https://www.synapse.org/#!Synapse:syn3193805/wiki/217789, register for the challenge, go to (https://www.synapse.org/#!Synapse:syn3379050) and download the 'RawData.zip'. Unzip the file and put it under `./data/Altas/`.
To preprocess the dataset, run the following command
```
python data_preprocessing.py --phase train
```

### LiTs
Visit LiTs Challenge (https://www.google.com/search?client=safari&rls=en&q=lits+challenge&ie=UTF-8&oe=UTF-8), register for the event, download the dataset and put the unzipped folder under `./data/LiTs`.
To preprocess the dataset, run the following command
```
python data_preprocessing.py --phase tests
```

The preprocessed images would be saved at `./data/liver_dataset`.

