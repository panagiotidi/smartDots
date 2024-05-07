# Instructions

Modify `config.py` to include the settings you desire. Thern run one of the three following options.

### Preprocess

According to the format of the input data:

`python3 preprocess/preprocess_data_v1.py`

or

`python3 preprocess/preprocess_data_v2.py`

### Main train

#### Training of generic classifier (ViT, Clip, ResNet, Inception3, GoogleNet models)

`python3 main_classifier/trainer.py`

#### Training of classifier with lora adapter (ViT and Clip models only)

`python3 lora_classifier/trainer_lora.py`

#### Run sklearn SVC with embeddings from misc models (ViT, Clip, ResNet, Inception3, GoogleNet)

`python3 svc_classifier/SVC_sklearn.py`
