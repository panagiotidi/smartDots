from utils import inverse_weights

# ---------------------------------------------------------------------------------------------------------------
# Settings for downloading data
# metadata_csv = '/Users/sofia/PycharmProjects/smartDots/data/SmartDotsSummaryExtraction3.csv'
original_path = '/Users/sofia/PycharmProjects/smartDots/data/original_06_05_2024'
# original_path = '/Users/sofia/PycharmProjects/smartDots/subset'
# ---------------------------------------------------------------------------------------------------------------
# Settings for preprocessing data
data_csv = '/Users/sofia/PycharmProjects/smartDots/data/SmartDotsSummaryExtraction3.csv'
clean_data_path = '/Users/sofia/PycharmProjects/smartDots/data/original_06_05_2024_processed'
# clean_data_path = '/Users/sofia/PycharmProjects/smartDots/data/original_02_04_2024_processed_simple'
# clean_data_path = '/Users/sofia/PycharmProjects/smartDots/data/subset/out_sofia'
# clean_data_path = '/Users/sofia/PycharmProjects/smartDots/data/all_accordance_processed256x256'
MinMaxAgeDif = 3
MaxModalAge = 20
# ---------------------------------------------------------------------------------------------------------------

# set the input height and width
INPUT_HEIGHT = 512  # (416, 416)
INPUT_WIDTH = 512

# Options: 'ResNet', 'GoogLeNet', 'ResNetUNet', 'Inception', 'Clip', 'ViT'
model_name = 'Clip'

# Options:
#  'categorical_abs': classification, logistic regression, one absolute age
#  'categorical_prob': classification, logistic regression, age probability range
#  'continuous' : continuous, non-linear
#  'ordinal' : Ordinal regression is half-way between classification and real-valued regression. When you perform multiclass classification of your ordinal data, you are assigning the same penalty whenever your classifier predicts a wrong class, no matter which one.
regression = 'categorical_prob'

total_classes = 10

# Possible values: None, 'inverse_quantities'
weights = 'inverse_quantities'

# Possible values: None, or List including 'Pleuronectes platessa', 'Ammodytes', 'Solea solea', 'Pollachius pollachius', 'Micromesistius poutassou', 'Pollachius virens', 'Gadus morhua', etc.
# filter_species = None
filter_species = ['Gadus morhua', 'Pleuronectes platessa', 'Micromesistius poutassou',
                  'Lepidorhombus whiffiagonis', 'Clupea harengus', 'Solea solea']

device = 'mps'
subsample_fraction = 1.0
BATCH_SIZE = 12
VAL_SPLIT = 0.2

# ----------- For trainer -------------#
epochs = 3
learning_rate = 1e-03
weight_decay = 0.01
# ----------- For SVC_regression -------------#
C = 0.6
# ----------- Metric settings -------------#
metric_max_diff = 1

# ----------- Misc Models ----------------#

GoogLeNet_model = 'googlenet'

ViT_model_proc = "google/vit-base-patch16-224-in21k"
# ViT_model = "google/vit-base-patch16-224"
ViT_model = ViT_model_proc
ViT_clip_model_preprocess_version = 'ViT-L/14@336px'

Clip_model = 'openai/clip-vit-base-patch32'
# Clip_model = "openai/clip-vit-large-patch14"
