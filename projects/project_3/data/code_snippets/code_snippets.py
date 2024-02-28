import re

def from_file(file_name):
    try:
        with open (f'projects/project_3/data/code_snippets/{file_name}.txt') as f:
            s = f.readlines()
    except:
        with open (f'projects/projects/project_3/data/code_snippets/{file_name}.txt') as f:
            s = f.readlines()


    return ''.join(s)


IMPORTS = from_file('imports')

VISUALIZATION = from_file('visualization')

STATISTICAL = from_file('statistical')

AMAZON_S3_temp = from_file('amazons3_funcs')
mask = " '/Users/konstantinsokolovskiy/Desktop/My_Big_Project/final/projects/projects/project_3/"
AMAZON_S3 = re.sub(mask, " '", AMAZON_S3_temp)[:-1]

RAW_DATA = from_file('raw_data')

WITHOUT_NULL = from_file('without_null')

CORRELATION_MATRIX = from_file('correlation_matrix')

MULTICOLLINEARITY = from_file('multicollinearity')

CHI2_TEST = from_file('chi_2_test')

BASE_TRAIN_TEST_DATA_PREPARATION = from_file('base_train_test_data_preparation')

BASE_MODELS = from_file('base_models')

C_V_OBJECT = from_file('c_v_object')

BASE_TOP_FEATURE_IMPORTANCE = from_file('base_top_feature_importance')

BASE_GRID_SEARCH = from_file('base_grid_search')

BASE_BEST_MODELS_TRAIN = from_file('base_best_models_train')

BALANCE_ENGINEERED = from_file('balance_engineered')

CAMPAIGN_ENGINEERED = from_file('campaign_engineered')

PDAYS_ENGINEERED = from_file('pdays_engineered')

PREVIOUS_ENGINEERED = from_file('previous_engineered')
