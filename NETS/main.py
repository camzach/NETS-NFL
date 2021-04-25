from pretraining import pretrain
from finetuning import finetune
from extract_tactics import generate_data

trajectory_yaml = 'trajectory_settings'
transformer_yaml = 'transformer_settings'
save_folder = 'saved_models'

#pretrain
print('pretraining')
date_str = pretrain(trajectory_yaml, transformer_yaml)

# weak labeling
print('weak labeling data')
labeled_data = 'saved_data/tactics_weaklabeled.pkl'
generate_data(file_path = 'possession_data', save_name = labeled_data)

#finetune
print('finetuning')
model_name = 'classifier'
finetune(transformer_yaml, labeled_data, save_folder, new_name=model_name,
         load_date=date_str, yaml_pretraining=trajectory_yaml)