from bebert_arc.bebert import BeBertEncoder
from train_utils.bebert_trainer import BERTTrainer
from train_utils.nsp_model import NSPModel
from train_utils.data_utils import NSPDataset, collate_fn_nsp
from train_pipelines.load_data_func import load_data

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import yaml



if __name__ == "__main__":
    config_file_path = "train_pipelines/train_config.yaml"
    with open(config_file_path, "r") as yamlfile:
        train_config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    

    # load all 
    tokenizer = AutoTokenizer.from_pretrained(train_config['data_params']['tokenizer_hf_path'])

    # load corpus
    data_ru, data_en = load_data(train_config['data_params']['data_path_en'], train_config['data_params']['data_path_ru'])
    pairs = []
    for ind in range(len(data_ru)):
        pairs.append([data_ru[ind], data_en[ind]])
    
    # make datasets
    train_data = NSPDataset(pairs[:int(train_config['data_params']['train_size']*len(pairs))], tokenizer=tokenizer,
                            max_seq_len=train_config['data_params']['max_seq_len'])
    test_data = NSPDataset(pairs[int(train_config['data_params']['train_size']*len(pairs)):], tokenizer=tokenizer,
                           max_seq_len=train_config['data_params']['max_seq_len'])

    # make loaders
    train_loader = DataLoader(train_data, batch_size=train_config['data_params']['train_bs'], collate_fn=collate_fn_nsp)
    test_loader = DataLoader(test_data, batch_size=train_config['data_params']['test_bs'], collate_fn=collate_fn_nsp)

    # create bebert and mlm models
    encoder_model = BeBertEncoder(tokenizer.vocab_size, model_dim=train_config['bebert_params']['model_dim'], 
                                  n_blocks=train_config['bebert_params']['n_blocks'], n_heads=train_config['bebert_params']['n_heads'])
    
    mlm_train_model = NSPModel(encoder_model, train_config['bebert_params']['model_dim'])

    # create trainer
    bert_trainer = BERTTrainer(mlm_train_model, train_loader, test_loader, train_config['trainer_params']['lr'],
                               train_config['trainer_params']['weight_decay'], 
                               (train_config['trainer_params']['beta1'], train_config['trainer_params']['beta2']), 
                               train_config['trainer_params']['warmup_steps'], train_config['trainer_params']['log_freq'], 
                               train_config['trainer_params']['device'], 
                               train_config['trainer_params']['checkpoint_dir'],
                               train_config['trainer_params']['train_task'])

    # train
    for epoch in range(train_config['trainer_params']['num_epochs']):
        bert_trainer.train(epoch)