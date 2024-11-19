from bebert_arc.bebert import BeBertEncoder
from train_utils.bebert_trainer import BERTTrainer
from train_utils.mlm_model import MLMModel
from train_utils.data_utils import MLMDataset, LengthSortedBatchSampler
from train_pipelines.load_data_func import load_data

from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from torch.utils.data import DataLoader
import yaml



if __name__ == "__main__":
    config_file_path = "train_pipelines/train_config.yaml"
    with open(config_file_path, "r") as yamlfile:
        train_config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    # load all 
    tokenizer = AutoTokenizer.from_pretrained(train_config['data_params']['tokenizer_hf_path'])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=train_config['data_params']['mlm_probability'])

    # load corpus
    data_ru, data_en = load_data(train_config['data_params']['data_path_en'], train_config['data_params']['data_path_ru'])
    data = data_ru + data_en
    
    # make datasets
    train_data = MLMDataset(data[:int(train_config['data_params']['train_size']*len(data))], tokenizer=tokenizer,
                            max_seq_len=train_config['data_params']['max_seq_len'])
    test_data = MLMDataset(data[int(train_config['data_params']['train_size']*len(data)):], tokenizer=tokenizer,
                           max_seq_len=train_config['data_params']['max_seq_len'])

    # make samplers
    train_sampler = LengthSortedBatchSampler(train_data, batch_size=train_config['data_params']['train_bs'])
    test_sampler = LengthSortedBatchSampler(test_data, batch_size=train_config['data_params']['test_bs'])

    # make loaders
    train_loader = DataLoader(train_data, batch_sampler=train_sampler, collate_fn=data_collator)
    test_loader = DataLoader(test_data, batch_sampler=test_sampler, collate_fn=data_collator)

    # create bebert and mlm models
    encoder_model = BeBertEncoder(tokenizer.vocab_size, model_dim=train_config['bebert_params']['model_dim'], 
                                  n_blocks=train_config['bebert_params']['n_blocks'], n_heads=train_config['bebert_params']['n_heads'])    
    
    mlm_train_model = MLMModel(encoder_model, train_config['bebert_params']['model_dim'], tokenizer.vocab_size)

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