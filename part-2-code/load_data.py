import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        # [cite_start]Initialize the T5 tokenizer [cite: 93]
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        
        # Define the task prefix commonly used for T5 to specify the task
        self.prefix = "translate English to SQL: "
        
        # Load and process the data
        self.inputs, self.targets = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        '''
        Helper function to load the .nl and .sql files.
        '''
        # Load natural language inputs
        nl_path = os.path.join(data_folder, f"{split}.nl")
        with open(nl_path, 'r') as f:
            inputs = [line.strip() for line in f.readlines()]
            
        targets = []
        # Load SQL targets only if we are not in the test split
        if split != "test":
            sql_path = os.path.join(data_folder, f"{split}.sql")
            with open(sql_path, 'r') as f:
                targets = [line.strip() for line in f.readlines()]
        else:
            # For test set, we fill targets with None
            targets = [None] * len(inputs)
            
        return inputs, targets
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        '''
        Tokenize a single example.
        '''
        input_text = self.inputs[idx]
        target_text = self.targets[idx]
        
        # Add the prefix to the input text
        input_text = self.prefix + input_text
        
        # Tokenize the input text (encoder input)
        # We return basic lists here and pad them dynamically in the collate_fn
        input_ids = self.tokenizer(input_text, truncation=True, max_length=512).input_ids
        
        target_ids = []
        if target_text is not None:
            # Tokenize the target SQL (decoder output)
            target_ids = self.tokenizer(target_text, truncation=True, max_length=512).input_ids
            
        return {"input_ids": input_ids, "target_ids": target_ids}

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # Convert list of input/target ids into tensors
    input_ids_list = [torch.tensor(item["input_ids"]) for item in batch]
    target_ids_list = [torch.tensor(item["target_ids"]) for item in batch]
    
    # Pad encoder inputs dynamically to the max length in the batch
    encoder_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=PAD_IDX)
    
    # Create attention mask for encoder (1 for real tokens, 0 for padding)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Pad decoder targets dynamically
    decoder_targets = pad_sequence(target_ids_list, batch_first=True, padding_value=PAD_IDX)
    
    # Create decoder inputs by shifting the targets to the right.
    # T5 uses standard teacher forcing where the input to the decoder at step t is the ground truth token at step t-1.
    # The first token should be the pad token (index 0) for T5.
    batch_size = decoder_targets.size(0)
    start_token = torch.full((batch_size, 1), PAD_IDX, dtype=torch.long)
    
    # Slice off the last token of targets (usually EOS) and prepend the start token
    decoder_inputs = torch.cat([start_token, decoder_targets[:, :-1]], dim=1)
    
    # Create initial decoder inputs (just the start token) for inference generation
    initial_decoder_inputs = start_token.view(-1, 1)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    input_ids_list = [torch.tensor(item["input_ids"]) for item in batch]
    
    # Pad encoder inputs
    encoder_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=PAD_IDX)
    
    # Create attention mask
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # For test set, we just need the initial start token to begin generation
    batch_size = encoder_ids.size(0)
    initial_decoder_inputs = torch.full((batch_size, 1), PAD_IDX, dtype=torch.long)
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # Simple loader for the prompting task scripts
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x