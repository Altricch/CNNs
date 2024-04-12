import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from dataset import BilingualDataset, causal_mask
from model import build_transformer

# Tensorboard
from torch.utils.tensorboard import SummaryWriter

# Huggingface tokenizers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# Library to create absolute paths given relative paths
from pathlib import Path

from config import get_config, get_weights_file_path

from tqdm import tqdm

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    # config['tokenizer_file] == "../tokenizers/tokenizer_{0}.json" -> depending on our defined language
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency= 2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# Get dataset
def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split = "train")
    
    # Build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])
    
    # Keep 80% for training and 20% for validation
    train_ds_size = int(0.8 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    # breakpoint()
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_src.encode(item["translation"][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
    print(f"Max length of src {max_len_src}, Max length of tgt {max_len_tgt}")
    
    train_data_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_data_loader = DataLoader(val_ds, batch_size=1, shuffle=True)
    
    return train_data_loader, val_data_loader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], config["d_model"])
    return model


def train_model(config):
    # define the device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is", device)
    
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    
    model = get_model(config,tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    model.to(device)
    
    # TensorBoard
    writer = SummaryWriter(config["experiment_name"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print("Preloading model", model_filename)
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
        
    # We ignore Pad for loss (shouldnt contribute to loss)
    # Label_smoothing: distribute certain percentage of max softmax to other labels
    # Thus, model becomes "less sure", avoids overfitting and better learning
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)
    
    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epcoch {epoch:02d}")
        
        for batch in batch_iterator:
            
            encoder_input = batch["encoder_input"].to(device) #B, seq_len
            decoder_input = batch["decoder_input"].to(device) #B, seq_len
            
            encoder_mask = batch["encoder_mask"].to(device) #B,1,1 seq_len
            decoder_mask = batch["decoder_mask"].to(device) #B,1, seq_len, seq_len
            
            # Run tensors through transformer:
            encoder_output = model.encode(encoder_input,encoder_mask) #(B,seq_en, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) #(B,seq_en, d_model)
            proj_output = model.project(decoder_output)
            
            label = batch["label"].to(device) #B,seq_len
            
            # (B, seq_len, tgt_vocab_size) --> (B * SeqLen, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
            
            # log loss to tensorboard
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            global_step +=1
            
        # Save model per epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        
if __name__ == '__main__':
    config = get_config()
    train_model(config)  
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    