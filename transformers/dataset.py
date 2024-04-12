import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        
        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype = torch.int64)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_tgt_pair = self.ds[index]
        src_text = src_tgt_pair["translation"][self.src_lang]
        tgt_text = src_tgt_pair["translation"][self.tgt_lang]
        
        enc_input_token = self.tokenizer_src.encode(src_text).ids
        dec_input_token = self.tokenizer_src.encode(tgt_text).ids
        
        # -2 because of SOS and EOS
        encode_num_pad = self.seq_len - len(enc_input_token) - 2
        # Decoder only SOS
        decode_num_pad = self.seq_len - len(dec_input_token) - 1
        
        if encode_num_pad < 0 or decode_num_pad < 0:
            raise ValueError("Sentence is too long")
        
        
        # Add SOS and EOS to source text
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_token, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * encode_num_pad, dtype= torch.int64)
        ])
        
        # Only add SOS 
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_token, dtype=torch.int64),
            torch.tensor([self.pad_token] * decode_num_pad, dtype= torch.int64)
        ]
        )
        
        # Only add EOS to the label (what we expect from the decoder)
        label = torch.cat([
            torch.tensor(dec_input_token, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * decode_num_pad, dtype= torch.int64)
        ])
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return {
            "encoder_input": encoder_input, #seq_len
            "decoder_input": decoder_input, #seq_len
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            # Causal mask builds matrix (seq_len x seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label, #seq_len,
            "src_text": src_text,
            "tgt_text": tgt_text
        }
        
# To only look at words that came before
def causal_mask(size):
    mask = torch.triu(torch.ones(1,size,size), diagonal=1).type(torch.int)
    return mask==0
        