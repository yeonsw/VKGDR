from transformers import BertTokenizerFast

def get_tokenizer(init_checkpoint):
    tokenizer = \
        BertTokenizerFast \
            .from_pretrained(
                init_checkpoint, \
                do_lower_case=True \
            )
    special_tokens_dict = { \
        'additional_special_tokens': [
            '[unused1]', '[unused2]', '[unused3]' \
        ] \
    }
    tokenizer.add_special_tokens( \
        special_tokens_dict)
    
    return tokenizer
