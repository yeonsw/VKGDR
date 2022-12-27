import re
import string

def normalize_string(s):

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def normalize_techqa_string(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

class TechQATokenizer:
    def __init__(self, e2id):
        self.e2id = e2id
        self.longest_e_len = min( \
            max([len(e.split()) for e in self.e2id]), 5 \
        )

    def techqa_tokenize(self, doc_tokens, mentions):
        tokens = []
        entities = []
        if len(mentions) == 0:
            return {
                "tokens": doc_tokens,
                "entities": entities,
            }
        
        mentions_sorted = sorted(mentions, key=lambda x: x[0][0], reverse=True)
        for i in range(len(mentions_sorted) - 1):
            assert mentions_sorted[i][0][0]  \
                        >= mentions_sorted[i+1][0][1]

        tokens = doc_tokens[:]
        for m in mentions_sorted:
            s, e = m[0]
            text = m[1]
            mention_text = " ".join(tokens[s:e])
            tokens = tokens[:s] + [(mention_text, text)] + tokens[e:]
        
        for i, tok in enumerate(tokens):
            if isinstance(tok, tuple):
                entities.append((i, self.e2id[tok[1]]))
        tokens = [tok[0] if isinstance(tok, tuple) else tok for tok in tokens]
        return {
            "tokens": tokens,
            "entities": entities
        }
    
    def techqa_match_entities(self, doc_tokens, s, e, l):
        if l == 0:
            return []
        if s == e:
            return []
        
        ngrams = []
        for n in range(l, 0, -1):
            for i in range(s, e - n + 1):
                word = " ".join(doc_tokens[i:i+n])
                word = word[:-len("'s")] if word.endswith("'s") else word
                ngram_word = normalize_techqa_string(word)
                if ngram_word in self.e2id:
                    left = self.techqa_match_entities( \
                        doc_tokens, s, i, n - 1 \
                    )
                    right = self.techqa_match_entities( \
                        doc_tokens, i+n, e, n \
                    )
                    return left + [((i, i+n), ngram_word)] + right
        return []

    def techqa_entity_matching_and_tokenizing(self, doc):
        doc_tokens = doc.split()
        
        matchings = self.techqa_match_entities( \
            doc_tokens, \
            0, len(doc_tokens), \
            min(self.longest_e_len, len(doc_tokens))
        )
        result = self.techqa_tokenize(doc_tokens, matchings)
        return result
