
``` python
# Input: English doc (M lines), Vietnamese doc (N lines)

# First translate the two docs to the other language.
en2vi_doc = vietai_dab_tiny_model_translate_envi(english_doc)
vi2en_doc = vietai_dab_tiny_model_translate_vien(vietnamese_doc)

# Next compute M x N BLEU scores:
M = len(en2vi_doc)
N = len(vi2en_doc)
bleu_score = np.zero((M, N))

for en_line, en2vi_line in zip(english_doc, en2vi_doc):
 for vi_line, vi2en_line in zip(vietnamese_doc, vi2en_doc):

   bleu_score[m, n] = bleu(en_line, vi2en_line) + 
                      bleu(vi_line, en2vi_line)

# Dynamic Programming for pair matching:
# F[m, n] is sum of bleu scores of all pairs in the best matching 
# between en_doc[:m] and vi_doc[:n], then:
F[m, n] = max(F[m-1, n-1] + bleu_score[m, n],
              F[m-1, n],
              F[m, n-1])
```

### 
