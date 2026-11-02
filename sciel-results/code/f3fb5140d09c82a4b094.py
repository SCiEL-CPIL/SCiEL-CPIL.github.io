import random
import time
import itertools

def construct(q, n, radius):
       start_time = time.time()
       deadline = start_time + 110
       
       # Precompute index mapping
       all_words = list(itertools.product(range(q), repeat=n))
       word_to_idx = {w: i for i, w in enumerate(all_words)}
       idx_to_word = [None] * (q**n)
       for w, i in word_to_idx.items():
           idx_to_word[i] = w
           
       # Precompute neighbors indices
       # neighbors_idx[i] is list of indices of words at dist <= radius
       neighbors_idx = [[] for _ in range(q**n)]
       for i, w in enumerate(all_words):
           w_list = list(w)
           nbs = [i]
           for k in range(n):
               orig = w_list[k]
               for v in range(q):
                   if v != orig:
                       w_list[k] = v
                       nbs.append(word_to_idx[tuple(w_list)])
               w_list[k] = orig
           neighbors_idx[i] = nbs
           
       tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
       initial_code_idx = [word_to_idx[tuple(h + list(tail))] for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
       
       best_code_idx = initial_code_idx.copy()
       best_len = len(initial_code_idx)
       
       while time.time() < deadline:
           code_idx = initial_code_idx.copy()
           counts = [0] * (q**n)
           for c in code_idx:
               for nb in neighbors_idx[c]:
                   counts[nb] += 1
           
           improved = True
           while improved:
               improved = False
               random.shuffle(code_idx)
               for i in range(len(code_idx)):
                   c = code_idx[i]
                   removable = True
                   for nb in neighbors_idx[c]:
                       if counts[nb] == 1:
                           removable = False
                           break
                   if removable:
                       for nb in neighbors_idx[c]:
                           counts[nb] -= 1
                       code_idx.pop(i)
                       improved = True
                       
           if len(code_idx) < best_len:
               best_len = len(code_idx)
               best_code_idx = code_idx.copy()
               
       return [list(idx_to_word[i]) for i in best_code_idx]
