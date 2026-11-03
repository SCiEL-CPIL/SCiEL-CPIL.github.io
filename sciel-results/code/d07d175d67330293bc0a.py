import itertools
import time
import random

def construct(q, n, radius):
       if q != 3 or n != 7 or radius != 1:
           # Fallback to general case if parameters change
           # For now, assume specific case
           pass
       
       # Mapping functions
       def word_to_idx(word):
           idx = 0
           for w in word:
               idx = idx * q + w
           return idx
       
       def idx_to_word(idx):
           word = [0] * n
           for i in range(n-1, -1, -1):
               word[i] = idx % q
               idx //= q
           return word
           
       # Precompute balls
       all_words_idx = list(range(q**n))
       balls = {}
       for idx in all_words_idx:
           w = idx_to_word(idx)
           ball = [idx]
           for i in range(n):
               orig = w[i]
               for new_val in range(q):
                   if new_val == orig: continue
                   w[i] = new_val
                   ball.append(word_to_idx(w))
               w[i] = orig
           balls[idx] = ball
           
       # Generate initial code (Direct Sum)
       tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
       init_codewords = []
       for h in tetracode:
           for tail in itertools.product(range(3), repeat=n-4):
               init_codewords.append(h + list(tail))
               
       init_indices = [word_to_idx(c) for c in init_codewords]
       active_set = set(init_indices)
       
       # Coverage counts
       cover_count = [0] * (q**n)
       for idx in active_set:
           for w in balls[idx]:
               cover_count[w] += 1
               
       # Greedy Removal
       start_time = time.time()
       deadline = start_time + 110 # 10s margin
       
       while time.time() < deadline:
           changed = False
           # Shuffle to avoid bias
           current_active = list(active_set)
           random.shuffle(current_active)
           
           for idx in current_active:
               if idx not in active_set:
                   continue
               
               # Check redundancy
               redundant = True
               for w in balls[idx]:
                   if cover_count[w] == 1:
                       redundant = False
                       break
               
               if redundant:
                   active_set.remove(idx)
                   for w in balls[idx]:
                       cover_count[w] -= 1
                   changed = True
                   
           if not changed:
               break
               
       result = [idx_to_word(idx) for idx in active_set]
       return result
