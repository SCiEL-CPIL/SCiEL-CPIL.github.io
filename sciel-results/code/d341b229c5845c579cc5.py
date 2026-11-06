import random
import time
import math

def construct():
       n = 20
       max_val = 10000
       
       def score(A):
           sums = set()
           prods = set()
           for a in A:
               for b in A:
                   sums.add(a + b)
                   prods.add(a * b)
           return max(len(sums), len(prods))
       
       # Generate valid seeds
       seeds = []
       
       # Seed 1: 1..20
       seeds.append(list(range(1, n + 1)))
       
       # Seed 2: Powers of 2 where possible, fill rest with small primes
       p2 = [2**i for i in range(14)] # up to 8192
       # Need 6 more
       extras = [3, 5, 7, 11, 13, 17]
       # Ensure distinct
       s2 = p2 + extras
       # Check distinctness and max
       if len(set(s2)) == n and max(s2) <= max_val:
           seeds.append(s2)
           
       # Seed 3: Random
       s3 = random.sample(range(1, max_val + 1), n)
       seeds.append(s3)
       
       # Seed 4: Arithmetic Progression
       s4 = [1 + i * 500 for i in range(n)]
       if max(s4) <= max_val:
           seeds.append(s4)
           
       best_A = None
       best_score = float('inf')
       
       for s in seeds:
           sc = score(s)
           if sc < best_score:
               best_score = sc
               best_A = s[:]
               
       start_time = time.time()
       deadline = start_time + 55 # 55s limit for 60s kill, safe margin
       
       current_A = best_A[:]
       current_score = best_score
       
       T = 5.0
       cooling = 0.9995
       
       # To speed up, we can inline score or use local vars
       # But function call overhead is negligible compared to set ops
       
       while time.time() < deadline:
           idx = random.randint(0, n - 1)
           old_val = current_A[idx]
           
           # Perturbation
           if random.random() < 0.05:
               new_val = random.randint(1, max_val)
           else:
               # Local move
               delta = random.randint(-20, 20)
               new_val = old_val + delta
               if new_val < 1: new_val = 1
               if new_val > max_val: new_val = max_val
               
           if new_val in current_A:
               continue
               
           current_A[idx] = new_val
           new_score = score(current_A)
           
           delta_s = new_score - current_score
           
           if delta_s < 0 or (T > 0.1 and random.random() < math.exp(-delta_s / T)):
               current_score = new_score
               if new_score < best_score:
                   best_score = new_score
                   best_A = current_A[:]
           else:
               current_A[idx] = old_val
               
           T *= cooling
           if T < 0.1:
               T = 0.1
               
       return best_A
