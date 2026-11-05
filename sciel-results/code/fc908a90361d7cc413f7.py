import numpy as np
import time
import random

def construct():
       # Primes <= 1000
       def get_primes(n):
           sieve = [True] * (n + 1)
           sieve[0] = sieve[1] = False
           for i in range(2, int(n**0.5) + 1):
               if sieve[i]:
                   sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
           return [i for i, is_prime in enumerate(sieve) if is_prime]
       
       primes = get_primes(1000)
       P = len(primes)
       
       # Max L to check
       L_MAX = 2000 
       
       # State
       C = np.zeros(L_MAX + 1, dtype=np.int8)
       residues = [0] * P
       
       def calc_L():
           # Find first x in 1..L_MAX where C[x] < 2
           # C[0] is unused
           # We want min index i >= 1 such that C[i] < 2
           # If none, return L_MAX
           # np.nonzero is fast
           idx = np.nonzero(C[1:L_MAX+1] < 2)[0]
           if len(idx) == 0:
               return L_MAX
           return idx[0]
           
       # Initial Randomization
       for i, p in enumerate(primes):
           residues[i] = random.randint(0, p-1)
           C[residues[i]::p] += 1
       
       current_L = calc_L()
       best_L = current_L
       best_residues = residues.copy()
       
       start_time = time.time()
       time_limit = 110 # Safety margin for 120s limit
       
       # SA / Hill Climbing Loop
       # Temperature for SA? Or just Hill Climbing with random restarts?
       # Given the structure, Hill Climbing with "Repair" is likely sufficient.
       # Add some random moves to escape local minima.
       
       while time.time() - start_time < time_limit:
           # 50% Repair, 50% Random
           if random.random() < 0.5:
               # Repair Move
               if current_L >= L_MAX:
                   # Try to extend L_MAX? Or just break.
                   # For now, break or do random moves.
                   pass
               else:
                   x_fail = current_L # The first failing index
                   # Pick a random prime
                   i = random.randint(0, P-1)
                   p = primes[i]
                   r_old = residues[i]
                   
                   # If this prime already covers x_fail, moving it is likely bad for x_fail
                   # But maybe it's the only way to fix something else?
                   # For repair, we want to ADD coverage to x_fail.
                   # So we should pick a prime that does NOT cover x_fail.
                   if (x_fail - r_old) % p == 0:
                       # This prime covers x_fail. Skip for repair?
                       # Or try a random move instead.
                       continue
                   
                   r_new = x_fail % p
                   # Apply change
                   C[r_old::p] -= 1
                   C[r_new::p] += 1
                   residues[i] = r_new
                   
                   new_L = calc_L()
                   
                   if new_L > current_L:
                       current_L = new_L
                       if current_L > best_L:
                           best_L = current_L
                           best_residues = residues.copy()
                   else:
                       # Revert
                       C[r_new::p] -= 1
                       C[r_old::p] += 1
                       residues[i] = r_old
           else:
               # Random Move
               i = random.randint(0, P-1)
               p = primes[i]
               r_old = residues[i]
               r_new = random.randint(0, p-1)
               
               if r_new == r_old:
                   continue
                   
               C[r_old::p] -= 1
               C[r_new::p] += 1
               residues[i] = r_new
               
               new_L = calc_L()
               
               if new_L > current_L:
                   current_L = new_L
                   if current_L > best_L:
                       best_L = current_L
                       best_residues = residues.copy()
               else:
                   # Simple Hill Climbing: Revert
                   # Or SA: Accept with prob
                   # Let's stick to Hill Climbing for stability, 
                   # but maybe allow some degradation to escape?
                   # If we just revert, we might get stuck.
                   # Let's try: Accept if new_L >= current_L - 10 (small penalty)
                   # Or just revert. With 168 vars, random moves might be enough.
                   C[r_new::p] -= 1
                   C[r_old::p] += 1
                   residues[i] = r_old
                   
       return best_residues
