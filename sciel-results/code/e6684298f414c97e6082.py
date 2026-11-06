import time

def construct():
       MOD = 1009
       # Precompute inverses
       inv = [0] * (MOD + 1)
       for x in range(1, MOD):
           inv[x] = pow(x, MOD - 2, MOD)
       
       best_len = 0
       best_start = 2 # default valid start
       
       for s in range(1, MOD):
           # Check if s itself is invalid (self-inverse)
           if s == 1 or s == 1008:
               continue
           
           current_end = s
           next_val = s + 1
           
           while next_val <= 1008:
               # Check if next_val is self-inverse
               if next_val == 1 or next_val == 1008:
                   break
               # Check if inverse of next_val is in [s, current_end]
               if s <= inv[next_val] <= current_end:
                   break
               
               current_end = next_val
               next_val += 1
           
           length = current_end - s + 1
           if length > best_len:
               best_len = length
               best_start = s
               
       if best_len == 0:
           return [2, 1]
       
       return [best_start, best_len]
