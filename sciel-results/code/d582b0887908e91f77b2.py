def construct():
         import time
         start_time = time.time()
         deadline = start_time + 80
         
         LIMIT = 1009
         MAX_VAL = 1007
         MIN_VAL = 2
         
         inv = [0] * (LIMIT + 1)
         for i in range(1, LIMIT):
             inv[i] = pow(i, -1, LIMIT)
             
         best_len = 0
         best_start = MIN_VAL
         
         for L in range(MIN_VAL, MAX_VAL + 1):
             if time.time() > deadline:
                 break
             R = L - 1
             # Extend R as far as possible
             while R < MAX_VAL:
                 next_val = R + 1
                 inv_next = inv[next_val]
                 # Check if inv_next is in [L, R]
                 if L <= inv_next <= R:
                     break
                 R = next_val
             
             length = R - L + 1
             if length > best_len:
                 best_len = length
                 best_start = L
                 
         if best_len == 0:
             best_start = MIN_VAL
             best_len = 1
             
         return [best_start, best_len]
