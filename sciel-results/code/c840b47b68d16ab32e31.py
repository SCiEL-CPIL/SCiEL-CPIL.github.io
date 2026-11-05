import time

def construct():
      MOD = 1009
      N = 1008
      start_time = time.time()
      deadline = 80
      
      inv = [0] * MOD
      for x in range(1, MOD):
          inv[x] = pow(x, MOD - 2, MOD)
          
      best_start = 2
      best_len = 0
      
      R = 1
      for L in range(2, MOD):
          if time.time() - start_time > deadline:
              break
          # Extend R as far as possible
          # Condition for [L, R+1] to be valid:
          # inv[R+1] must not be in [L, R+1]
          # i.e., inv[R+1] < L or inv[R+1] > R+1
          while R < MOD - 1:
              next_R = R + 1
              ix = inv[next_R]
              if ix < L or ix > next_R:
                  R = next_R
              else:
                  break
          
          if R >= L:
              length = R - L + 1
              if length > best_len:
                  best_len = length
                  best_start = L
          else:
              # If R < L, we need to move R to at least L
              # But R only increases. If R < L, it means we couldn't extend R
              # from a previous state where R was < L?
              # Actually, R starts at 1. L starts at 2.
              # If R < L, the window is empty.
              # We should set R = L - 1? No, R is global.
              # If R < L, it implies that for the current L, we couldn't even fit L.
              # But L itself might be valid?
              # Check if [L, L] is valid: inv[L] < L or inv[L] > L.
              # If inv[L] == L, invalid.
              # If R < L, it means the while loop didn't run or failed immediately.
              # If R < L, we cannot have a valid interval starting at L with end >= L.
              # So length 0.
              pass
              
      return [best_start, best_len]
