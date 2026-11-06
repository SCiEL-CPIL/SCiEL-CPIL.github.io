import time

def construct():
       start_time = time.time()
       limit = 50.0 # Safety margin
       
       def check_valid(S, x, subset_info):
           # Check 1: x is not mean of subset of S
           for s, k, mask in subset_info:
               if k >= 2 and s == k * x:
                   return False
           
           # Check 2: no y in S is mean of subset of S U {x} \ {y}
           # Subsets of S U {x} \ {y} that contain x are {x} U T' where T' subset of S \ {y}
           for i, y in enumerate(S):
               bit_y = 1 << i
               for s, k, mask in subset_info:
                   if mask & bit_y:
                       continue
                   # T' is represented by (s, k, mask)
                   # Check if s == (1 + k) * y - x
                   if s == (1 + k) * y - x:
                       return False
           return True
       
       S = []
       subset_info = [(0, 0, 0)]
       
       for x in range(1, 513):
           if time.time() - start_time > limit:
               break
           
           if check_valid(S, x, subset_info):
               S.append(x)
               # Update subset_info
               new_info = []
               idx = len(S) - 1
               bit_x = 1 << idx
               for s, k, mask in subset_info:
                   new_info.append((s, k, mask))
                   new_info.append((s + x, k + 1, mask | bit_x))
               subset_info = new_info
       
       return S
