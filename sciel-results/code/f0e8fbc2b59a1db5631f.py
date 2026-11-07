import time
from math import gcd
from functools import reduce

def construct():
       divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
       n = len(divisors)
       
       def lcm(a, b):
           return a * b // gcd(a, b)
       LCM = reduce(lcm, divisors)
       
       start_time = time.time()
       deadline = start_time + 85.0
       
       best_span = float('inf')
       best_reps = None
       
       # Sort indices by divisor size descending for MRV heuristic in DFS
       # Actually, for MRV, we sort by number of candidates, which changes with W.
       # So we can't pre-sort.
       
       for L in range(1, LCM + 1):
           if time.time() > deadline:
               break
           
           # We want to find min W for this L
           # Start W=0
           # We can break if current W >= best_span
           
           # Precompute first multiple >= L for each divisor
           # first_m[i]
           first_m = [0] * n
           for i in range(n):
               d = divisors[i]
               first_m[i] = ((L + d - 1) // d) * d
           
           # We will iterate W from 0 upwards
           # We maintain a list of candidates for each divisor
           # To avoid rebuilding, we can just check multiples in range
           # But checking multiples in range is O(W/d).
           
           # Let's try a simple loop for W
           # Max W to check: best_span - 1
           # If best_span is inf, check up to, say, 200
           
           max_w_to_check = min(best_span - 1, 200)
           
           for W in range(0, max_w_to_check + 1):
               if time.time() > deadline:
                   break
               
               # Build candidates
               candidates = [[] for _ in range(n)]
               feasible = True
               for i in range(n):
                   d = divisors[i]
                   # First multiple >= L
                   m = first_m[i]
                   while m <= L + W:
                       candidates[i].append(m)
                       m += d
                   if not candidates[i]:
                       feasible = False
                       break
               
               if not feasible:
                   continue
               
               # Check if distinct assignment exists
               # Use DFS with MRV
               # Sort indices by len(candidates[i])
               order = sorted(range(n), key=lambda i: len(candidates[i]))
               
               used = set()
               assignment = [None] * n
               
               def dfs(idx):
                   if idx == n:
                       return True
                   i = order[idx]
                   for m in candidates[i]:
                       if m not in used:
                           used.add(m)
                           assignment[i] = m
                           if dfs(idx + 1):
                               return True
                           used.remove(m)
                           assignment[i] = None
                   return False
               
               if dfs(0):
                   # Found solution
                   span = W
                   if span < best_span:
                       best_span = span
                       best_reps = [assignment[i] for i in range(n)]
                   break # Found min W for this L, move to next L
           
           # Optional: if best_span is very small (e.g., 10), we might stop early?
           # No, we need to check all L to find global min.
           # But if best_span is 1, we can stop.
           if best_span == 1:
               break
       
       if best_reps is None:
           # Fallback
           best_reps = [d for d in divisors]
       
       return [int(x) for x in best_reps]
