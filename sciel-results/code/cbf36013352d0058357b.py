import random
import time
import numpy as np

N = 64

def construct():
       # Use numpy for fast distance calculations if possible, but standard lib is fine for small sets
       
       def dist_sq(p1, p2):
           dx = p1[0] - p2[0]
           dy = p1[1] - p2[1]
           return dx*dx + dy*dy

       def is_valid_addition(p, current_set):
           # Check if adding p to current_set keeps it valid
           # Condition: For all q in current_set, d(p,q) must be distinct from d(p, r) for r!=q
           # AND d(p,q) must be distinct from d(q, r) for r in current_set, r!=q, r!=p
           
           # 1. Distances from p to all q in current_set must be unique
           d_to_s = {}
           for q in current_set:
               d = dist_sq(p, q)
               if d in d_to_s:
                   return False
               d_to_s[d] = q
           
           # 2. For each q in current_set, d(p,q) must not equal d(q, r) for any r in current_set
           # We need to check d(p,q) against all distances from q to other points in current_set
           # This is O(N^2) per addition if we do it naively.
           # Optimization: Precompute for each q in current_set, the set of distances to other points.
           # Let's maintain a list of sets: dist_from[i] = set of distances from point i to others in set.
           # But we don't have indices.
           
           # For simplicity and correctness, we do the O(N^2) check.
           # If current_set size is < 200, this is fast.
           for q in current_set:
               d_pq = dist_sq(p, q)
               for r in current_set:
                   if r == q:
                       continue
                   if dist_sq(q, r) == d_pq:
                       return False
           return True

       # Initial greedy set
       random.seed(42)
       points = [(x, y) for x in range(N) for y in range(N)]
       random.shuffle(points)
       
       current_set = []
       for p in points:
           if len(current_set) >= 150: # Cap initial size to keep speed up
               break
           if is_valid_addition(p, current_set):
               current_set.append(p)
               
       best_set = current_set[:]
       
       # Local Search
       start_time = time.time()
       timeout = 280 # Leave some margin
       
       while time.time() - start_time < timeout:
           # Try to add a random point
           p = random.choice(points)
           if p in current_set:
               continue
               
           if is_valid_addition(p, current_set):
               current_set.append(p)
               if len(current_set) > len(best_set):
                   best_set = current_set[:]
           else:
               # Try swap
               if current_set:
                   q_idx = random.randint(0, len(current_set)-1)
                   q = current_set[q_idx]
                   # Remove q
                   current_set.pop(q_idx)
                   # Try add p
                   if is_valid_addition(p, current_set):
                       current_set.append(p)
                       if len(current_set) > len(best_set):
                           best_set = current_set[:]
                   else:
                       # Revert
                       current_set.append(q)
                       # Try to add a different point? Or just revert.
                       
       return best_set
