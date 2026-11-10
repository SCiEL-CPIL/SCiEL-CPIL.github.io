import time
import random
from math import gcd

def construct():
       start_time = time.time()
       deadline = start_time + 145.0
       
       best_classes = []
       best_score = 0.0
       
       moduli = list(range(20, 201))
       
       # Helper to check compatibility
       def is_compatible(m, r, selected):
           for m2, r2 in selected:
               g = gcd(m, m2)
               if r % g == r2 % g:
                   return False
           return True
       
       # Helper to find valid residues
       def get_valid_residues(m, selected):
           return [r for r in range(m) if is_compatible(m, r, selected)]

       # We try to find a complete solution.
       # Use a stack for backtracking.
       
       for _ in range(1000): # Max restarts if needed, but time bounded
           if time.time() > deadline:
               break
           
           # State: list of [m, r]
           current = []
           # We process moduli in increasing order
           # But we can backtrack.
           
           # To make backtracking efficient, we store the list of choices made.
           # Simple recursive backtracker with depth limit?
           # Or iterative with stack.
           
           # Let's use a simple recursive function with a depth limit or just let it run
           # but with randomization to avoid getting stuck in bad local minima.
           
           def backtrack(index):
               if time.time() > deadline:
                   return False
               
               if index == len(moduli):
                   return True
               
               m = moduli[index]
               valids = get_valid_residues(m, current)
               
               if not valids:
                   return False
               
               # Shuffle valids to randomize search
               random.shuffle(valids)
               
               for r in valids:
                   current.append([m, r])
                   if backtrack(index + 1):
                       return True
                   current.pop()
               
               return False
           
           # The above recursive backtracker might be too slow if it explores too much.
           # However, the constraints are loose.
           # To prevent exponential blowup, we can add a heuristic:
           # If we get stuck, restart.
           
           # Let's try a simpler approach first: Pure greedy with random restarts.
           # If that fails to find a full solution, we can try backtracking.
           
           # Actually, let's just use the greedy with random restarts first.
           # It's O(N^2 * M) per run, which is very fast.
           
           success = False
           current = []
           for m in moduli:
               valids = get_valid_residues(m, current)
               if not valids:
                   break
               r = random.choice(valids)
               current.append([m, r])
           
           if len(current) == len(moduli):
               score = sum(1.0 / m for m, r in current)
               if score > best_score:
                   best_score = score
                   best_classes = current
               break # Found optimal, stop
           
           # If we didn't find a full solution, keep the best partial one?
           # No, we want to maximize score. A full solution is best.
           # If no full solution found yet, keep the best partial.
           score = sum(1.0 / m for m, r in current)
           if score > best_score:
               best_score = score
               best_classes = current

       return best_classes
