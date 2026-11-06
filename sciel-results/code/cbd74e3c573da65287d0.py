   import time
   import math
   import random

   MAX_MOD = 128
   DEADLINE = 85.0

   def _disjoint(r1, m1, r2, m2):
       g = math.gcd(m1, m2)
       return (r2 - r1) % g != 0

   def construct():
       start_time = time.time()
       
       # Focus on even moduli
       even_moduli = [m for m in range(2, MAX_MOD + 1) if m % 2 == 0]
       
       # Sort by ascending modulus (harder first? or easier first?)
       # Smaller moduli have fewer residues, so they are more constrained.
       # Placing them first reduces the branching factor early?
       # Or does it restrict the rest too much?
       # Usually MRV (Minimum Remaining Values) suggests placing most constrained first.
       # Here, domain size is m. So smallest m is most constrained.
       moduli_order = sorted(even_moduli)
       
       best_solution = []
       
       # Backtracking
       def backtrack(idx, current_assignment):
           nonlocal best_solution
           if time.time() - start_time > DEADLINE:
               return
           
           if idx == len(moduli_order):
               # Found a full solution
               best_solution = [(r, m) for m, r in current_assignment]
               return
           
           m = moduli_order[idx]
           
           # Try residues
           # Heuristic: try residues that are "central" or random?
           # Let's try 0..m-1
           for r in range(m):
               # Check compatibility
               ok = True
               for i in range(idx):
                   m_prev = moduli_order[i]
                   r_prev = current_assignment[i]
                   if not _disjoint(r, m, r_prev, m_prev):
                       ok = False
                       break
               
               if ok:
                   current_assignment.append((m, r))
                   backtrack(idx + 1, current_assignment)
                   current_assignment.pop()
                   
                   # If we found a full solution, we can stop or continue to find another?
                   # We just want max count. Full solution is max for this set.
                   if best_solution:
                       return
           
           # If we couldn't place this modulus, we might want to record the best partial?
           # But we want to maximize count.
           # If we fail at idx, we have a solution of size idx.
           # We should update best_solution if idx > len(best_solution)
           if len(current_assignment) > len(best_solution):
               best_solution = [(r, m) for m, r in current_assignment]

       backtrack(0, [])
       
       # If we didn't find a full solution, best_solution is a partial one.
       # We can try other orderings or random restarts if time permits.
       
       if time.time() - start_time < DEADLINE - 5:
           # Try random shuffles of the even moduli
           rng = random.Random(42)
           for _ in range(10):
               if time.time() - start_time > DEADLINE - 2:
                   break
               order = even_moduli[:]
               rng.shuffle(order)
               best_sol_temp = []
               
               def backtrack_random(idx, current_assignment):
                   nonlocal best_sol_temp
                   if time.time() - start_time > DEADLINE:
                       return
                   
                   if idx == len(order):
                       best_sol_temp = [(r, m) for m, r in current_assignment]
                       return
                   
                   m = order[idx]
                   # Shuffle residues too?
                   residues = list(range(m))
                   rng.shuffle(residues)
                   for r in residues:
                       ok = True
                       for i in range(idx):
                           m_prev = order[i]
                           r_prev = current_assignment[i]
                           if not _disjoint(r, m, r_prev, m_prev):
                               ok = False
                               break
                       
                       if ok:
                           current_assignment.append((m, r))
                           backtrack_random(idx + 1, current_assignment)
                           current_assignment.pop()
                           if best_sol_temp:
                               return
                   
                   if len(current_assignment) > len(best_sol_temp):
                       best_sol_temp = [(r, m) for m, r in current_assignment]

               backtrack_random(0, [])
               if len(best_sol_temp) > len(best_solution):
                   best_solution = best_sol_temp
       
       return best_solution
   