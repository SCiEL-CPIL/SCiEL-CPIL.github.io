   import time
   import random
   import math

   LCM = 232792560
   P = 1000000007 # Prime > sum(LCM/x)
   
   def _is_dissociated_mod(elements, recips_mod):
       sums = {0}
       for e in elements:
           r = recips_mod[e]
           new_sums = set()
           for s in sums:
               ns = (s + r) % P
               if ns in sums:
                   return False
               new_sums.add(ns)
           sums.update(new_sums)
       return True

   def construct():
       start_time = time.time()
       deadline = start_time + 85
       
       recips_mod = {x: (LCM // x) % P for x in range(1, 23)}
       
       best = []
       
       # Greedy with random shuffles
       # We can run this in a loop until deadline
       # Use a fixed seed for reproducibility? Or random?
       # Random is better for exploration.
       
       # To avoid re-checking the same set, we can keep a set of tried tuples?
       # No, just run many trials.
       
       trial = 0
       while time.time() < deadline - 2:
           order = list(range(1, 23))
           random.shuffle(order)
           current = []
           for x in order:
               if time.time() > deadline - 2:
                   break
               candidate = current + [x]
               if _is_dissociated_mod(candidate, recips_mod):
                   current = candidate
           if len(current) > len(best):
               best = current
               # If we found 22, we can break? Unlikely.
               if len(best) == 22:
                   break
           trial += 1
           
       # Local search on best
       if time.time() < deadline - 5:
           current = list(best)
           improved = True
           while improved and time.time() < deadline - 5:
               improved = False
               for x in range(1, 23):
                   if x not in current:
                       candidate = current + [x]
                       if _is_dissociated_mod(candidate, recips_mod):
                           current = candidate
                           improved = True
                           break
               if len(current) > len(best):
                   best = current
                   
       return sorted(best)
   