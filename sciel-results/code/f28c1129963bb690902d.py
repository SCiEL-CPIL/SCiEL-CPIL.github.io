import time
import random

def construct():
       start_time = time.time()
       deadline = 58.0
       
       best_set = []
       
       # First, run the deterministic greedy (1 to 256)
       # This is a good baseline.
       def run_greedy(order):
           S = [set() for _ in range(20)]
           S[0] = {0}
           current_size = 0
           A = []
           
           for x in order:
               if time.time() - start_time > deadline:
                   break
               
               valid = True
               # Check validity
               # For each j (new cardinality index for sums using x),
               # new sums are S[j-1] + x.
               # These must not intersect S[i] for i != j.
               # Note: S[j] (old) is also part of new S[j].
               # We already know old S[j] and old S[i] are disjoint.
               # We just need to check (S[j-1] + x) vs S[i] for i != j.
               
               for j in range(1, current_size + 2):
                   # j is the new cardinality of the subset including x
                   # The sums are from old S[j-1] + x
                   if not S[j-1]:
                       continue
                   # Check against all i != j
                   # i can be 0 to current_size (old max cardinality)
                   # Note: new S[j] includes old S[j].
                   # We don't need to check new S[j] against itself.
                   # We need to check (S[j-1] + x) against:
                   # 1. Old S[i] for i != j
                   # 2. Old S[j] is part of new S[j], so no conflict with (S[j-1]+x) within same cardinality is allowed.
                   # Wait, the condition is: Sums of DIFFERENT cardinalities must differ.
                   # So S_new[k] and S_new[l] must be disjoint for k != l.
                   # S_new[j] = Old S[j] U (Old S[j-1] + x).
                   # S_new[i] = Old S[i] U (Old S[i-1] + x) for i != j?
                   # No, for i != j, S_new[i] is just Old S[i] U (Old S[i-1] + x) ONLY if i was affected?
                   # Actually, adding x creates new sums for ALL cardinalities $j$ where we pick x.
                   # A subset of size $j$ either contains x (sum from $S_{j-1}+x$) or not (sum from $S_j$).
                   # So $S'_{j} = S_j \cup (S_{j-1} + x)$.
                   
                   # We need $S'_j \cap S'_i = \emptyset$ for $i \neq j$.
                   # This is equivalent to checking:
                   # 1. $S_j \cap S_i = \emptyset$ (True)
                   # 2. $(S_{j-1} + x) \cap S_i = \emptyset$ for $i \neq j$
                   # 3. $S_j \cap (S_{i-1} + x) = \emptyset$ for $i \neq j$
                   # 4. $(S_{j-1} + x) \cap (S_{i-1} + x) = \emptyset$ for $i \neq j$
                   
                   # Note that 3 is the same as 2 with indices swapped.
                   # Note that 4: $(S_{j-1} + x) \cap (S_{i-1} + x) = (S_{j-1} \cap S_{i-1}) + x$.
                   # Since $S_{j-1}$ and $S_{i-1}$ are disjoint (if $j-1 \neq i-1$), their intersection is empty.
                   # If $j-1 = i-1$, then $j=i$, which is excluded.
                   # So 4 is always satisfied.
                   
                   # So we only need to check:
                   # $(S_{j-1} + x) \cap S_i = \emptyset$ for all $i \neq j$.
                   
                   for s in S[j-1]:
                       candidate = s + x
                       for i in range(current_size + 1):
                           if i == j:
                               continue
                           if candidate in S[i]:
                               valid = False
                               break
                       if not valid:
                           break
                   if not valid:
                       break
               
               if valid:
                   A.append(x)
                   current_size += 1
                   while len(S) <= current_size:
                       S.append(set())
                   for k in range(current_size, 0, -1):
                       S[k].update(s + x for s in S[k-1])
           
           return A

       # Deterministic run
       best_set = run_greedy(list(range(1, 257)))
       
       # Randomized runs
       while time.time() - start_time < deadline:
           random.seed() # or use time
           order = list(range(1, 257))
           random.shuffle(order)
           candidate = run_greedy(order)
           if len(candidate) > len(best_set):
               best_set = candidate
       
       return [int(x) for x in best_set]
