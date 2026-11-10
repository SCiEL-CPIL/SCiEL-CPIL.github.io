import time
import random

def construct():
       start_time = time.time()
       deadline = 80.0 # 80s limit, 10s margin
       
       # 1. Initialize with large numbers
       S = set(range(91, 181))
       
       # Helper to check if x is safe to add to S
       def is_safe(x, S):
           # Check if there exist b, c in S, b != c, such that bc = x(b+c)
           # (b-x)(c-x) = x^2
           # d1*d2 = x^2
           # b = x + d1, c = x + d2
           # 91 <= x+d1 <= 180 => 91-x <= d1 <= 180-x
           # Same for d2
           # d1 != d2 (for b != c)
           
           x2 = x * x
           # Find factors of x^2
           # Iterate d1 up to sqrt(x^2)
           limit = int(x**0.5)
           for d1 in range(1, limit + 1):
               if x2 % d1 == 0:
                   d2 = x2 // d1
                   if d1 == d2:
                       continue # b=c, not a forbidden triple
                   
                   b1 = x + d1
                   c1 = x + d2
                   b2 = x + d2
                   c2 = x + d1
                   
                   # Check if b1, c1 are in S and in range [1, 180]
                   # Actually S is subset of [1, 180], so range check is implicit if we check membership
                   # But we must ensure b1, c1 are within [1, 180] to be valid candidates
                   # However, S only contains [91, 180] initially, and we add smalls.
                   # So b, c must be in S.
                   
                   if b1 in S and c1 in S:
                       return False
                   if b2 in S and c2 in S:
                       return False
                       
           return True

       # 2. Add safe small numbers
       for x in range(1, 91):
           if time.time() > deadline:
               break
           if is_safe(x, S):
               S.add(x)
       
       # 3. Try to improve by swapping
       # Iterate through numbers not in S (unsafe smalls)
       # For each, see if we can add it by removing one conflicting element
       # This is O(N^2) roughly, but N is small.
       
       # List of numbers not in S
       missing = [x for x in range(1, 181) if x not in S]
       
       # Shuffle to avoid bias
       random.shuffle(missing)
       
       for x in missing:
           if time.time() > deadline:
               break
           
           # Find all conflicting pairs (b, c) in S
           # If we remove b, does it resolve the conflict?
           # We need to check if removing b allows x to be added.
           # Actually, we just need to check if x is safe in S \ {b}.
           
           # Get all b in S that form a conflict with x
           # Conflict: b, c in S, b!=c, bc=x(b+c)
           # We can find all such b.
           
           conflicting_bs = []
           x2 = x * x
           limit = int(x**0.5)
           for d1 in range(1, limit + 1):
               if x2 % d1 == 0:
                   d2 = x2 // d1
                   if d1 == d2: continue
                   
                   b1 = x + d1
                   c1 = x + d2
                   b2 = x + d2
                   c2 = x + d1
                   
                   if b1 in S and c1 in S:
                       conflicting_bs.append(b1)
                       conflicting_bs.append(c1)
                   if b2 in S and c2 in S:
                       conflicting_bs.append(b2)
                       conflicting_bs.append(c2)
           
           # If no conflicts, it would have been added in step 2.
           # So conflicting_bs is not empty.
           
           # Try removing each unique conflicting b
           # If we remove b, check if x becomes safe.
           # Note: Removing b might create new conflicts for other elements?
           # No, removing elements only reduces conflicts.
           
           for b in set(conflicting_bs):
               # Check if x is safe in S - {b}
               S_test = S.copy()
               S_test.remove(b)
               if is_safe(x, S_test):
                   # We can add x and remove b
                   # Is it worth it?
                   # Size stays same.
                   # But maybe it allows other additions?
                   # Let's do it to potentially unlock other numbers.
                   S = S_test
                   S.add(x)
                   break # Only one removal per x
       
       return list(S)
