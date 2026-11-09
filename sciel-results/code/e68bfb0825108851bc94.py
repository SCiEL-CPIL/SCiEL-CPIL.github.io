   import random
   
   def construct():
       n = 128
       S = set()
       
       def is_valid_add(x):
           if x in S:
               return True
           # Check if adding x creates a violation
           # Violation: distinct a,b,c in S U {x} such that a+b, a+c, b+c in S U {x}
           # Since x is new, any violation must involve x.
           # Let the triple be (x, a, b) with a,b in S.
           # Conditions:
           # 1. x+a in S U {x} => x+a in S (since x+a != x for a>0)
           # 2. x+b in S U {x} => x+b in S
           # 3. a+b in S U {x}
           
           S_list = list(S)
           for i in range(len(S_list)):
               a = S_list[i]
               if a >= x: continue
               for j in range(i+1, len(S_list)):
                   b = S_list[j]
                   if b >= x: continue
                   
                   # Check a+b
                   sum_ab = a + b
                   if sum_ab == x:
                       # sum_ab is in S U {x}
                       # Check x+a and x+b
                       if (x + a) in S and (x + b) in S:
                           return False
                   else:
                       # sum_ab must be in S
                       if sum_ab in S:
                           if (x + a) in S and (x + b) in S:
                               return False
           return True

       # Initial greedy: Add large numbers first
       for x in range(n, 0, -1):
           if is_valid_add(x):
               S.add(x)
       
       # Local Search
       # Try to add missing numbers
       missing = [x for x in range(1, n+1) if x not in S]
       random.shuffle(missing)
       
       for _ in range(5000):
           added = False
           for x in missing:
               if is_valid_add(x):
                   S.add(x)
                   missing.remove(x)
                   added = True
                   break
           if not added:
               # Try to remove a number to allow more additions?
               # Simple strategy: Try swapping
               # Pick a small number y in S, remove it, try to add a missing x
               # This is complex. Let's just try adding again with different order?
               # Or break if no progress.
               pass
               
       return sorted(S)
   