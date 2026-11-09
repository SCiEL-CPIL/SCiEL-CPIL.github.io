   import random
   import time

   def construct():
       n = 128
       best_size = 0
       best_set = set()
       
       # Try a few random orders and some heuristic orders
       orders = []
       # Descending
       orders.append(list(range(n, 0, -1)))
       # Ascending
       orders.append(list(range(1, n + 1)))
       # Random
       for _ in range(50):
           perm = list(range(1, n + 1))
           random.shuffle(perm)
           orders.append(perm)
           
       for order in orders:
           S = set()
           # To speed up checks, we can maintain a list of S
           S_list = []
           
           for x in order:
               # Check if x can be added
               valid = True
               
               # Find a in S such that x+a in S
               # We can iterate through S
               # A_x = {a in S | x+a in S}
               A_x = []
               for a in S_list:
                   if (x + a) in S:
                       A_x.append(a)
               
               if len(A_x) < 2:
                   S.add(x)
                   S_list.append(x)
                   continue
               
               # Check pairs in A_x
               # a, b in A_x, a != b
               # Condition: a+b in S OR a+b == x
               for i in range(len(A_x)):
                   for j in range(i + 1, len(A_x)):
                       a = A_x[i]
                       b = A_x[j]
                       s_ab = a + b
                       if s_ab in S or s_ab == x:
                           valid = False
                           break
                   if not valid:
                       break
               
               if valid:
                   S.add(x)
                   S_list.append(x)
                   
           if len(S) > best_size:
               best_size = len(S)
               best_set = S.copy()
               
       return sorted(best_set)
   