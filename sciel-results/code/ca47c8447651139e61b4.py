   def construct():
       from math import gcd
       
       N = 128
       nums = list(range(1, N + 1))
       
       # Precompute GCD matrix
       gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
       for i in range(1, N + 1):
           for j in range(i, N + 1):
               g = gcd(i, j)
               gcd_mat[i][j] = g
               gcd_mat[j][i] = g
       
       def is_valid(selected):
           sel = list(selected)
           n = len(sel)
           for i in range(n):
               a = sel[i]
               for j in range(i + 1, n):
                   b = sel[j]
                   gab = gcd_mat[a][b]
                   for k in range(j + 1, n):
                       c = sel[k]
                       if gcd_mat[a][c] == gab and gcd_mat[b][c] == gab:
                           return False
           return True
       
       def can_add(selected, x):
           sel = list(selected)
           n = len(sel)
           for i in range(n):
               a = sel[i]
               gax = gcd_mat[a][x]
               for j in range(i + 1, n):
                   b = sel[j]
                   if gcd_mat[a][b] == gax and gcd_mat[b][x] == gax:
                       return False
           return True
           
       # Heuristic: Start with all prime powers
       def is_prime_power(n):
           if n == 1: return True
           p = 2
           while p * p <= n:
               if n % p == 0:
                   m = n
                   while m % p == 0:
                       m //= p
                   return m == 1
               p += 1
           return True # n is prime
           
       initial = [n for n in nums if is_prime_power(n)]
       # Add 1? No, 1 conflicts with 2 and 3.
       
       # Try to add composites
       best_set = set(initial)
       
       # Local search
       improved = True
       while improved:
           improved = False
           for x in nums:
               if x in best_set:
                   continue
               if can_add(best_set, x):
                   best_set.add(x)
                   improved = True
           
           # Try swaps: remove y, add x
           # If we remove y, we might be able to add multiple x's?
           # Simple swap:
           for y in list(best_set):
               temp_set = set(best_set)
               temp_set.remove(y)
               added = 0
               for x in nums:
                   if x in temp_set:
                       continue
                   if can_add(temp_set, x):
                       temp_set.add(x)
                       added += 1
               if len(temp_set) > len(best_set):
                   best_set = temp_set
                   improved = True
                   
       return list(best_set)
   