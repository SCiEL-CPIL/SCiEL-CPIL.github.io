import numpy as np
import time
import random

def construct():
       start_time = time.time()
       deadline = start_time + 110  # Leave 10s margin

       def get_color_matrix(colors, n):
           A = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
           idx = 0
           for i in range(n):
               for j in range(i + 1, n):
                   c = colors[idx]
                   A[c][i, j] = 1
                   A[c][j, i] = 1
                   idx += 1
           return A

       def find_violations(A):
           violations = []
           for c in range(5):
               # A[c] @ A[c]
               counts = A[c] @ A[c]
               np.fill_diagonal(counts, 0)
               # Find indices where counts > 1
               # To avoid creating a huge list if many, we can sample or just take all
               # For n=27, max pairs is 351.
               idxs = np.argwhere(counts > 1)
               for i, j in idxs:
                   if i < j:
                       violations.append((i, j, c))
           return violations

       def search(n, time_limit):
           # Initialize random
           num_edges = n * (n - 1) // 2
           colors = [random.randint(0, 4) for _ in range(num_edges)]
           
           # Optional: Try a few random restarts
           best_score = float('inf')
           best_colors = None
           
           restarts = 0
           while time.time() < start_time + time_limit and restarts < 20:
               restarts += 1
               # If not first restart, randomize colors
               if restarts > 1:
                   colors = [random.randint(0, 4) for _ in range(num_edges)]
                   
               A = get_color_matrix(colors, n)
               violations = find_violations(A)
               
               steps = 0
               while violations and time.time() < start_time + time_limit:
                   steps += 1
                   # Pick random violation
                   u, v, c = random.choice(violations)
                   
                   # Find common neighbors in color c
                   # Neighbors of u in c: A[c][u] == 1
                   # Neighbors of v in c: A[c][v] == 1
                   # Intersection
                   neigh_u = np.where(A[c][u] == 1)[0]
                   neigh_v = np.where(A[c][v] == 1)[0]
                   common = np.intersect1d(neigh_u, neigh_v)
                   
                   if len(common) < 2:
                       # Should not happen if violation is valid, but safety
                       break
                   
                   # Pick two common neighbors
                   w1, w2 = random.sample(list(common), 2)
                   
                   # Edges to flip candidates: (u, w1), (u, w2), (v, w1), (v, w2)
                   # All are currently color c
                   candidates = [
                       (min(u, w1), max(u, w1)),
                       (min(u, w2), max(u, w2)),
                       (min(v, w1), max(v, w1)),
                       (min(v, w2), max(v, w2))
                   ]
                   
                   # Pick one candidate
                   i, j = random.choice(candidates)
                   idx = i * n - (i * (i + 1) // 2) + (j - i - 1) 
                   # Wait, index calculation:
                   # Index of (i, j) in lexicographic list of pairs (x, y) with x < y
                   # Total pairs before i: i * n - i * (i + 1) / 2
                   # Pairs (i, k) for k in i+1..j-1: j - i - 1
                   # So idx = i * n - i * (i + 1) // 2 + j - i - 1
                   
                   # Verify index formula:
                   # n=5, pairs:
                   # (0,1)->0, (0,2)->1, (0,3)->2, (0,4)->3
                   # (1,2)->4, (1,3)->5, (1,4)->6
                   # (2,3)->7, (2,4)->8
                   # (3,4)->9
                   # Formula for (i,j):
                   # i=0, j=1: 0*5 - 0 + 0 = 0. Correct.
                   # i=0, j=4: 0*5 - 0 + 3 = 3. Correct.
                   # i=1, j=2: 1*5 - 1*2//2 + 0 = 5 - 1 = 4. Correct.
                   # i=1, j=4: 1*5 - 1 + 2 = 6. Correct.
                   # i=3, j=4: 3*5 - 3*4//2 + 0 = 15 - 6 = 9. Correct.
                   
                   new_c = random.choice([x for x in range(5) if x != c])
                   colors[idx] = new_c
                   
                   # Update A matrices for efficiency?
                   # Recalculating A is O(n^2). Flipping is O(1).
                   # But find_violations is O(n^3).
                   # For n=27, O(n^3) is fast. Recalculating A is fine.
                   A = get_color_matrix(colors, n)
                   violations = find_violations(A)
                   
                   if not violations:
                       return colors
                       
               # If loop finished without success, restart
           
           return None

       # Try n=27
       result = search(27, 30)
       if result:
           return {'n': 27, 'colors': result}
       
       # Try n=26
       result = search(26, 30)
       if result:
           return {'n': 26, 'colors': result}
       
       # Fallback to n=11
       n = 11
       return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
