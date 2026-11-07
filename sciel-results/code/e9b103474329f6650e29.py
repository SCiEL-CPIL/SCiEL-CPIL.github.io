import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
       n = P['n']
       adj = [0] * n
       for u, v in P['edges']:
           adj[u] |= (1 << v)
           adj[v] |= (1 << u)
       
       full_mask = (1 << n) - 1
       
       is_clique = [False] * (1 << n)
       is_indep = [False] * (1 << n)
       is_clique[0] = True
       is_indep[0] = True
       
       for mask in range(1, 1 << n):
           lsb = mask & (-mask)
           v = lsb.bit_length() - 1
           rest = mask ^ lsb
           
           if is_clique[rest]:
               # Check if v is connected to all in rest
               if (rest & ~adj[v] & full_mask) == 0:
                   is_clique[mask] = True
                   
           if is_indep[rest]:
               # Check if v is NOT connected to any in rest
               if (rest & adj[v]) == 0:
                   is_indep[mask] = True
       
       INF = 10**9
       dp = [INF] * (1 << n)
       dp[0] = 0
       
       # To speed up, we can precompute the list of valid submasks for each mask?
       # Or just iterate.
       
       for mask in range(1, 1 << n):
           best = INF
           sub = mask
           while sub > 0:
               if is_clique[sub] or is_indep[sub]:
                   val = 1 + dp[mask ^ sub]
                   if val < best:
                       best = val
                       if best == 1:
                           break
               sub = (sub - 1) & mask
           dp[mask] = best
           
       # Reconstruct
       result_parts = []
       mask = full_mask
       while mask > 0:
           found = False
           sub = mask
           while sub > 0:
               if (is_clique[sub] or is_indep[sub]) and dp[mask ^ sub] == dp[mask] - 1:
                   part = []
                   s = sub
                   while s > 0:
                       lsb = s & (-s)
                       v = lsb.bit_length() - 1
                       part.append(v)
                       s ^= lsb
                   result_parts.append(part)
                   mask ^= sub
                   found = True
                   break
               sub = (sub - 1) & mask
           if not found:
               break
               
       return [[int(v) for v in part] for part in result_parts]
