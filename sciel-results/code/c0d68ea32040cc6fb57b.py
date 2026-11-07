   def construct():
       n = P['n']
       edges = P['edges']
       
       adj = [0] * n
       for u, v in edges:
           adj[u] |= (1 << v)
           adj[v] |= (1 << u)
           
       full_mask = (1 << n) - 1
       
       is_clique = [False] * (1 << n)
       is_ind = [False] * (1 << n)
       is_clique[0] = True
       is_ind[0] = True
       
       for mask in range(1, 1 << n):
           lsb = mask & -mask
           v = lsb.bit_length() - 1
           rest = mask ^ lsb
           
           if is_clique[rest] and (adj[v] & rest) == rest:
               is_clique[mask] = True
               
           if is_ind[rest] and (adj[v] & rest) == 0:
               is_ind[mask] = True
               
       is_valid = [is_clique[m] or is_ind[m] for m in range(1 << n)]
       
       memo = {}
       memo[0] = 0
       
       import time
       start_time = time.time()
       deadline = start_time + 55.0 # 55s limit
       
       def solve(rem):
           if time.time() > deadline:
               return None # Signal timeout
           
           if rem in memo:
               return memo[rem]
           
           if is_valid[rem]:
               memo[rem] = 1
               return 1
           
           # Get lowest set bit
           lsb = rem & -rem
           v = lsb.bit_length() - 1
           
           best = n + 1
           # Iterate submasks of rem that contain v
           sub = rem
           while sub:
               if (sub & lsb) and is_valid[sub]:
                   rest = rem ^ sub
                   res = solve(rest)
                   if res is not None:
                       val = res + 1
                       if val < best:
                           best = val
                           if best == 2: # Can't be better than 2 if rem is not valid?
                               # Actually min is 1. If rem is not valid, min is >= 2.
                               # So 2 is the lower bound for non-valid rem.
                               break
               sub = (sub - 1) & rem
           
           memo[rem] = best
           return best
       
       result_val = solve(full_mask)
       
       if result_val is None or result_val > n:
           # Fallback to greedy
           return greedy_solution() # Need to define this or keep old one
       
       # Reconstruct
       # We need to store the choice made in memo?
       # Or just re-run with the known value?
       # Better to store choice.
       
       # Let's modify solve to store choice
       choice = {}
       
       def solve2(rem):
           if time.time() > deadline:
               return None
           
           if rem in memo:
               return memo[rem]
           
           if is_valid[rem]:
               memo[rem] = 1
               choice[rem] = rem
               return 1
           
           lsb = rem & -rem
           v = lsb.bit_length() - 1
           
           best = n + 1
           best_sub = None
           
           sub = rem
           while sub:
               if (sub & lsb) and is_valid[sub]:
                   rest = rem ^ sub
                   res = solve2(rest)
                   if res is not None:
                       val = res + 1
                       if val < best:
                           best = val
                           best_sub = sub
                           if best == 2:
                               break
               sub = (sub - 1) & rem
           
           memo[rem] = best
           choice[rem] = best_sub
           return best
       
       # Reset memo and choice
       memo = {0: 0}
       choice = {0: 0}
       
       result_val = solve2(full_mask)
       
       if result_val is not None and result_val <= n:
           parts = []
           rem = full_mask
           while rem:
               sub = choice[rem]
               if sub is None: break
               verts = []
               temp = sub
               while temp:
                   b = temp & -temp
                   idx = b.bit_length() - 1
                   verts.append(idx)
                   temp ^= b
               parts.append(verts)
               rem ^= sub
           
           # Verify
           if len(parts) == result_val:
               all_verts = []
               for part in parts:
                   all_verts.extend(part)
               if sorted(all_verts) == list(range(n)):
                   return parts
       
       # Fallback
       def greedy_solution():
           parts = []
           for v in range(n):
               placed = False
               for part in parts:
                   trial = part + [v]
                   is_c = True
                   is_i = True
                   for i in range(len(trial)):
                       for j in range(i+1, len(trial)):
                           if adj[trial[i]] & (1 << trial[j]):
                               is_i = False
                           else:
                               is_c = False
                           if not is_c and not is_i:
                               break
                       if not is_c and not is_i:
                           break
                   if is_c or is_i:
                       part.append(v)
                       placed = True
                       break
               if not placed:
                   parts.append([v])
           return parts
       
       return greedy_solution()
   