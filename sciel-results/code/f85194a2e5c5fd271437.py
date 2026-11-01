   def greedy_mis_bitset(adj_bitsets, n, rng):
       avail = (1 << n) - 1
       selected = 0
       count = 0
       while avail:
           best_u = -1
           min_deg = n
           candidates = []
           a = avail
           while a:
               l = a & -a
               u = l.bit_length() - 1
               a ^= l
               deg = (adj_bitsets[u] & avail).bit_count()
               if deg < min_deg:
                   min_deg = deg
                   candidates = [u]
                   if min_deg == 0:
                       break
               elif deg == min_deg:
                   candidates.append(u)
           
           if candidates:
               best_u = candidates[rng.integers(len(candidates))]
           else:
               break # Should not happen if avail is not empty
           
           selected |= (1 << best_u)
           count += 1
           avail &= ~(adj_bitsets[best_u] | (1 << best_u))
       
       return count, selected
   