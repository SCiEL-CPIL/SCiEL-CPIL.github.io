   def max_clique(adj, n):
       # adj: list of bitmasks
       # returns list of vertices in max clique
       best = []
       
       def color_sort(P):
           # Greedy coloring of P
           colors = {}
           uncolored = list(P)
           c = 0
           while uncolored:
               c += 1
               to_color = []
               still_uncolored = []
               for v in uncolored:
                   if all(not (adj[v] & (1 << u)) for u in to_color):
                       colors[v] = c
                       to_color.append(v)
                   else:
                       still_uncolored.append(v)
               uncolored = still_uncolored
           # Return sorted by color
           return sorted(P, key=lambda x: colors[x])
       
       def expand(R, P):
           nonlocal best
           if not P:
               if len(R) > len(best):
                   best = R[:]
               return
           
           # Color sort P
           P_sorted = color_sort(P)
           
           # Branch and bound
           # We iterate in reverse color order
           for i in range(len(P_sorted) - 1, -1, -1):
               v = P_sorted[i]
               # Upper bound: current size + color of v
               if len(R) + colors[v] <= len(best):
                   break
               
               # Branch
               expand(R + [v], [u for u in P_sorted[:i] if (adj[v] >> u) & 1])
               
               # Remove v from P for subsequent iterations in this level
               # (Implicitly handled by slicing in recursive call, 
               # but we need to ensure we don't use v again in this level's loop?
               # The standard Tomita algorithm modifies P in place or uses indices.
               # Using list comprehension creates new lists, which is safe but slower.
               # For n=64, this is fine.
       
       expand([], list(range(n)))
       return best
   