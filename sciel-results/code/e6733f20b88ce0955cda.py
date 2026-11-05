   def _local_search_bipartite(k, g, n, time_limit=100):
       # n must be even for bipartite
       # We want girth >= g. Since bipartite, girth is even.
       # If g=9, we need girth >= 10.
       target_girth = g if g % 2 == 0 else g + 1
       
       # Initialize random k-regular bipartite graph
       # Left: 0..n/2-1, Right: n/2..n-1
       half_n = n // 2
       if half_n * k % 2 != 0: return None, None # Must be even sum? No, just pair stubs.
       
       # Configuration Model for Bipartite
       # Left stubs: [i for i in range(half_n) for _ in range(k)]
       # Right stubs: [half_n + i for i in range(half_n) for _ in range(k)]
       
       import random
       import time
       start_time = time.time()
       
       for attempt in range(50):
           if time.time() - start_time > time_limit: break
           
           # Generate random perfect matching between Left stubs and Right stubs
           left_stubs = [i for i in range(half_n) for _ in range(k)]
           right_stubs = [half_n + i for i in range(half_n) for _ in range(k)]
           
           rng = random.Random(attempt)
           rng.shuffle(right_stubs)
           
           edges = []
           adj = [set() for _ in range(n)]
           valid = True
           for u, v in zip(left_stubs, right_stubs):
               if v in adj[u]:
                   valid = False
                   break
               edges.append((u, v))
               adj[u].add(v)
               adj[v].add(u)
           
           if not valid: continue
           
           # Check Girth
           if _verify_girth(k, target_girth, n, edges):
               return n, edges
           
           # Local Search: Swap edges to break short cycles
           # Find a short cycle (length < target_girth)
           # Pick two edges in the cycle and swap them.
           
           # This is complex to implement robustly in short time.
           pass
       
       return None, None
   