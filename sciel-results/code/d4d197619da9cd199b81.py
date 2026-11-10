<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Colour a triple red exactly when its vertex sum is divisible by 3.
    from itertools import combinations
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 125
    
    def greedy_init(n, seed):
        """Greedy initialization: color each triple to minimize violations."""
        random.seed(seed)
        triples = list(combinations(range(n), 3))
        lookup = {t: i for i, t in enumerate(triples)}
        colors = [None] * len(triples)
        
        # Precompute which triples share vertices
        for i, (a, b, c) in enumerate(triples):
            # Count potential red K4s if this triple is red
            red_count = 0
            for d in range(n):
                if d not in (a, b, c):
                    # Check if other 3 triples in the K4 could be red
                    # For now, just use random
                    pass
            
            # Use random with slight bias
            colors[i] = random.randint(0, 1)
        
        return colors, triples, lookup
    
    def local_search(n, colors, triples, lookup, max_iters):
        """Local search to reduce violations."""
        k4s = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            k4s.append([lookup[(a,b,c)], lookup[(a,b,d)], lookup[(a,c,d)], lookup[(b,c,d)]])
        
        k5s = []
        for quint in combinations(range(n), 5):
            k5s.append([lookup[tri] for tri in combinations(quint, 3)])
        
        colors = list(colors)
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                return None
            
            red_k4 = sum(1 for idxs in k4s if all(colors[i] == 0 for i in idxs))
            blue_k5 = sum(1 for idxs in k5s if all(colors[i] == 1 for i in idxs))
            
            if red_k4 == 0 and blue_k5 == 0:
                return colors
            
            idx = random.randint(0, len(triples) - 1)
            old = colors[idx]
            colors[idx] = 1 - old
            
            new_red_k4 = sum(1 for idxs in k4s if all(colors[i] == 0 for i in idxs))
            new_blue_k5 = sum(1 for idxs in k5s if all(colors[i] == 1 for i in idxs))
            
            if new_red_k4 + new_blue_k5 <= red_k4 + blue_k5:
                pass
            else:
                colors[idx] = old
        
        return None
    
    best_n = 1
    best_colors = []
    
    for n in range(12, 32):
        if time.time() > deadline - 3:
            break
        
        for seed in range(2):
            if time.time() > deadline - 3:
                break
            
            colors, triples, lookup = greedy_init(n, seed * 7 + n)
            result = local_search(n, colors, triples, lookup, max_iters=1000)
            if result is not None:
                best_n = n
                best_colors = result
                break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

This is still going to be slow. I think the realistic expectation is that we'll find valid colorings up to about n=20-25 with this approach, which is better than n=9.

Let me finalize with a cleaner version that's more likely to actually work within the time budget:

