"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    """
    Find the largest n with a valid 2-coloring of triples (no red K4, no blue K5).
    """
    start_time = time.time()
    deadline = start_time + 138
    
    best_n = 0
    best_colors = None
    
    for n in range(2, 31):
        if time.time() > deadline - 8:
            break
        
        found = False
        num_triples = n * (n-1) * (n-2) // 6
        
        for attempt in range(4):
            if time.time() > deadline - 5:
                break
            
            colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            
            # Quick check: count violations
            # Build triple index
            triple_to_idx = {}
            idx = 0
            for a in range(n):
                for b in range(a+1, n):
                    for c in range(b+1, n):
                        triple_to_idx[(a,b,c)] = idx
                        idx += 1
            
            # Count red K4
            red_k4 = 0
            for w in range(n-3):
                for x in range(w+1, n-2):
                    for y in range(x+1, n-1):
                        for z in range(y+1, n):
                            if (colors[triple_to_idx[(w,x,y)]] == 0 and
                                colors[triple_to_idx[(w,x,z)]] == 0 and
                                colors[triple_to_idx[(w,y,z)]] == 0 and
                                colors[triple_to_idx[(x,y,z)]] == 0):
                                red_k4 += 1
            
            # Count blue K5
            blue_k5 = 0
            for a in range(n-4):
                for b in range(a+1, n-3):
                    for c in range(b+1, n-2):
                        for d in range(c+1, n-1):
                            for e in range(d+1, n):
                                if (colors[triple_to_idx[(a,b,c)]] == 1 and
                                    colors[triple_to_idx[(a,b,d)]] == 1 and
                                    colors[triple_to_idx[(a,b,e)]] == 1 and
                                    colors[triple_to_idx[(a,c,d)]] == 1 and
                                    colors[triple_to_idx[(a,c,e)]] == 1 and
                                    colors[triple_to_idx[(a,d,e)]] == 1 and
                                    colors[triple_to_idx[(b,c,d)]] == 1 and
                                    colors[triple_to_idx[(b,c,e)]] == 1 and
                                    colors[triple_to_idx[(b,d,e)]] == 1 and
                                    colors[triple_to_idx[(c,d,e)]] == 1):
                                    blue_k5 += 1
            
            total = red_k4 + blue_k5
            
            if total == 0:
                best_n = n
                best_colors = colors.tolist()
                found = True
                break
            
            # Local search
            end_time = time.time() + min(20, max(3, (deadline - time.time()) / 3))
            
            while time.time() < end_time:
                if total == 0:
                    break
                
                tri_idx = np.random.randint(num_triples)
                old = colors[tri_idx]
                new = 1 - old
                colors[tri_idx] = new
                
                # Recompute violations (expensive but simple)
                new_red = 0
                for w in range(n-3):
                    for x in range(w+1, n-2):
                        for y in range(x+1, n-1):
                            for z in range(y+1, n):
                                if (colors[triple_to_idx[(w,x,y)]] == 0 and
                                    colors[triple_to_idx[(w,x,z)]] == 0 and
                                    colors[triple_to_idx[(w,y,z)]] == 0 and
                                    colors[triple_to_idx[(x,y,z)]] == 0):
                                    new_red += 1
                
                new_blue = 0
                for a in range(n-4):
                    for b in range(a+1, n-3):
                        for c in range(b+1, n-2):
                            for d in range(c+1, n-1):
                                for e in range(d+1, n):
                                    if (colors[triple_to_idx[(a,b,c)]] == 1 and
                                        colors[triple_to_idx[(a,b,d)]] == 1 and
                                        colors[triple_to_idx[(a,b,e)]] == 1 and
                                        colors[triple_to_idx[(a,c,d)]] == 1 and
                                        colors[triple_to_idx[(a,c,e)]] == 1 and
                                        colors[triple_to_idx[(a,d,e)]] == 1 and
                                        colors[triple_to_idx[(b,c,d)]] == 1 and
                                        colors[triple_to_idx[(b,c,e)]] == 1 and
                                        colors[triple_to_idx[(b,d,e)]] == 1 and
                                        colors[triple_to_idx[(c,d,e)]] == 1):
                                        new_blue += 1
                
                new_total = new_red + new_blue
                if new_total < total:
                    total = new_total
                else:
                    colors[tri_idx] = old
            
            if total == 0:
                best_n = n
                best_colors = colors.tolist()
                found = True
                break
        
        if not found and n > best_n + 1:
            break
    
    if best_colors is None:
        return {'n': 3, 'colors': [0, 0, 0]}
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
