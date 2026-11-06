import time
import itertools
import random

def _idx(i, j):
    """Index for pair (i,j) with 0<=i<j<16 in lexicographic order."""
    if i > j:
        i, j = j, i
    return i * 16 - i * (i + 1) // 2 + j - i - 1

def _make_1factorization_cyclic():
    """Standard cyclic 1-factorization of K16. Returns list of 120 colours."""
    # K16 has 15 perfect matchings.
    # Vertices 0..14, 15 is the "infinity" point in the cyclic construction for K16?
    # Standard 1-factorization of K_{2n} uses vertices 0..2n-1.
    # Here n=8, vertices 0..15.
    # The standard method:
    # One fixed vertex 15.
    # For k in 0..14:
    #   Edge (15, k) gets color k.
    #   For j in 1..7:
    #     a = (k + j) % 15
    #     b = (k - j) % 15
    #     Edge (min(a,b), max(a,b)) gets color k.
    
    colors = [0] * 120
    for k in range(15):
        # Edge (15, k)
        i, j = 15, k
        if i > j: i, j = j, i
        colors[_idx(i, j)] = k
        
        for j in range(1, 8):
            a = (k + j) % 15
            b = (k - j) % 15
            if a > b: a, b = b, a
            colors[_idx(a, b)] = k
            
    return colors

def construct():
    t0 = time.time()
    deadline = 145.0 # Safety margin
    
    # Precompute K4s
    all_k4s = list(itertools.combinations(range(16), 4))
    k4_edges = []
    for quad in all_k4s:
        a, b, c, d = quad
        edges = [
            _idx(a, b), _idx(a, c), _idx(a, d),
            _idx(b, c), _idx(b, d), _idx(c, d)
        ]
        k4_edges.append(edges)
        
    # Number of K4s is C(16,4) = 1820
    
    def count_bad(cl):
        bad = 0
        for edges in k4_edges:
            # 6 edges
            e1, e2, e3, e4, e5, e6 = edges
            # Check if at least 5 distinct colors
            # Fast check: 
            # If all 6 same -> bad
            # If 2 colors -> bad
            # If 3 colors -> bad
            # If 4 colors -> bad
            # If 5 or 6 colors -> good
            
            # Simple set check is usually fast enough for 1820 items
            if len({cl[e1], cl[e2], cl[e3], cl[e4], cl[e5], cl[e6]}) < 5:
                bad += 1
        return bad

    # Initial coloring: 1-factorization (15 colors)
    colors = _make_1factorization_cyclic()
    bad = count_bad(colors)
    
    if bad == 0:
        return colors

    # Local Search / Simulated Annealing
    # We want to minimize bad count.
    # If we reach 0, we have a valid 15-coloring.
    
    best_colors = colors[:]
    best_bad = bad
    
    # If initial is already good, return
    if best_bad == 0:
        return best_colors

    # Search
    # We use a simple hill climbing with random restarts or simulated annealing.
    # Given the tight constraint, hill climbing might get stuck.
    # Let's try a simple random walk that accepts improvements.
    
    num_colors = 15
    current_colors = colors[:]
    current_bad = bad
    
    # To speed up count_bad, we can't easily incrementally update it without complex bookkeeping.
    # 1820 K4s * 6 edges = ~11k lookups per count.
    # 1000 iterations = 11M lookups. Should be fast in Python? Maybe 1-2 seconds.
    # We have 145 seconds. We can do many iterations.
    
    for it in range(100000):
        if time.time() > deadline:
            break
            
        # Pick random edge
        edge_idx = random.randint(0, 119)
        old_color = current_colors[edge_idx]
        new_color = random.randint(0, num_colors - 1)
        
        if new_color == old_color:
            continue
            
        current_colors[edge_idx] = new_color
        new_bad = count_bad(current_colors)
        
        if new_bad <= current_bad:
            current_bad = new_bad
            if new_bad < best_bad:
                best_bad = new_bad
                best_colors = current_colors[:]
                if best_bad == 0:
                    return best_colors
        else:
            # Revert
            current_colors[edge_idx] = old_color
            
    # If we found a valid 15-coloring, return it
    if best_bad == 0:
        return best_colors
        
    # If we didn't find a valid 15-coloring, 
    # should we try to find a valid 16-coloring? 
    # Or just fallback to 17?
    # 17 is guaranteed valid. Score 1.0.
    # 15 is target. Score 1.13.
    # 16 is intermediate. Score 17/16 = 1.06.
    
    # Let's try to find a valid 16-coloring if 15 fails quickly?
    # No, if 15 fails, it's likely hard. 
    # The cyclic 1-factorization is a very strong starting point.
    # If local search fails, maybe the 1-factorization structure is the issue?
    # But 15 colors is the theoretical max for a 1-factorization.
    # Any 15-coloring of K16 doesn't have to be a 1-factorization.
    
    # Let's do a few random restarts with random 15-colorings if the 1-factorization search stalls.
    # But time is precious.
    
    # Fallback to 17-coloring to ensure validity and score 1.0
    # The (i+j)%17 coloring is valid.
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
