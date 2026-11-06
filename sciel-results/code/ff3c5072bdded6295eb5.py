# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s wall-clock limit
    
    def get_triples(n):
        """Generate all triples in lexicographic order."""
        return list(combinations(range(n), 3))
    
    def triple_index(triple, n):
        """Map a triple (a,b,c) to its index in lexicographic order."""
        # Precomputed or computed index
        # For n up to 40, C(40,3) = 9880
        pass
    
    def build_index(n):
        """Build mapping from triple to index."""
        idx = {}
        i = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    idx[(a,b,c)] = i
                    i += 1
        return idx
    
    def count_violations(colors, n, idx):
        """Count red K4^3 and blue K5^3 violations."""
        triples = list(combinations(range(n), 3))
        red_count = 0
        blue_count = 0
        
        # Check for red K4^3: 4 vertices where all 4 triples are red
        for quad in combinations(range(n), 4):
            all_red = True
            for tri in combinations(quad, 3):
                if colors[idx[tri]] != 0:
                    all_red = False
                    break
            if all_red:
                red_count += 1
        
        # Check for blue K5^3: 5 vertices where all 10 triples are blue
        for quint in combinations(range(n), 5):
            all_blue = True
            for tri in combinations(quint, 3):
                if colors[idx[tri]] != 1:
                    all_blue = False
                    break
            if all_blue:
                blue_count += 1
        
        return red_count, blue_count
    
    def find_violating_triples(colors, n, idx):
        """Find triples that are part of violations."""
        violating = set()
        
        # Red K4^3 violations
        for quad in combinations(range(n), 4):
            all_red = True
            for tri in combinations(quad, 3):
                if colors[idx[tri]] != 0:
                    all_red = False
                    break
            if all_red:
                for tri in combinations(quad, 3):
                    violating.add(idx[tri])
        
        # Blue K5^3 violations
        for quint in combinations(range(n), 5):
            all_blue = True
            for tri in combinations(quint, 3):
                if colors[idx[tri]] != 1:
                    all_blue = False
                    break
            if all_blue:
                for tri in combinations(quint, 3):
                    violating.add(idx[tri])
        
        return violating
    
    def local_search(n, p_red, max_iters=5000):
        """Run local search to find valid coloring."""
        idx = build_index(n)
        n_triples = len(idx)
        
        for restart in range(200):
            if time.time() > deadline - 5:
                break
            
            # Random initial coloring with bias
            colors = np.random.choice([0, 1], size=n_triples, p=[p_red, 1-p_red])
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                
                violating = find_violating_triples(colors, n, idx)
                if not violating:
                    return colors.tolist()
                
                # Pick a random violating triple and flip it
                tri_idx = random.choice(list(violating))
                colors[tri_idx] = 1 - colors[tri_idx]
        
        return None
    
    # Try increasing n
    best_n = 1
    best_colors = [0]  # trivial valid coloring for n=1
    
    for n in range(2, 35):
        if time.time() > deadline - 10:
            break
        
        result = local_search(n, 0.5, max_iters=3000)
        if result is not None:
            best_n = n
            best_colors = result
            # Try to go higher
            continue
        else:
            # Try with different biases
            for p in [0.3, 0.4, 0.6, 0.7]:
                if time.time() > deadline - 10:
                    break
                result = local_search(n, p, max_iters=2000)
                if result is not None:
                    best_n = n
                    best_colors = result
                    break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
