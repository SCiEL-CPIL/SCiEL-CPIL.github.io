"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    """
    Construct a valid two-coloring of triples on n vertices with no red K4^(3)
    and no blue K5^(3), maximizing n.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    def _build_lookup(colors, n):
        """Build a 3D numpy array lookup for O(1) triple color access."""
        lookup = np.full((n, n, n), -1, dtype=np.int8)
        idx = 0
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    lookup[a, b, c] = colors[idx]
                    idx += 1
        return lookup

    def _validate(lookup, n):
        """
        Check if a coloring (given as lookup table) is valid.
        Returns True if no red K4^(3) and no blue K5^(3).
        """
        # Check Red K4: 4 vertices, all 4 triples red (0)
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (lookup[a, b, c] == 0 and 
                lookup[a, b, d] == 0 and 
                lookup[a, c, d] == 0 and 
                lookup[b, c, d] == 0):
                return False
        
        # Check Blue K5: 5 vertices, all 10 triples blue (1)
        for quint in combinations(range(n), 5):
            v0, v1, v2, v3, v4 = quint
            if (lookup[v0, v1, v2] == 1 and 
                lookup[v0, v1, v3] == 1 and 
                lookup[v0, v1, v4] == 1 and 
                lookup[v0, v2, v3] == 1 and 
                lookup[v0, v2, v4] == 1 and 
                lookup[v0, v3, v4] == 1 and 
                lookup[v1, v2, v3] == 1 and 
                lookup[v1, v2, v4] == 1 and 
                lookup[v1, v3, v4] == 1 and 
                lookup[v2, v3, v4] == 1):
                return False
        
        return True

    def _count_violations(lookup, n):
        """Count the number of red K4s and blue K5s."""
        count = 0
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (lookup[a, b, c] == 0 and 
                lookup[a, b, d] == 0 and 
                lookup[a, c, d] == 0 and 
                lookup[b, c, d] == 0):
                count += 1
        
        for quint in combinations(range(n), 5):
            v0, v1, v2, v3, v4 = quint
            if (lookup[v0, v1, v2] == 1 and 
                lookup[v0, v1, v3] == 1 and 
                lookup[v0, v1, v4] == 1 and 
                lookup[v0, v2, v3] == 1 and 
                lookup[v0, v2, v4] == 1 and 
                lookup[v0, v3, v4] == 1 and 
                lookup[v1, v2, v3] == 1 and 
                lookup[v1, v2, v4] == 1 and 
                lookup[v1, v3, v4] == 1 and 
                lookup[v2, v3, v4] == 1):
                count += 1
        return count

    def _generate_candidate(n, seed_idx):
        """
        Generate a candidate coloring for n vertices using various strategies.
        """
        random.seed(seed_idx)
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        colors = [0] * num_triples
        
        strategy = seed_idx % 8
        
        if strategy == 0:
            # Sum mod n
            for i, (a, b, c) in enumerate(triples):
                colors[i] = 0 if (a + b + c) % n < n // 2 else 1
        elif strategy == 1:
            # Random
            colors = [random.randint(0, 1) for _ in range(num_triples)]
        elif strategy == 2:
            # Difference based
            for i, (a, b, c) in enumerate(triples):
                if b - a == c - b:
                    colors[i] = 0
                else:
                    colors[i] = 1
        elif strategy == 3:
            # XOR based
            for i, (a, b, c) in enumerate(triples):
                colors[i] = (a ^ b ^ c) & 1
        elif strategy == 4:
            # Product mod n
            for i, (a, b, c) in enumerate(triples):
                colors[i] = 0 if (a * b * c) % n < n // 2 else 1
        elif strategy == 5:
            # Sum mod 3
            for i, (a, b, c) in enumerate(triples):
                colors[i] = 0 if (a + b + c) % 3 == 0 else 1
        elif strategy == 6:
            # Sum mod 5
            for i, (a, b, c) in enumerate(triples):
                colors[i] = 0 if (a + b + c) % 5 < 2 else 1
        else:
            # Random with bias
            p = 0.5
            colors = [1 if random.random() < p else 0 for _ in range(num_triples)]
        
        return colors

    def _local_search(colors, n, max_iters=20000, time_limit=30.0):
        """
        Perform local search to reduce violations.
        Returns (colors, violations).
        """
        start_ls = time.time()
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        current_colors = colors[:]
        lookup = _build_lookup(current_colors, n)
        
        best_violations = _count_violations(lookup, n)
        if best_violations == 0:
            return current_colors, 0
        
        best_colors = current_colors[:]
        
        for i in range(max_iters):
            if time.time() > start_ls + time_limit:
                break
            
            # Pick a random triple to flip
            idx = random.randint(0, num_triples - 1)
            old_color = current_colors[idx]
            new_color = 1 - old_color
            
            # Apply flip
            current_colors[idx] = new_color
            a, b, c = triples[idx]
            lookup[a, b, c] = new_color
            
            # Count violations (expensive but necessary)
            violations = _count_violations(lookup, n)
            
            if violations < best_violations:
                best_violations = violations
                best_colors = current_colors[:]
                if best_violations == 0:
                    return best_colors, 0
            else:
                # Revert flip
                current_colors[idx] = old_color
                lookup[a, b, c] = old_color
        
        return best_colors, best_violations

    # Try to find a valid coloring for n=34
    best_n = 0
    best_colors = None
    
    # Precompute triples for n=34
    n = 34
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    
    # Try multiple seeds with local search
    for seed in range(50):
        if time.time() > deadline:
            break
        
        colors = _generate_candidate(n, seed)
        lookup = _build_lookup(colors, n)
        
        if _validate(lookup, n):
            best_n = n
            best_colors = colors
            break
        
        # Try local search
        colors_ls, violations = _local_search(colors, n, max_iters=5000, time_limit=2.0)
        
        if violations == 0:
            best_n = n
            best_colors = colors_ls
            break
    
    if best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
    
    # Fallback: try smaller n values
    for n in range(33, 10, -1):
        if time.time() > deadline:
            break
        
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        
        for seed in range(20):
            if time.time() > deadline:
                break
            
            colors = _generate_candidate(n, seed)
            lookup = _build_lookup(colors, n)
            
            if _validate(lookup, n):
                best_n = n
                best_colors = colors
                break
        
        if best_colors is not None:
            break
    
    if best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
    
    # Final fallback: return a small valid coloring
    # n=9 with mod 3
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 
              for a in range(n) for b in range(a+1, n) for c in range(b+1, n)]
    
    lookup = _build_lookup(colors, n)
    if _validate(lookup, n):
        return {'n': n, 'colors': colors}
    
    # If even that fails, return n=1
    return {'n': 1, 'colors': [0]}

# EVOLVE-BLOCK-END
