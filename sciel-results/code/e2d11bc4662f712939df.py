"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from itertools import combinations
    import random
    import numpy as np
    
    # Try to find a valid coloring for n=34 using local search
    # The known construction has 34 points
    
    n = 34
    
    # Precompute all triples and their indices
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    
    # Create lookup: triple -> index
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute for each 4-set, which 4 triples it contains
    # and for each 5-set, which 10 triples it contains
    red_forbidden_4sets = []
    for quad in combinations(range(n), 4):
        # 4 triples in a 4-set
        idxs = []
        for i in range(4):
            for j in range(i+1, 4):
                for k in range(j+1, 4):
                    t = (quad[i], quad[j], quad[k])
                    idxs.append(triple_to_idx[t])
        red_forbidden_4sets.append(idxs)
    
    blue_forbidden_5sets = []
    for quint in combinations(range(n), 5):
        # 10 triples in a 5-set
        idxs = []
        for tri in combinations(quint, 3):
            idxs.append(triple_to_idx[tri])
        blue_forbidden_5sets.append(idxs)
    
    # Initialize randomly
    colors = np.random.randint(0, 2, size=num_triples).astype(int)
    
    # Count violations
    def count_violations(colors):
        red_viol = 0
        for idxs in red_forbidden_4sets:
            if all(colors[i] == 0 for i in idxs):
                red_viol += 1
        blue_viol = 0
        for idxs in blue_forbidden_5sets:
            if all(colors[i] == 1 for i in idxs):
                blue_viol += 1
        return red_viol + blue_viol
    
    # Local search: flip a random triple that reduces violations
    best_colors = colors.copy()
    best_viol = count_violations(colors)
    
    # Try multiple random restarts
    max_restarts = 5
    for restart in range(max_restarts):
        if restart > 0:
            colors = np.random.randint(0, 2, size=num_triples).astype(int)
        
        for iteration in range(50000):
            viol = count_violations(colors)
            if viol == 0:
                best_colors = colors.copy()
                break
            
            if viol < best_viol:
                best_colors = colors.copy()
                best_viol = viol
            
            # Pick a random violating 4-set or 5-set and flip one of its triples
            if viol > 0:
                # Find a random violation
                # Sample a few random 4-sets and 5-sets
                found_flip = False
                for _ in range(20):
                    # Try a random 4-set
                    if random.random() < 0.5:
                        quad = random.sample(range(n), 4)
                        quad = tuple(sorted(quad))
                        idxs = [triple_to_idx[t] for t in combinations(quad, 3)]
                        if all(colors[i] == 0 for i in idxs):
                            # Flip one of these
                            flip_idx = idxs[random.randint(0, len(idxs)-1)]
                            colors[flip_idx] = 1 - colors[flip_idx]
                            found_flip = True
                            break
                    else:
                        quint = random.sample(range(n), 5)
                        quint = tuple(sorted(quint))
                        idxs = [triple_to_idx[t] for t in combinations(quint, 3)]
                        if all(colors[i] == 1 for i in idxs):
                            # Flip one of these
                            flip_idx = idxs[random.randint(0, len(idxs)-1)]
                            colors[flip_idx] = 1 - colors[flip_idx]
                            found_flip = True
                            break
                
                if not found_flip:
                    # Just flip a random triple
                    flip_idx = random.randint(0, num_triples - 1)
                    colors[flip_idx] = 1 - colors[flip_idx]
        
        if count_violations(colors) == 0:
            best_colors = colors.copy()
            break
    
    # Check if we found a valid coloring
    if count_violations(best_colors) == 0:
        return {'n': n, 'colors': best_colors.tolist()}
    
    # Fallback: try n=9 with the simple construction
    n_small = 9
    return {'n': n_small, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_small), 3)]}
# EVOLVE-BLOCK-END
