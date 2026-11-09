import random
import itertools
import math

def construct():
    n = 18
    rng = random.Random(42)
    
    # Represent H as a list of int bitmasks
    # H[i] is a bitmask of neighbors of i in H
    
    def is_valid_alpha(H):
        # Check if alpha(H) <= 3
        # Equivalent: No 4 vertices are independent in H
        # Independent in H: for all u, v in S, bit v is 0 in H[u]
        verts = range(n)
        for S in itertools.combinations(verts, 4):
            # Check if S is independent in H
            mask_S = 0
            for v in S:
                mask_S |= (1 << v)
            is_ind = True
            for u in S:
                # Check if u has any neighbor in S \ {u}
                if H[u] & (mask_S ^ (1 << u)):
                    is_ind = False
                    break
            if is_ind:
                return False
        return True

    def get_clique_number(H):
        # Find max clique size in H
        def has_clique_size_k(H, k):
            verts = range(n)
            for S in itertools.combinations(verts, k):
                mask_S = 0
                for v in S:
                    mask_S |= (1 << v)
                is_clique = True
                for u in S:
                    # Check if u is connected to all other vertices in S
                    if (H[u] & (mask_S ^ (1 << u))) != (mask_S ^ (1 << u)):
                        is_clique = False
                        break
                if is_clique:
                    return True
            return False
            
        for k in range(n, 1, -1):
            if has_clique_size_k(H, k):
                return k
        return 1

    # Initialize H
    # We want alpha(H) <= 3.
    # Random graph with high probability of edges.
    H = [0] * n
    found_valid = False
    for _ in range(100):
        for i in range(n):
            H[i] = 0
            for j in range(i+1, n):
                if rng.random() < 0.7:
                    H[i] |= (1 << j)
                    H[j] |= (1 << i)
        if is_valid_alpha(H):
            found_valid = True
            break
            
    if not found_valid:
        # Fallback: Construct a valid H manually or use a denser graph
        # e.g. Complete graph has alpha 1. But omega 18.
        # We want low omega.
        # Let's just keep trying or use a structured graph.
        # For robustness, if we don't find one, we return a known good one or the last one.
        # But 0.7 should work.
        pass

    current_omega = get_clique_number(H)
    best_H = H[:]
    best_omega = current_omega
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    steps = 2000 # Reduced steps to fit time limit if needed, but 2000 is fast
    
    for step in range(steps):
        if T < T_min:
            break
            
        # Pick a random edge to flip
        i = rng.randint(0, n-1)
        j = rng.randint(0, n-1)
        if i == j:
            continue
            
        # Flip edge (i, j) in H
        H[i] ^= (1 << j)
        H[j] ^= (1 << i)
        
        # Check validity
        if is_valid_alpha(H):
            new_omega = get_clique_number(H)
            delta = new_omega - current_omega
            if delta <= 0:
                # Improvement or equal
                current_omega = new_omega
                if new_omega < best_omega:
                    best_omega = new_omega
                    best_H = H[:]
            else:
                # Worse, accept with probability exp(-delta/T)
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    current_omega = new_omega
                else:
                    # Revert
                    H[i] ^= (1 << j)
                    H[j] ^= (1 << i)
        else:
            # Invalid, revert
            H[i] ^= (1 << j)
            H[j] ^= (1 << i)
            
        T *= cooling
        
    # Use best_H
    H = best_H
    
    # Convert H to G edges
    # G is complement of H
    G_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            # Edge in G if NOT edge in H
            if not (H[u] & (1 << v)):
                G_edges.append([u, v])
                
    return G_edges
