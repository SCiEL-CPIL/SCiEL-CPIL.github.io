import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 320  # Safety margin (evaluator kills at 330s)

    # 1. Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # adj[i] is a bitmask of neighbors of i (Python int)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    best_n = n
    best_adj = adj[:]

    def is_k4(a, b, c, d, adj):
        """Check if a,b,c,d form a K4."""
        return ((adj[a] >> b) & 1) and ((adj[a] >> c) & 1) and ((adj[a] >> d) & 1) and \
               ((adj[b] >> c) & 1) and ((adj[b] >> d) & 1) and ((adj[c] >> d) & 1)

    def is_ind4(a, b, c, d, adj):
        """Check if a,b,c,d form an independent set of size 4."""
        return not ((adj[a] >> b) & 1) and not ((adj[a] >> c) & 1) and not ((adj[a] >> d) & 1) and \
               not ((adj[b] >> c) & 1) and not ((adj[b] >> d) & 1) and not ((adj[c] >> d) & 1)

    def precompute_forbidden(adj, n):
        """Precompute forbidden 4-sets: K4s and independent 4-sets.
        Returns (k4_set, ind4_set) where each is a set of frozensets of 4 vertices."""
        k4s = set()
        ind4s = set()
        for combo in itertools.combinations(range(n), 4):
            a, b, c, d = combo
            if is_k4(a, b, c, d, adj):
                k4s.add(combo)
            elif is_ind4(a, b, c, d, adj):
                ind4s.add(combo)
        return k4s, ind4s

    def check_validity_fast(new_mask, adj, n, k4s, ind4s):
        """Check validity using precomputed forbidden sets.
        A new vertex with neighbor mask new_mask is valid iff:
        - No K4 from k4s is entirely contained in the neighbor set
        - No ind4 from ind4s is entirely contained in the non-neighbor set
        """
        # Check: for each K4 (a,b,c,d) in k4s, not all four are neighbors
        for k4 in k4s:
            a, b, c, d = k4
            if ((new_mask >> a) & 1) and ((new_mask >> b) & 1) and \
               ((new_mask >> c) & 1) and ((new_mask >> d) & 1):
                return False
        # Check: for each ind4 (a,b,c,d) in ind4s, not all four are non-neighbors
        for ind4 in ind4s:
            a, b, c, d = ind4
            if not ((new_mask >> a) & 1) and not ((new_mask >> b) & 1) and \
               not ((new_mask >> c) & 1) and not ((new_mask >> d) & 1):
                return False
        return True

    def add_vertex(adj, new_mask):
        n = len(adj)
        new_adj = adj[:]
        new_adj.append(0)
        for i in range(n):
            if (new_mask >> i) & 1:
                new_adj[i] |= (1 << n)
                new_adj[n] |= (1 << i)
        return new_adj

    def find_valid_mask(adj, n, k4s, ind4s, rng):
        """Try to find a valid neighbor mask for a new vertex using iterative repair."""
        # Start with a random mask
        mask = 0
        for i in range(n):
            if rng.random() < 0.5:
                mask |= (1 << i)
        
        # Iterative repair: flip bits to fix violations
        for iteration in range(200):
            if check_validity_fast(mask, adj, n, k4s, ind4s):
                return mask
            
            # Find a violation and try to fix it by flipping a bit
            violated = False
            for k4 in k4s:
                a, b, c, d = k4
                if ((mask >> a) & 1) and ((mask >> b) & 1) and \
                   ((mask >> c) & 1) and ((mask >> d) & 1):
                    # Flip one of these to 0
                    v = [a, b, c, d]
                    rng.shuffle(v)
                    mask &= ~(1 << v[0])
                    violated = True
                    break
            
            if not violated:
                for ind4 in ind4s:
                    a, b, c, d = ind4
                    if not ((mask >> a) & 1) and not ((mask >> b) & 1) and \
                       not ((mask >> c) & 1) and not ((mask >> d) & 1):
                        # Flip one of these to 1
                        v = [a, b, c, d]
                        rng.shuffle(v)
                        mask |= (1 << v[0])
                        violated = True
                        break
            
            if not violated:
                # No specific violation found but still invalid? Shouldn't happen
                break
        
        return None

    # Random restart search
    best_overall_n = best_n
    best_overall_adj = best_adj[:]
    
    num_restarts = 30
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Reset to best known
        current_adj = best_overall_adj[:]
        current_n = best_overall_n
        
        # Use a different random seed for each restart
        rng = random.Random(1000 + restart * 137)
        
        max_add_attempts = 1000
        add_success_count = 0
        
        # Precompute forbidden sets for current graph
        k4s, ind4s = precompute_forbidden(current_adj, current_n)
        
        while current_n < 46 and add_success_count < max_add_attempts:
            if time.time() > deadline:
                break
            
            found = False
            
            # Try multiple random initializations with iterative repair
            for attempt in range(50):
                if time.time() > deadline:
                    break
                
                mask = find_valid_mask(current_adj, current_n, k4s, ind4s, rng)
                if mask is not None:
                    current_adj = add_vertex(current_adj, mask)
                    current_n += 1
                    found = True
                    if current_n > best_overall_n:
                        best_overall_n = current_n
                        best_overall_adj = current_adj[:]
                    add_success_count += 1
                    # Re-precompute forbidden sets for the new graph
                    k4s, ind4s = precompute_forbidden(current_adj, current_n)
                    break
            
            if not found:
                # Try pure random sampling as fallback
                for trial in range(1000):
                    if time.time() > deadline:
                        break
                    mask = 0
                    for i in range(current_n):
                        if rng.random() < 0.5:
                            mask |= (1 << i)
                    if check_validity_fast(mask, current_adj, current_n, k4s, ind4s):
                        current_adj = add_vertex(current_adj, mask)
                        current_n += 1
                        found = True
                        if current_n > best_overall_n:
                            best_overall_n = current_n
                            best_overall_adj = current_adj[:]
                        add_success_count += 1
                        k4s, ind4s = precompute_forbidden(current_adj, current_n)
                        break
            
            if not found:
                break

    # Reconstruct edges from best graph
    best_edges = []
    for i in range(best_overall_n):
        for j in range(i + 1, best_overall_n):
            if (best_overall_adj[i] >> j) & 1:
                best_edges.append((i, j))
    
    return best_overall_n, best_edges
