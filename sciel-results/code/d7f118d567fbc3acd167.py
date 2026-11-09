import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety

    def popcount(x):
        return bin(x).count('1')

    def to_result(n, adj_masks):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj_masks[u] & (1 << v):
                    edges.append([u, v])
        return {'n': int(n), 'edges': edges}

    def generate_paley(n, p=None):
        """Generate a Paley-type graph on n vertices.
        If p is not provided, finds the smallest prime p >= n such that p % 4 == 1.
        """
        if p is None:
            # Find smallest prime p >= n with p % 4 == 1
            candidate = n
            while True:
                if candidate % 4 == 1 and is_prime(candidate):
                    p = candidate
                    break
                candidate += 1
        
        # Quadratic residues modulo p
        residues = set()
        for i in range(1, p):
            residues.add((i * i) % p)
        
        adj_masks = [0] * n
        # Use the first n vertices of the Paley graph on p vertices
        # Edge (u, v) exists if (u - v) % p is in residues
        for u in range(n):
            for v in range(u + 1, n):
                diff = (u - v) % p
                if diff in residues:
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
        return adj_masks

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    def find_violation_bitmask(adj_masks, n, max_k=None):
        """Find a regular induced subgraph of size >= 6 using bitmask iteration.
        Optimized to check k=6 first, then larger.
        Returns (mask, k) or (None, None).
        """
        if max_k is None:
            max_k = n
        
        # Precompute masks for k=6 to speed up the most common case?
        # Or just iterate combinations of indices and convert to mask.
        # Bitmask iteration over all subsets is 2^n. For n=20, that's 1M.
        # Iterating combinations C(n,k) is much smaller for k=6.
        
        for k in range(6, min(max_k, n) + 1):
            # Use itertools for combination generation as it's faster than 
            # manual bitmask next-combination for specific k in Python?
            # Actually, manual bitmask iteration over ALL masks is 2^n.
            # Iterating C(n,k) is much better for small k.
            
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                
                # Check regularity
                first_deg = -1
                is_reg = True
                for v in comb:
                    deg = popcount(adj_masks[v] & mask)
                    if first_deg == -1:
                        first_deg = deg
                    elif deg != first_deg:
                        is_reg = False
                        break
                if is_reg:
                    return mask, k
        return None, None

    def validate_full(adj_masks, n):
        """Check all k >= 6. Returns True if valid."""
        mask, k = find_violation_bitmask(adj_masks, n)
        return mask is None

    def local_search(adj_masks, n, max_steps, time_limit):
        """Local search to eliminate regular induced subgraphs."""
        # Work on a copy
        adj_masks = adj_masks[:]
        
        for step in range(max_steps):
            if time.time() > time_limit:
                return adj_masks, False
            
            mask, k = find_violation_bitmask(adj_masks, n, max_k=10)
            if mask is None:
                # If no violation up to k=10, do a final full check
                # to be safe, but usually k=6,7,8 cover most cases.
                # If time allows, check up to n.
                mask_full, k_full = find_violation_bitmask(adj_masks, n)
                if mask_full is None:
                    return adj_masks, True
                else:
                    mask, k = mask_full, k_full
            
            # Extract vertices from mask
            vertices = []
            temp = mask
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                vertices.append(v)
                temp ^= lsb
            
            if len(vertices) >= 2:
                # Flip a random edge in the violating set
                u, v = random.sample(vertices, 2)
                adj_masks[u] ^= (1 << v)
                adj_masks[v] ^= (1 << u)
        
        return adj_masks, False

    def try_n(n, max_restarts=10, max_steps_per_restart=100):
        """Try to find a valid graph on n vertices."""
        if time.time() > deadline - 10:
            return None
        
        # Strategy: Try Paley initialization first, then random.
        # Paley graphs have high symmetry but might have regular substructures?
        # Random is safer but slower to converge.
        
        init_strategies = []
        
        # 1. Paley
        try:
            adj_paley = generate_paley(n)
            init_strategies.append(adj_paley)
        except Exception:
            pass
            
        # 2. Random
        for _ in range(max_restarts - 1):
            adj_rand = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_rand[i] |= (1 << j)
                        adj_rand[j] |= (1 << i)
            init_strategies.append(adj_rand)

        for init_adj in init_strategies:
            if time.time() > deadline - 10:
                break
            
            adj_masks, success = local_search(init_adj, n, max_steps_per_restart, deadline - 15)
            
            if success:
                # Final validation
                if validate_full(adj_masks, n):
                    return adj_masks
        
        return None

    # Fallback: small valid graph
    n_fallback = 5
    adj_fallback = [0] * n_fallback
    best_result = to_result(n_fallback, adj_fallback)
    
    # Try n values in descending order
    # 21 is the target for improvement. 20 is the known baseline.
    for n_target in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 10:
            break
        
        # Adjust restarts based on n
        # Larger n is harder.
        max_restarts = 20 if n_target <= 18 else 10
        max_steps = 200 if n_target <= 18 else 100
        
        adj = try_n(n_target, max_restarts=max_restarts, max_steps_per_restart=max_steps)
        if adj is not None:
            best_result = to_result(n_target, adj)
            break
    
    return best_result
