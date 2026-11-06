import random
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (150s limit, 35s buffer)
    
    def check_violation_fast(adj_mask, mask, n):
        """Returns True if induced subgraph on 'mask' is regular. Uses bit operations."""
        degs = []
        m = mask
        while m:
            v = (m & -m).bit_length() - 1
            m &= m - 1
            deg = bin(adj_mask[v] & mask).count('1')
            degs.append(deg)
            if len(degs) > 1 and degs[0] != deg:
                return False
        return len(set(degs)) == 1

    def full_check(adj_mask, n):
        """Check all subsets of size >= 6. Returns True if valid."""
        for k in range(6, n + 1):
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                if check_violation_fast(adj_mask, mask, n):
                    return False
        return True

    def build_edges(adj_mask, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj_mask[i] & (1 << j):
                    edges.append([i, j])
        return edges

    # Algebraic construction: Polarity graph from finite field
    # For n vertices, we use a quadratic form over GF(q) where q is prime
    # The polarity graph has vertices as projective points, edges when orthogonal
    # This structure tends to minimize regular induced subgraphs
    
    def polarity_graph(n):
        """Construct a polarity graph on n vertices using finite field geometry."""
        # Find a prime q such that we can embed n vertices
        # For n <= 22, we can use q=7 (projective plane PG(2,7) has 57 points)
        # Or use a simpler construction: random algebraic graph
        
        # Method: Use a quadratic form Q(x) = x_1^2 + x_2^2 + ... + x_n^2 mod p
        # Vertices are non-zero vectors mod p, edges when Q(u+v) != Q(u) + Q(v)
        # This is complex. Let's use a simpler algebraic structure.
        
        # Simpler: Use a Paley-type graph or a graph from a quadratic form
        # For n vertices, label them 0..n-1, edge (i,j) if (i-j)^2 is a quadratic residue mod p
        # where p is a prime > n and p ≡ 3 mod 4
        
        # Find suitable prime
        def is_prime(x):
            if x < 2: return False
            if x == 2: return True
            if x % 2 == 0: return False
            for i in range(3, int(x**0.5)+1, 2):
                if x % i == 0: return False
            return True
        
        # Find prime p ≡ 3 mod 4 with p > n
        p = n + 1
        while True:
            if p % 4 == 3 and is_prime(p):
                break
            p += 1
        
        # Legendre symbol
        def legendre(a, p):
            if a % p == 0:
                return 0
            return pow(a, (p-1)//2, p) == 1
        
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = (i - j) % p
                if legendre(d, p):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    # Try algebraic constructions first
    for n_target in [21, 20]:
        if time.time() > deadline - 30:
            break
        
        adj = polarity_graph(n_target)
        
        # Check if valid
        if full_check(adj, n_target):
            return {'n': n_target, 'edges': build_edges(adj, n_target)}
        
        # If not valid, use local search to fix violations
        # Find violations and flip edges
        violations_found = 0
        for k in range(6, min(11, n_target+1)):
            for combo in itertools.combinations(range(n_target), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                if check_violation_fast(adj, mask, n_target):
                    violations_found += 1
                    if violations_found > 50:
                        break
            if violations_found > 50:
                break
        
        if violations_found == 0:
            return {'n': n_target, 'edges': build_edges(adj, n_target)}
        
        # Local search: flip edges to eliminate violations
        for step in range(200):
            if time.time() > deadline - 30:
                break
            
            # Find a violation
            violation_mask = None
            for k in range(6, min(10, n_target+1)):
                for combo in itertools.combinations(range(n_target), k):
                    mask = 0
                    for v in combo:
                        mask |= (1 << v)
                    if check_violation_fast(adj, mask, n_target):
                        violation_mask = mask
                        break
                if violation_mask:
                    break
            
            if violation_mask is None:
                # No small violations found, do full check
                if full_check(adj, n_target):
                    return {'n': n_target, 'edges': build_edges(adj, n_target)}
                break
            
            # Get vertices in violation
            vs = []
            m = violation_mask
            while m:
                v = (m & -m).bit_length() - 1
                m &= m - 1
                vs.append(v)
            
            # Try flipping each edge in the violation
            best_flip = None
            best_score = float('inf')
            
            for u_idx in range(len(vs)):
                for v_idx in range(u_idx + 1, len(vs)):
                    u, v = vs[u_idx], vs[v_idx]
                    
                    # Temporarily flip
                    if adj[u] & (1 << v):
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        was_present = True
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        was_present = False
                    
                    # Count violations in a sample
                    sample_violations = 0
                    for k in range(6, min(9, n_target+1)):
                        for _ in range(20):
                            combo = random.sample(range(n_target), k)
                            mask = 0
                            for vv in combo:
                                mask |= (1 << vv)
                            if check_violation_fast(adj, mask, n_target):
                                sample_violations += 1
                    
                    if sample_violations < best_score:
                        best_score = sample_violations
                        best_flip = (u, v, was_present)
                    
                    # Revert
                    if was_present:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
            
            if best_flip:
                u, v, was_present = best_flip
                if was_present:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

    # Fallback: Random graph with local search
    for n_target in [20, 18, 15]:
        if time.time() > deadline - 10:
            break
        
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            adj = [0] * n_target
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search
            for step in range(300):
                if time.time() > deadline - 10:
                    break
                
                # Find violation
                violation_mask = None
                for k in range(6, min(10, n_target+1)):
                    for _ in range(50):
                        combo = random.sample(range(n_target), k)
                        mask = 0
                        for vv in combo:
                            mask |= (1 << vv)
                        if check_violation_fast(adj, mask, n_target):
                            violation_mask = mask
                            break
                    if violation_mask:
                        break
                
                if violation_mask is None:
                    if full_check(adj, n_target):
                        return {'n': n_target, 'edges': build_edges(adj, n_target)}
                    break
                
                # Flip random edge in violation
                vs = []
                m = violation_mask
                while m:
                    v = (m & -m).bit_length() - 1
                    m &= m - 1
                    vs.append(v)
                
                u, v = random.sample(vs, 2)
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    
    # Final fallback: small graph
    n = 10
    adj = [0] * n
    random.seed(42)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    if full_check(adj, n):
        return {'n': n, 'edges': build_edges(adj, n)}
    
    return {'n': 5, 'edges': []}
