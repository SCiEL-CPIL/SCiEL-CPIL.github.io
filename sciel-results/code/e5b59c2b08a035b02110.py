import random
import time
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Leave 10s margin
    
    def count_bad_5sets_containing(n, adj_bits, u, v):
        """
        Count bad 5-sets (K5 or I5) that contain both u and v.
        """
        bad = 0
        other_vertices = [i for i in range(n) if i != u and i != v]
        for combo in itertools.combinations(other_vertices, 3):
            x, y, z = combo
            vertices = (u, v, x, y, z)
            
            # Check K5
            is_clique = True
            for i in range(5):
                for j in range(i+1, 5):
                    a, b = vertices[i], vertices[j]
                    if not ((adj_bits[a] >> b) & 1):
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                bad += 1
                continue
            
            # Check Independent
            is_indep = True
            for i in range(5):
                for j in range(i+1, 5):
                    a, b = vertices[i], vertices[j]
                    if (adj_bits[a] >> b) & 1:
                        is_indep = False
                        break
                if not is_indep:
                    break
            if is_indep:
                bad += 1
        return bad

    def is_valid(n, adj_bits):
        # Full check
        for combo in itertools.combinations(range(n), 5):
            x, y, z, w, t = combo
            # Check K5
            if (adj_bits[x] >> y) & 1 and (adj_bits[x] >> z) & 1 and (adj_bits[x] >> w) & 1 and (adj_bits[x] >> t) & 1 and \
               (adj_bits[y] >> z) & 1 and (adj_bits[y] >> w) & 1 and (adj_bits[y] >> t) & 1 and \
               (adj_bits[z] >> w) & 1 and (adj_bits[z] >> t) & 1 and \
               (adj_bits[w] >> t) & 1:
                return False
            # Check Independent
            if not ((adj_bits[x] >> y) & 1) and not ((adj_bits[x] >> z) & 1) and not ((adj_bits[x] >> w) & 1) and not ((adj_bits[x] >> t) & 1) and \
               not ((adj_bits[y] >> z) & 1) and not ((adj_bits[y] >> w) & 1) and not ((adj_bits[y] >> t) & 1) and \
               not ((adj_bits[z] >> w) & 1) and not ((adj_bits[z] >> t) & 1) and \
               not ((adj_bits[w] >> t) & 1):
                return False
        return True

    # Try to find a graph with n=42
    n = 42
    adj_bits = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    current_bad = sum(count_bad_5sets_containing(n, adj_bits, i, j) for i in range(n) for j in range(i+1, n))
    # Note: This counts each 5-set multiple times (once for each edge in it).
    # A 5-set has 10 edges. So this is 10 * actual_bad_count.
    # This is fine for comparison, but we need to be careful.
    # Actually, let's just use the full count for simplicity and correctness, 
    # but only do it periodically.
    
    # For SA, we need the delta.
    # Let's just do random flips and check validity periodically.
    
    best_adj = adj_bits[:]
    best_bad = 0 # We don't know the full bad count yet
    
    # Initialize full bad count
    full_bad = 0
    for combo in itertools.combinations(range(n), 5):
        x, y, z, w, t = combo
        if (adj_bits[x] >> y) & 1 and (adj_bits[x] >> z) & 1 and (adj_bits[x] >> w) & 1 and (adj_bits[x] >> t) & 1 and \
           (adj_bits[y] >> z) & 1 and (adj_bits[y] >> w) & 1 and (adj_bits[y] >> t) & 1 and \
           (adj_bits[z] >> w) & 1 and (adj_bits[z] >> t) & 1 and \
           (adj_bits[w] >> t) & 1:
            full_bad += 1
            continue
        if not ((adj_bits[x] >> y) & 1) and not ((adj_bits[x] >> z) & 1) and not ((adj_bits[x] >> w) & 1) and not ((adj_bits[x] >> t) & 1) and \
           not ((adj_bits[y] >> z) & 1) and not ((adj_bits[y] >> w) & 1) and not ((adj_bits[y] >> t) & 1) and \
           not ((adj_bits[z] >> w) & 1) and not ((adj_bits[z] >> t) & 1) and \
           not ((adj_bits[w] >> t) & 1):
            full_bad += 1
            
    if full_bad == 0:
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if (adj_bits[i] >> j) & 1]
        return n, edges

    best_bad = full_bad
    T = 10.0
    cool_rate = 0.99
    
    while time.time() < deadline and T > 0.01:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        # Flip
        adj_bits[u] ^= (1 << v)
        adj_bits[v] ^= (1 << u)
        
        # Recalculate full bad count? No, too slow.
        # We will just check validity every 1000 steps.
        # For now, we accept all flips (random walk).
        
        if random.randint(0, 1000) == 0:
            new_bad = 0
            for combo in itertools.combinations(range(n), 5):
                x, y, z, w, t = combo
                if (adj_bits[x] >> y) & 1 and (adj_bits[x] >> z) & 1 and (adj_bits[x] >> w) & 1 and (adj_bits[x] >> t) & 1 and \
                   (adj_bits[y] >> z) & 1 and (adj_bits[y] >> w) & 1 and (adj_bits[y] >> t) & 1 and \
                   (adj_bits[z] >> w) & 1 and (adj_bits[z] >> t) & 1 and \
                   (adj_bits[w] >> t) & 1:
                    new_bad += 1
                    continue
                if not ((adj_bits[x] >> y) & 1) and not ((adj_bits[x] >> z) & 1) and not ((adj_bits[x] >> w) & 1) and not ((adj_bits[x] >> t) & 1) and \
                   not ((adj_bits[y] >> z) & 1) and not ((adj_bits[y] >> w) & 1) and not ((adj_bits[y] >> t) & 1) and \
                   not ((adj_bits[z] >> w) & 1) and not ((adj_bits[z] >> t) & 1) and \
                   not ((adj_bits[w] >> t) & 1):
                    new_bad += 1
            
            if new_bad < best_bad:
                best_bad = new_bad
                best_adj = adj_bits[:]
                if new_bad == 0:
                    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if (adj_bits[i] >> j) & 1]
                    return n, edges
            else:
                # Revert
                adj_bits[u] ^= (1 << v)
                adj_bits[v] ^= (1 << u)
        
        T *= cool_rate

    # Fallback: Paley graph on 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    return q, edges
