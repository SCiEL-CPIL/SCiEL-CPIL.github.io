import time
import random
import sys

def _popcount(x):
    return bin(x).count('1')

def _has_independent_set_fast(adj_bits, n, k):
    """
    Fast check for independent set of size k using bitsets and Branch and Bound.
    """
    def search(rem, size):
        if size >= k:
            return True
        if rem == 0:
            return False
        
        # Pruning
        if size + _popcount(rem) < k:
            return False
            
        # Choose a vertex to branch on.
        # Heuristic: Pick vertex with highest degree in the remaining subgraph.
        best_v = -1
        best_deg = -1
        temp_rem = rem
        while temp_rem:
            lsb = temp_rem & -temp_rem
            v = lsb.bit_length() - 1
            deg = _popcount(adj_bits[v] & rem)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp_rem ^= lsb
            
        if best_v == -1:
            return False
            
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        new_rem = rem & ~(adj_bits[best_v] | v_bit)
        if search(new_rem, size + 1):
            return True
            
        # Branch 2: Exclude best_v
        new_rem2 = rem & ~v_bit
        if search(new_rem2, size):
            return True
            
        return False
        
    return search((1 << n) - 1, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Fallback: n=39 construction if needed
    # But first try the algebraic construction for n=43
    
    q = 7
    # Generate points of PG(2, q)
    # Points are 1D subspaces of F_q^3
    # Represented by tuples (x, y, z) not all zero, modulo scaling
    
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero element 1
                # Or just use the tuple and handle scaling in adjacency
                # It's easier to generate unique representatives
                # If x != 0, normalize to (1, y/x, z/x)
                # If x == 0, y != 0, normalize to (0, 1, z/y)
                # If x == 0, y == 0, z != 0, normalize to (0, 0, 1)
                
                if x != 0:
                    inv_x = pow(x, q-2, q)
                    rep = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q-2, q)
                    rep = (0, 1, (z * inv_y) % q)
                else:
                    rep = (0, 0, 1)
                    
                if rep not in seen:
                    seen.add(rep)
                    points.append(rep)
    
    n_total = len(points) # Should be q^2 + q + 1 = 57
    
    # Define adjacency
    # P ~ Q iff P . Q = 0 (using standard dot product)
    # This corresponds to the polarity defined by the identity matrix.
    
    adj = [set() for _ in range(n_total)]
    adj_bits = [0] * n_total
    
    for i in range(n_total):
        p = points[i]
        for j in range(i + 1, n_total):
            q_pt = points[j]
            # Dot product
            dot = (p[0]*q_pt[0] + p[1]*q_pt[1] + p[2]*q_pt[2]) % q
            if dot == 0:
                adj[i].add(j)
                adj[j].add(i)
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
                
    # Check for loops?
    # The definition of graph in the task is simple.
    # If i ~ i, it's a loop. We should not include loops in 'edges'.
    # The adjacency sets above only contain i != j because of j in range(i+1, ...).
    # So adj[i] does not contain i.
    # However, if p . p = 0, it is an absolute point.
    # This doesn't affect the simple graph structure, just that it's a "self-polar" point.
    # The graph is C4-free.
    
    # We want a subset of size n_target = 43
    n_target = 43
    if n_target > n_total:
        n_target = n_total
        
    rng = random.Random(42)
    
    best_edges = []
    best_n = 0
    
    # Try to find a valid subset
    # We can try the first n_target vertices, or random ones
    # Since the graph is symmetric, random subsets should be similar in properties.
    
    for attempt in range(100):
        if time.time() > deadline - 5:
            break
            
        if attempt == 0:
            # Try the first n_target vertices
            indices = list(range(n_target))
        else:
            indices = rng.sample(range(n_total), n_target)
            
        # Build induced subgraph
        # Map old indices to new 0..n_target-1
        old_to_new = {old: new for new, old in enumerate(indices)}
        
        sub_adj = [set() for _ in range(n_target)]
        sub_adj_bits = [0] * n_target
        
        for new_i in range(n_target):
            old_i = indices[new_i]
            for new_j in range(new_i + 1, n_target):
                old_j = indices[new_j]
                if old_j in adj[old_i]:
                    sub_adj[new_i].add(new_j)
                    sub_adj[new_j].add(new_i)
                    sub_adj_bits[new_i] |= (1 << new_j)
                    sub_adj_bits[new_j] |= (1 << new_i)
                    
        # Check alpha
        if not _has_independent_set_fast(sub_adj_bits, n_target, 11):
            # Valid
            edges = []
            for u in range(n_target):
                for v in sub_adj[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n_target, 'edges': edges}
            
    # If no subset of size 43 worked, try smaller n
    # Or just return the best we found?
    # We didn't keep the best invalid one.
    # Let's try to construct a valid graph for n=39 using the same method
    # If 43 failed, 39 is likely to succeed if the full graph alpha is close to 11.
    
    # Actually, if the full graph has alpha >= 11, subsets might still have alpha < 11.
    # If random sampling failed, maybe we need to be smarter.
    # But for now, if 43 fails, we can try 39.
    
    for n_try in range(39, 30, -1):
        if time.time() > deadline - 5:
            break
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            indices = rng.sample(range(n_total), n_try)
            old_to_new = {old: new for new, old in enumerate(indices)}
            sub_adj = [set() for _ in range(n_try)]
            sub_adj_bits = [0] * n_try
            for new_i in range(n_try):
                old_i = indices[new_i]
                for new_j in range(new_i + 1, n_try):
                    old_j = indices[new_j]
                    if old_j in adj[old_i]:
                        sub_adj[new_i].add(new_j)
                        sub_adj[new_j].add(new_i)
                        sub_adj_bits[new_i] |= (1 << new_j)
                        sub_adj_bits[new_j] |= (1 << new_i)
            if not _has_independent_set_fast(sub_adj_bits, n_try, 11):
                edges = []
                for u in range(n_try):
                    for v in sub_adj[u]:
                        if u < v:
                            edges.append([u, v])
                return {'n': n_try, 'edges': edges}
                
    # Fallback: Disjoint triangles n=30
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

if __name__ == "__main__":
    res = construct()
    print(f"Constructed graph with n={res['n']}")
