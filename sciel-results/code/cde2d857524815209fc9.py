import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 20
    all_mask = (1 << n) - 1
    
    def common_count(mask, i, j, k):
        return (mask[i] & mask[j] & mask[k]).bit_count()
    
    def is_valid(mask):
        for i in range(n):
            mi = mask[i]
            for j in range(i + 1, n):
                mij = mi & mask[j]
                for k in range(j + 1, n):
                    if (mij & mask[k]).bit_count() >= 3:
                        return False
        return True
    
    def edge_count(mask):
        return sum(mask[i].bit_count() for i in range(n)) // 2
    
    def find_violations(mask):
        viols = []
        for i in range(n):
            mi = mask[i]
            for j in range(i + 1, n):
                mij = mi & mask[j]
                for k in range(j + 1, n):
                    if (mij & mask[k]).bit_count() >= 3:
                        viols.append((i, j, k))
        return viols

    def check_add(mask, u, v):
        # Check if adding edge (u, v) creates a violation
        # We must check triples involving u or v
        # Triples {u, v, w}: Common neighbors are N(u) \cap N(v) \cap N(w)
        # Note: Adding (u,v) does not change N(u) \cap N(v) \cap N(w) for w != u,v
        # because u is not in N(u) and v is not in N(v).
        # So we only need to check triples {u, w, z} and {v, w, z} where w,z != u,v
        # And we must account for the new edge in the masks.
        
        # Instead of modifying mask and reverting, we can simulate.
        # But modifying and reverting is O(1) per bit set.
        # Let's just modify, check, and revert.
        
        # Save original
        mu = mask[u]
        mv = mask[v]
        
        # Add edge
        mask[u] = mu | (1 << v)
        mask[v] = mv | (1 << u)
        
        valid = True
        # Check triples involving u
        for w in range(n):
            if w == u or w == v:
                continue
            mw = mask[w]
            # Triple {u, v, w}
            if (mask[u] & mask[v] & mw).bit_count() >= 3:
                valid = False
                break
            for z in range(w + 1, n):
                if z == u or z == v:
                    continue
                mz = mask[z]
                # Triple {u, w, z}
                if (mask[u] & mw & mz).bit_count() >= 3:
                    valid = False
                    break
            if not valid:
                break
            # Triple {v, w, z}
            for z in range(w + 1, n):
                if z == u or z == v:
                    continue
                mz = mask[z]
                if (mask[v] & mw & mz).bit_count() >= 3:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            # Revert
            mask[u] = mu
            mask[v] = mv
            return False
        
        # If we got here, we also need to check triples involving v but not u?
        # My loop structure:
        # w iterates all.
        # If w != u,v:
        #   Check {u,v,w}
        #   z iterates > w.
        #   Check {u,w,z}
        #   Check {v,w,z}
        # This covers all triples containing u or v?
        # Triples with u: {u,v,w}, {u,w,z}. Covered.
        # Triples with v: {u,v,w}, {v,w,z}. Covered.
        # Triples with neither u nor v: Unchanged.
        
        # Revert is NOT needed if valid, we keep the change!
        # Wait, the function returns True/False.
        # If True, the mask has been modified.
        # If False, the mask has been reverted.
        # This is a side-effecting function.
        
        return True

    def repair(mask):
        max_iter = 10000
        for _ in range(max_iter):
            if time.time() > deadline:
                return False
            viols = find_violations(mask)
            if not viols:
                return True
            
            i, j, k = viols[0]
            # Remove an edge incident to one of the vertices in the violation
            # Candidates: (i,j), (i,k), (j,k) and edges from i,j,k to others
            # Just pick a random edge incident to i, j, or k
            candidates = []
            for v in range(n):
                if v != i and mask[i] & (1 << v):
                    candidates.append((i, v))
                if v != j and mask[j] & (1 << v):
                    candidates.append((j, v))
                if v != k and mask[k] & (1 << v):
                    candidates.append((k, v))
            
            if not candidates:
                return False
            
            u, v = random.choice(candidates)
            mask[u] &= ~(1 << v)
            mask[v] &= ~(1 << u)
        
        return not find_violations(mask)

    def add_edges(mask):
        improved = True
        while improved:
            if time.time() > deadline:
                return
            improved = False
            # Shuffle edges to avoid bias
            edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not (mask[i] & (1 << j))]
            random.shuffle(edges)
            for u, v in edges:
                if time.time() > deadline:
                    return
                if check_add(mask, u, v):
                    improved = True

    best_mask = None
    best_count = -1
    random.seed(42)
    
    for p in [0.4, 0.5, 0.6, 0.3, 0.7, 0.45, 0.55, 0.65]:
        if time.time() > deadline:
            break
        
        for _ in range(20):
            if time.time() > deadline:
                break
            
            mask = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        mask[i] |= (1 << j)
                        mask[j] |= (1 << i)
            
            if repair(mask):
                add_edges(mask)
                ec = edge_count(mask)
                if ec > best_count:
                    best_count = ec
                    best_mask = mask[:]
            
            # If repair failed, maybe try again with different random seed or just skip
            # Also try local search swaps if we have time?
            # For now, just restarts.

    if best_mask is None:
        # Fallback: empty graph or sparse
        return []
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_mask[i] & (1 << j):
                edges.append([i, j])
    
    return edges
