<<<<<<< SEARCH
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_graph = {'n': 5, 'edges': []}
    
    for target_n in [21, 20]:
        if time.time() > deadline - 5:
            break
            
        print(f"Searching for n={target_n}...")
        
        # Precompute indices for subsets of size 6
        # We only use size 6 for search cost
        import itertools
        subsets_6 = list(itertools.combinations(range(target_n), 6))
        
        # Local Search
        # Adjacency matrix (0/1)
        adj = np.zeros((target_n, target_n), dtype=np.uint8)
        
        def init_random():
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 1
        
        init_random()
        
        def cost_size_6():
            count = 0
            for s in subsets_6:
                # Extract submatrix
                # s is a tuple of 6 indices
                # adj[np.ix_(s, s)]
                # This creates a 6x6 matrix
                # Sum along axis 1
                # Check if all equal
                # Optimization: 
                # degs = np.sum(adj[np.ix_(s, s)], axis=1)
                # if np.all(degs == degs[0]): count += 1
                # This is slow in Python loop.
                
                # Faster:
                # Use bitsets in Python integers is actually quite fast for small n
                # Let's go back to bitsets but optimize the loop
                pass
            return count

        # Let's stick to Python bitsets but optimize the inner loop
        # Precompute masks
        masks_6 = []
        for s in subsets_6:
            m = 0
            for i in s:
                m |= (1 << i)
            masks_6.append(m)
            
        adj_bits = [0] * target_n
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if adj[i, j]:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)

        def fast_cost_size_6():
            c = 0
            # Local variable for speed
            ab = adj_bits
            for m in masks_6:
                # Check regularity
                # Get verts
                # Instead of getting verts, we can just check degrees
                # d_v = popcount(ab[v] & m)
                # We need to check if d_v is constant for all v in m.
                
                # Extract verts
                verts = []
                tmp = m
                while tmp:
                    lsb = tmp & -tmp
                    i = lsb.bit_length() - 1
                    verts.append(i)
                    tmp ^= lsb
                
                d0 = (ab[verts[0]] & m).bit_count()
                is_reg = True
                for v in verts[1:]:
                    if (ab[v] & m).bit_count() != d0:
                        is_reg = False
                        break
                if is_reg:
                    c += 1
            return c

        current_cost = fast_cost_size_6()
        best_cost = current_cost
        best_adj_bits = adj_bits[:]
        
        iterations = 0
        max_iters = 20000 # Adjust based on speed
        
        while time.time() < deadline - 10 and iterations < max_iters:
            iterations += 1
            
            if current_cost == 0:
                # Check full validity
                # Generate masks for k=7..n
                full_valid = True
                for k in range(7, target_n + 1):
                    masks_k = []
                    for s in itertools.combinations(range(target_n), k):
                        m = 0
                        for i in s:
                            m |= (1 << i)
                        masks_k.append(m)
                        
                    for m in masks_k:
                        verts = []
                        tmp = m
                        while tmp:
                            lsb = tmp & -tmp
                            i = lsb.bit_length() - 1
                            verts.append(i)
                            tmp ^= lsb
                            
                        d0 = (adj_bits[verts[0]] & m).bit_count()
                        is_reg = True
                        for v in verts[1:]:
                            if (adj_bits[v] & m).bit_count() != d0:
                                is_reg = False
                                break
                        if is_reg:
                            full_valid = False
                            break
                    if not full_valid:
                        break
                
                if full_valid:
                    edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if adj_bits[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': target_n, 'edges': edges}
            
            # Flip edge
            u = random.randint(0, target_n - 1)
            v = random.randint(0, target_n - 1)
            if u == v: continue
            
            adj_bits[u] ^= (1 << v)
            adj_bits[v] ^= (1 << u)
            
            new_cost = fast_cost_size_6()
            
            if new_cost < current_cost or (new_cost == current_cost and random.random() < 0.1):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj_bits = adj_bits[:]
            else:
                adj_bits[u] ^= (1 << v)
                adj_bits[v] ^= (1 << u)
                
            if iterations % 5000 == 0 and best_cost == current_cost:
                # Restart
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if random.random() < 0.5:
                            adj_bits[i] |= (1 << j)
                            adj_bits[j] |= (1 << i)
                        else:
                            adj_bits[i] &= ~(1 << j)
                            adj_bits[j] &= ~(1 << i)
                current_cost = fast_cost_size_6()

    return best_graph
=======
import numpy as np
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def verify_graph(n, adj_bits):
        """Check if graph has no regular induced subgraph on >= 6 vertices."""
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                m = 0
                for v in subset:
                    m |= (1 << v)
                # Check if induced subgraph is regular
                degs = []
                for v in subset:
                    degs.append((adj_bits[v] & m).bit_count())
                if len(set(degs)) == 1:
                    return False, subset
        return True, None
    
    def spectral_gap(adj):
        """Compute the spectral gap (difference between largest and second-largest eigenvalue in absolute value)."""
        try:
            eigvals = np.linalg.eigvalsh(adj)
            # Sort by absolute value
            abs_eigvals = np.abs(eigvals)
            idx = np.argsort(abs_eigvals)[::-1]
            abs_eigvals = abs_eigvals[idx]
            if len(abs_eigvals) >= 2:
                return abs_eigvals[0] - abs_eigvals[1]
            return 0.0
        except:
            return 0.0
    
    def count_violations_size6(n, adj_bits):
        """Count number of regular induced subgraphs on exactly 6 vertices."""
        count = 0
        for subset in itertools.combinations(range(n), 6):
            m = 0
            for v in subset:
                m |= (1 << v)
            degs = [(adj_bits[v] & m).bit_count() for v in subset]
            if len(set(degs)) == 1:
                count += 1
        return count
    
    def random_graph(n, p=0.5):
        """Generate a random graph with edge probability p."""
        adj_bits = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
        return adj_bits
    
    def edges_from_bits(n, adj_bits):
        """Convert adjacency bits to edge list."""
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj_bits[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    best_graph = {'n': 5, 'edges': []}
    
    for target_n in [21, 20]:
        if time.time() > deadline - 10:
            break
        
        print(f"Searching for n={target_n} with spectral gap heuristic...")
        
        # Phase 1: Generate many random graphs, compute spectral gap, keep top candidates
        candidates = []
        max_candidates = 500
        min_gap_threshold = 0.0
        
        while time.time() < deadline - 30 and len(candidates) < max_candidates:
            # Try different edge probabilities
            p = random.choice([0.3, 0.4, 0.5, 0.6, 0.7])
            adj_bits = random_graph(target_n, p)
            
            # Build adjacency matrix for spectral computation
            adj = np.zeros((target_n, target_n), dtype=np.float64)
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if adj_bits[i] & (1 << j):
                        adj[i, j] = 1.0
                        adj[j, i] = 1.0
            
            gap = spectral_gap(adj)
            
            # Count size-6 violations for quick filtering
            viol6 = count_violations_size6(target_n, adj_bits)
            
            candidates.append({
                'adj_bits': adj_bits,
                'gap': gap,
                'viol6': viol6,
                'p': p
            })
        
        # Sort by spectral gap (descending) and violation count (ascending)
        candidates.sort(key=lambda x: (-x['gap'], x['viol6']))
        
        print(f"Generated {len(candidates)} candidates, top gap: {candidates[0]['gap']:.3f}" if candidates else "No candidates")
        
        # Phase 2: Verify top candidates with full check
        verified = False
        for cand in candidates:
            if time.time() > deadline - 5:
                break
            
            # Quick check: if already has many size-6 violations, skip
            if cand['viol6'] > 10:
                continue
            
            valid, _ = verify_graph(target_n, cand['adj_bits'])
            if valid:
                edges = edges_from_bits(target_n, cand['adj_bits'])
                return {'n': target_n, 'edges': edges}
        
        # Phase 3: If no valid graph found, use local search on best candidate
        if candidates and time.time() < deadline - 10:
            best_cand = candidates[0]
            adj_bits = best_cand['adj_bits'][:]
            
            current_cost = count_violations_size6(target_n, adj_bits)
            best_cost = current_cost
            best_adj_bits = adj_bits[:]
            
            iterations = 0
            max_iters = 50000
            
            while time.time() < deadline - 5 and iterations < max_iters:
                iterations += 1
                
                if current_cost == 0:
                    valid, _ = verify_graph(target_n, adj_bits)
                    if valid:
                        edges = edges_from_bits(target_n, adj_bits)
                        return {'n': target_n, 'edges': edges}
                
                # Flip a random edge
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v:
                    continue
                
                adj_bits[u] ^= (1 << v)
                adj_bits[v] ^= (1 << u)
                
                new_cost = count_violations_size6(target_n, adj_bits)
                
                # Accept if better or with some probability for exploration
                if new_cost < current_cost or (new_cost == current_cost and random.random() < 0.3):
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_adj_bits = adj_bits[:]
                else:
                    adj_bits[u] ^= (1 << v)
                    adj_bits[v] ^= (1 << u)
                
                # Restart if stuck
                if iterations % 5000 == 0 and best_cost == current_cost:
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if random.random() < 0.5:
                                adj_bits[i] |= (1 << j)
                                adj_bits[j] |= (1 << i)
                            else:
                                adj_bits[i] &= ~(1 << j)
                                adj_bits[j] &= ~(1 << i)
                    current_cost = count_violations_size6(target_n, adj_bits)
            
            # Use best found
            if best_cost < current_cost:
                adj_bits = best_adj_bits
                valid, _ = verify_graph(target_n, adj_bits)
                if valid:
                    edges = edges_from_bits(target_n, adj_bits)
                    return {'n': target_n, 'edges': edges}
    
    # Fallback: return a known small valid graph
    return {'n': 5, 'edges': []}
>>>>>>> REPLACE
