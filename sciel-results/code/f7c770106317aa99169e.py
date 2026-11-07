# EVOLVE-BLOCK-START
import time
import numpy as np

def _fast_connectivity_check(graph, signs):
    """Check if 2-lift is connected using cycle product characterization.
    Lift is disconnected iff product of signs on every cycle is +1.
    For prism graph: check top cycle, bottom cycle, and a few 4-cycles."""
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    
    # Build edge index lookup
    edge_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i
    
    # Check top m-cycle: 0-1-2-...-(m-1)-0
    prod_top = 1
    for i in range(m):
        u = i
        v = (i + 1) % m
        if (u, v) in edge_idx:
            prod_top *= signs[edge_idx[(u, v)]]
        else:
            return True  # Edge not found, shouldn't happen
    
    # Check bottom m-cycle: m-(m+1)-...-(2m-1)-m
    prod_bot = 1
    for i in range(m):
        u = m + i
        v = m + (i + 1) % m
        if (u, v) in edge_idx:
            prod_bot *= signs[edge_idx[(u, v)]]
        else:
            return True
    
    # Check 4-cycles: (i, i+1, i+1+m, i+m) for i = 0..m-1
    # If any 4-cycle has product -1, lift is connected
    # If top and bottom both have product +1, need to check 4-cycles
    if prod_top == -1 or prod_bot == -1:
        return True
    
    # Check 4-cycles
    for i in range(m):
        u1 = i
        u2 = (i + 1) % m
        u3 = m + (i + 1) % m
        u4 = m + i
        prod4 = 1
        for (a, b) in [(u1, u2), (u2, u3), (u3, u4), (u4, u1)]:
            if (a, b) in edge_idx:
                prod4 *= signs[edge_idx[(a, b)]]
            else:
                # Try reverse
                if (b, a) in edge_idx:
                    prod4 *= signs[edge_idx[(b, a)]]
                else:
                    return True
        if prod4 == -1:
            return True
    
    return False

def _spectral_radius_of_signed_adj(graph, signs):
    n = graph['n']
    edges = graph['edges']
    S = np.zeros((n, n))
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        S[u, v] = s
        S[v, u] = s
    eigvals = np.linalg.eigvalsh(S)
    return np.max(np.abs(eigvals))

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    best_signs = None
    best_radius = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Precompute edge index lookup
    edge_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i
    
    # Precompute cycle edge indices for fast connectivity
    # Top cycle edges
    top_cycle_edges = []
    for i in range(m):
        u, v = i, (i + 1) % m
        top_cycle_edges.append(edge_idx[(u, v)])
    
    # Bottom cycle edges
    bot_cycle_edges = []
    for i in range(m):
        u, v = m + i, m + (i + 1) % m
        bot_cycle_edges.append(edge_idx[(u, v)])
    
    # 4-cycle edges
    four_cycle_edges = []
    for i in range(m):
        u1, u2 = i, (i + 1) % m
        u3, u4 = m + (i + 1) % m, m + i
        ec = []
        for (a, b) in [(u1, u2), (u2, u3), (u3, u4), (u4, u1)]:
            if (a, b) in edge_idx:
                ec.append(edge_idx[(a, b)])
            else:
                ec.append(edge_idx[(b, a)])
        four_cycle_edges.append(ec)
    
    def fast_is_connected(signs):
        """O(m) connectivity check using precomputed cycle edges."""
        prod_top = 1
        for ei in top_cycle_edges:
            prod_top *= signs[ei]
        if prod_top == -1:
            return True
        
        prod_bot = 1
        for ei in bot_cycle_edges:
            prod_bot *= signs[ei]
        if prod_bot == -1:
            return True
        
        for ec in four_cycle_edges:
            prod4 = 1
            for ei in ec:
                prod4 *= signs[ei]
            if prod4 == -1:
                return True
        
        return False
    
    def evaluate(signs):
        if not fast_is_connected(signs):
            return float('inf')
        return _spectral_radius_of_signed_adj(graph, signs)
    
    # For small m (4, 5), try exhaustive or near-exhaustive search
    if m <= 5:
        # Brute force over all 2^num_edges sign assignments
        # num_edges = 3m, so for m=5, that's 32 edges = 2^32 which is too many
        # Use local search with many restarts
        pass
    
    # Multi-start local search
    num_restarts = 500
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize with random signs
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        
        current_radius = evaluate(signs)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = signs.copy()
        
        if current_radius == float('inf'):
            continue
        
        # Local search: try all single-edge flips, pick best improvement
        improved = True
        while improved:
            if time.time() > deadline:
                break
            
            improved = False
            best_flip = -1
            best_new_radius = current_radius
            
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                
                old_sign = signs[i]
                signs[i] = -old_sign
                new_radius = evaluate(signs)
                signs[i] = old_sign
                
                if new_radius < best_new_radius:
                    best_new_radius = new_radius
                    best_flip = i
            
            if best_flip >= 0:
                signs[best_flip] *= -1
                current_radius = best_new_radius
                improved = True
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_signs = signs.copy()
        
        # 2-edge flip local search
        if time.time() > deadline:
            break
        
        improved2 = True
        while improved2:
            if time.time() > deadline:
                break
            
            improved2 = False
            best_flip_pair = None
            best_new_radius = current_radius
            
            # Sample a subset of pairs to keep it fast
            num_pairs = min(200, num_edges * (num_edges - 1) // 2)
            pair_indices = rng.choice(num_edges * (num_edges - 1) // 2, size=num_pairs, replace=False)
            
            for pi in range(num_pairs):
                if time.time() > deadline:
                    break
                
                # Decode pair index
                # This is slow; use a different approach
                pass
            
            # Simpler: random pairs
            for _ in range(100):
                if time.time() > deadline:
                    break
                
                i, j = rng.choice(num_edges, size=2, replace=False)
                if i > j:
                    i, j = j, i
                
                si, sj = signs[i], signs[j]
                signs[i] *= -1
                signs[j] *= -1
                new_radius = evaluate(signs)
                signs[i] = si
                signs[j] = sj
                
                if new_radius < best_new_radius:
                    best_new_radius = new_radius
                    best_flip_pair = (i, j)
            
            if best_flip_pair is not None:
                i, j = best_flip_pair
                signs[i] *= -1
                signs[j] *= -1
                current_radius = best_new_radius
                improved2 = True
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_signs = signs.copy()
    
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=np.int8)
        if not fast_is_connected(best_signs):
            for _ in range(10000):
                cand = rng.choice([-1, 1], size=num_edges).astype(np.int8)
                if fast_is_connected(cand):
                    best_signs = cand
                    break
    
    return best_signs.astype(int)
# EVOLVE-BLOCK-END
