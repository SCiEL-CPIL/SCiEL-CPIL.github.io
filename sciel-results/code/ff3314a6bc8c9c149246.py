import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    """
    Returns an array of signs (-1 or +1) for each edge in the graph.
    Strategy: Exploit prism graph symmetry to reduce search space.
    The prism graph has two m-cycles connected by matching edges.
    We enumerate cyclic patterns and use analytical spectral radius computation.
    """
    
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Identify edge types: cycle edges on bottom (0..m-1), cycle edges on top (m..2m-1), matching
    # Bottom cycle: edges (i, i+1 mod m) for i in 0..m-1
    # Top cycle: edges (m+i, m+(i+1 mod m)) for i in 0..m-1  
    # Matching: edges (i, m+i) for i in 0..m-1
    
    # Build edge type mapping
    edge_type = []  # 0=bottom cycle, 1=top cycle, 2=matching
    edge_index = []  # position within its type
    
    for (u, v) in edges:
        if u < m and v < m:
            # Bottom cycle edge
            i = min(u, v)
            j = max(u, v)
            if (i + 1) % m == j or (j + 1) % m == i:
                edge_type.append(0)
                edge_index.append(i)
            else:
                # Shouldn't happen for prism, but handle gracefully
                edge_type.append(2)
                edge_index.append(-1)
        elif u >= m and v >= m:
            # Top cycle edge
            u2 = u - m
            v2 = v - m
            i = min(u2, v2)
            j = max(u2, v2)
            if (i + 1) % m == j or (j + 1) % m == i:
                edge_type.append(1)
                edge_index.append(i)
            else:
                edge_type.append(2)
                edge_index.append(-1)
        else:
            # Matching edge
            i = min(u, v)
            edge_type.append(2)
            edge_index.append(i)
    
    # Verify we have the expected structure
    assert len(edge_type) == num_edges
    type_counts = [edge_type.count(t) for t in range(3)]
    # For a proper prism: m bottom cycle, m top cycle, m matching
    
    def build_signs_from_patterns(bottom_pattern, top_pattern, match_pattern):
        """
        Build full sign vector from cyclic patterns.
        bottom_pattern[i] = sign of bottom cycle edge i (connecting i to i+1 mod m)
        top_pattern[i] = sign of top cycle edge i (connecting m+i to m+i+1 mod m)
        match_pattern[i] = sign of matching edge i (connecting i to m+i)
        """
        signs = np.zeros(num_edges, dtype=np.int8)
        for idx in range(num_edges):
            t = edge_type[idx]
            pos = edge_index[idx]
            if t == 0:
                signs[idx] = bottom_pattern[pos]
            elif t == 1:
                signs[idx] = top_pattern[pos]
            else:
                signs[idx] = match_pattern[pos]
        return signs
    
    def is_lift_connected(signs):
        """
        Returns True if the 2-lift is connected (signing is NOT a coboundary).
        """
        adj = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            adj[u].append((v, i))
            adj[v].append((u, i))
        
        f = [-1] * n
        f[0] = 0
        queue = deque([0])
        
        while queue:
            u = queue.popleft()
            for v, idx in adj[u]:
                s = signs[idx]
                flip = 1 if s == -1 else 0
                required_f_v = f[u] ^ flip
                
                if f[v] == -1:
                    f[v] = required_f_v
                    queue.append(v)
                else:
                    if f[v] != required_f_v:
                        return True
        
        return False
    
    def compute_spectral_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        S = np.zeros((n, n), dtype=np.float64)
        u_arr = np.array([e[0] for e in edges])
        v_arr = np.array([e[1] for e in edges])
        S[u_arr, v_arr] = signs.astype(np.float64)
        S[v_arr, u_arr] = signs.astype(np.float64)
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))
    
    best_signs = None
    best_radius = np.inf
    
    rng = np.random.default_rng(3)
    
    # Strategy 1: Exhaustive search over cyclic patterns for small m
    # For m=4: 2^12 = 4096 patterns
    # For m=5: 2^15 = 32768 patterns
    # For m=7: 2^21 = 2M patterns (might be too many, use sampling)
    # For m=9: 2^27 = 134M patterns (definitely use sampling/local search)
    
    if m <= 5:
        # Exhaustive search over all 2^(3m) patterns
        total_patterns = 1 << (3 * m)
        for k in range(total_patterns):
            if time.time() > deadline - 2.0:
                break
            
            # Decode patterns
            bottom_pattern = np.array([(k >> (i)) & 1 for i in range(m)])
            top_pattern = np.array([(k >> (m + i)) & 1 for i in range(m)])
            match_pattern = np.array([(k >> (2*m + i)) & 1 for i in range(m)])
            
            # Convert 0/1 to -1/+1
            bottom_pattern = np.where(bottom_pattern == 0, -1, 1).astype(np.int8)
            top_pattern = np.where(top_pattern == 0, -1, 1).astype(np.int8)
            match_pattern = np.where(match_pattern == 0, -1, 1).astype(np.int8)
            
            signs = build_signs_from_patterns(bottom_pattern, top_pattern, match_pattern)
            
            if not is_lift_connected(signs):
                continue
            
            radius = compute_spectral_radius(signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        
        if best_signs is not None:
            return best_signs
    
    # Strategy 2: For larger m, use local search over pattern space
    def local_search_patterns(bottom_pattern, top_pattern, match_pattern, max_time=5.0):
        """
        Local search over the three pattern vectors.
        """
        current_bp = bottom_pattern.copy()
        current_tp = top_pattern.copy()
        current_mp = match_pattern.copy()
        
        current_signs = build_signs_from_patterns(current_bp, current_tp, current_mp)
        if not is_lift_connected(current_signs):
            return None, None, None, np.inf
        
        current_radius = compute_spectral_radius(current_signs)
        local_best_bp = current_bp.copy()
        local_best_tp = current_tp.copy()
        local_best_mp = current_mp.copy()
        local_best_radius = current_radius
        
        start_time = time.time()
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() - start_time < max_time and no_improve_count < max_no_improve:
            improved = False
            
            # Try flipping each element in each pattern
            for pattern_name, pattern, setter in [
                ('bottom', current_bp, lambda i, v: setattr(type('obj', (object,), {}), 'val', v) or None),
                ('top', current_tp, None),
                ('match', current_mp, None)
            ]:
                for i in range(m):
                    if time.time() > deadline - 1.0:
                        break
                    
                    old_val = pattern[i]
                    pattern[i] = -old_val
                    
                    signs = build_signs_from_patterns(current_bp, current_tp, current_mp)
                    if not is_lift_connected(signs):
                        pattern[i] = old_val
                        continue
                    
                    radius = compute_spectral_radius(signs)
                    if radius < current_radius - 1e-12:
                        current_radius = radius
                        improved = True
                        no_improve_count = 0
                        if radius < local_best_radius:
                            local_best_radius = radius
                            local_best_bp = current_bp.copy()
                            local_best_tp = current_tp.copy()
                            local_best_mp = current_mp.copy()
                    else:
                        pattern[i] = old_val
            
            if not improved:
                no_improve_count += 1
        
        return local_best_bp, local_best_tp, local_best_mp, local_best_radius
    
    # Multiple restarts with random initial patterns
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 3.0:
            break
        
        # Generate random initial patterns
        bottom_pattern = rng.choice([-1, 1], size=m).astype(np.int8)
        top_pattern = rng.choice([-1, 1], size=m).astype(np.int8)
        match_pattern = rng.choice([-1, 1], size=m).astype(np.int8)
        
        signs = build_signs_from_patterns(bottom_pattern, top_pattern, match_pattern)
        
        # Ensure connectivity
        attempts = 0
        while not is_lift_connected(signs) and attempts < 30:
            # Flip a random element in a random pattern
            pat_choice = rng.integers(0, 3)
            idx = rng.integers(0, m)
            if pat_choice == 0:
                bottom_pattern[idx] *= -1
            elif pat_choice == 1:
                top_pattern[idx] *= -1
            else:
                match_pattern[idx] *= -1
            signs = build_signs_from_patterns(bottom_pattern, top_pattern, match_pattern)
            attempts += 1
        
        if not is_lift_connected(signs):
            continue
        
        # Run local search
        ls_bp, ls_tp, ls_mp, ls_radius = local_search_patterns(
            bottom_pattern, top_pattern, match_pattern, max_time=3.0
        )
        
        if ls_bp is not None and ls_radius < best_radius:
            best_radius = ls_radius
            best_signs = build_signs_from_patterns(ls_bp, ls_tp, ls_mp)
    
    # Final polish: try all single flips in the best solution
    if best_signs is not None and time.time() < deadline - 2.0:
        for i in range(num_edges):
            if time.time() > deadline - 1.0:
                break
            old = best_signs[i]
            best_signs[i] = -old
            if not is_lift_connected(best_signs):
                best_signs[i] = old
                continue
            r = compute_spectral_radius(best_signs)
            if r < best_radius - 1e-12:
                best_radius = r
            else:
                best_signs[i] = old
        
        # Try all double flips
        for i in range(num_edges):
            if time.time() > deadline - 1.0:
                break
            for j in range(i + 1, num_edges):
                if time.time() > deadline - 1.0:
                    break
                si, sj = best_signs[i], best_signs[j]
                best_signs[i] = -si
                best_signs[j] = -sj
                if not is_lift_connected(best_signs):
                    best_signs[i] = si
                    best_signs[j] = sj
                    continue
                r = compute_spectral_radius(best_signs)
                if r < best_radius - 1e-12:
                    best_radius = r
                    si, sj = best_signs[i], best_signs[j]
                else:
                    best_signs[i] = si
                    best_signs[j] = sj
    
    if best_signs is None:
        # Fallback
        best_signs = np.ones(num_edges, dtype=np.int8)
        if not is_lift_connected(best_signs):
            best_signs[0] = -1
            if not is_lift_connected(best_signs):
                best_signs[1] = -1

    return best_signs
