import time
import numpy as np

def sign_edges(graph, budget):
    deadline = time.time() + 85.0  # Safety margin for 90s limit
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    
    def build_signed(signs):
        S = np.zeros((n, n), dtype=np.float64)
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            S[u, v] = s
            S[v, u] = s
        return S
    
    def objective(signs, S=None):
        if S is None:
            S = build_signed(signs)
        ev = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(ev)))
    
    def flip_signs(signs, edge_idx):
        """Return new signs with edge_idx flipped."""
        new_signs = signs.copy()
        new_signs[edge_idx] = -new_signs[edge_idx]
        return new_signs
    
    def update_S_after_flip(S, signs, edge_idx):
        """Update S by flipping the sign of edge edge_idx. Returns updated S."""
        u, v = edges[edge_idx]
        old_sign = signs[edge_idx]
        new_sign = -old_sign
        delta = 2 * new_sign  # S[u,v] changes from old_sign to new_sign, so delta = new_sign - old_sign = 2*new_sign
        S[u, v] += delta
        S[v, u] += delta
        return S
    
    # Identify edge types: Top Cycle, Bottom Cycle, Matching
    # Vertices 0..m-1 are top, m..2m-1 are bottom
    m = n // 2
    top_edges = []
    bottom_edges = []
    match_edges = []
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edges.append(idx)
        elif u >= m and v >= m:
            bottom_edges.append(idx)
        else:
            match_edges.append(idx)
            
    def get_pattern(pattern_id):
        signs = np.ones(num_edges, dtype=np.int8)
        if pattern_id == 0:
            # All +1
            pass
        elif pattern_id == 1:
            # Alternating cycles, +1 matching
            for i, idx in enumerate(top_edges):
                signs[idx] = 1 if i % 2 == 0 else -1
            for i, idx in enumerate(bottom_edges):
                signs[idx] = 1 if i % 2 == 0 else -1
        elif pattern_id == 2:
            # +1 cycles, Alternating matching
            for i, idx in enumerate(match_edges):
                signs[idx] = 1 if i % 2 == 0 else -1
        elif pattern_id == 3:
            # Alternating cycles (shifted), Alternating matching
            for i, idx in enumerate(top_edges):
                signs[idx] = 1 if i % 2 == 0 else -1
            for i, idx in enumerate(bottom_edges):
                signs[idx] = -1 if i % 2 == 0 else 1
            for i, idx in enumerate(match_edges):
                signs[idx] = 1 if i % 2 == 0 else -1
        elif pattern_id == 4:
            # All -1
            signs[:] = -1
        elif pattern_id == 5:
            # -1 cycles, +1 matching
            for i, idx in enumerate(top_edges):
                signs[idx] = -1
            for i, idx in enumerate(bottom_edges):
                signs[idx] = -1
        elif pattern_id == 6:
            # +1 cycles, -1 matching
            for i, idx in enumerate(match_edges):
                signs[idx] = -1
        elif pattern_id == 7:
            # Alternating top, -1 bottom, alternating matching
            for i, idx in enumerate(top_edges):
                signs[idx] = 1 if i % 2 == 0 else -1
            for i, idx in enumerate(bottom_edges):
                signs[idx] = -1
            for i, idx in enumerate(match_edges):
                signs[idx] = 1 if i % 2 == 0 else -1
        return signs

    best_signs = None
    best_obj = float('inf')
    
    def try_improve(signs):
        nonlocal best_signs, best_obj
        cur_signs = signs.copy()
        cur_S = build_signed(cur_signs)
        cur_obj = objective(cur_signs, cur_S)
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_signs = cur_signs.copy()
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            best_flip = -1
            best_flip_obj = cur_obj
            
            # Greedy single flip - use cached S for speed
            for e in range(num_edges):
                old = cur_signs[e]
                cur_signs[e] = -old
                cur_S = update_S_after_flip(cur_S, cur_signs, e)
                o = objective(cur_signs, cur_S)
                # Revert
                cur_S = update_S_after_flip(cur_S, cur_signs, e)
                cur_signs[e] = old
                if o < best_flip_obj - 1e-12:
                    best_flip_obj = o
                    best_flip = e
            
            if best_flip >= 0:
                cur_signs[best_flip] = -cur_signs[best_flip]
                cur_S = update_S_after_flip(cur_S, cur_signs, best_flip)
                cur_obj = best_flip_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_signs = cur_signs.copy()
                improved = True
            else:
                # Try random pair flips to escape local minima
                rng = np.random.default_rng(42)
                found = False
                for _ in range(150):
                    e1 = int(rng.integers(num_edges))
                    e2 = int(rng.integers(num_edges))
                    if e1 == e2:
                        continue
                    cur_signs[e1] = -cur_signs[e1]
                    cur_signs[e2] = -cur_signs[e2]
                    cur_S = update_S_after_flip(cur_S, cur_signs, e1)
                    cur_S = update_S_after_flip(cur_S, cur_signs, e2)
                    o = objective(cur_signs, cur_S)
                    if o < cur_obj - 1e-12:
                        cur_obj = o
                        if cur_obj < best_obj:
                            best_obj = cur_obj
                            best_signs = cur_signs.copy()
                        found = True
                        improved = True
                        break
                    else:
                        # Revert
                        cur_S = update_S_after_flip(cur_S, cur_signs, e2)
                        cur_S = update_S_after_flip(cur_S, cur_signs, e1)
                        cur_signs[e2] = -cur_signs[e2]
                        cur_signs[e1] = -cur_signs[e1]
                if not found:
                    break

    # Start from the best symmetric patterns
    for p in range(8):
        if time.time() > deadline - 5.0:
            break
        pattern = get_pattern(p)
        try_improve(pattern)
        
    # If time remains, do random restarts
    if time.time() < deadline - 15.0:
        rng = np.random.default_rng(3)
        for _ in range(15):
            if time.time() > deadline - 5.0:
                break
            signs = rng.choice([-1, 1], size=num_edges)
            try_improve(signs)
    
    # Final polish pass: if we have time, do more thorough local search on best_signs
    if best_signs is not None and time.time() < deadline - 3.0:
        cur_signs = best_signs.copy()
        cur_S = build_signed(cur_signs)
        cur_obj = objective(cur_signs, cur_S)
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            best_flip = -1
            best_flip_obj = cur_obj
            for e in range(num_edges):
                old = cur_signs[e]
                cur_signs[e] = -old
                cur_S = update_S_after_flip(cur_S, cur_signs, e)
                o = objective(cur_signs, cur_S)
                cur_S = update_S_after_flip(cur_S, cur_signs, e)
                cur_signs[e] = old
                if o < best_flip_obj - 1e-14:
                    best_flip_obj = o
                    best_flip = e
            if best_flip >= 0:
                cur_signs[best_flip] = -cur_signs[best_flip]
                cur_S = update_S_after_flip(cur_S, cur_signs, best_flip)
                cur_obj = best_flip_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_signs = cur_signs.copy()
                improved = True
            else:
                break

    if best_signs is None:
        best_signs = np.array([1] * num_edges, dtype=np.int8)
    
    return best_signs.astype(np.int8)
