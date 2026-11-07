import time
import numpy as np

def sign_edges(graph, budget):
    deadline = time.time() + 85.0  # Safety margin for 90s limit
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Build adjacency list for balancedness check
    # adj[v] = list of (neighbor, edge_index)
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))

    def is_balanced(signs):
        """
        Returns True if the signed graph is balanced (all cycles have product +1).
        A balanced signed graph has a disconnected 2-lift.
        """
        color = [0] * n  # 0: unvisited, 1: +, -1: -
        for start in range(n):
            if color[start] != 0:
                continue
            color[start] = 1
            stack = [start]
            while stack:
                u = stack.pop()
                for v, idx in adj[u]:
                    s = signs[idx]
                    expected = color[u] * s
                    if color[v] == 0:
                        color[v] = expected
                        stack.append(v)
                    else:
                        if color[v] != expected:
                            return False
        return True

    def build_signed(signs):
        S = np.zeros((n, n), dtype=np.float64)
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            S[u, v] = s
            S[v, u] = s
        return S
    
    def objective(signs):
        """
        Returns spectral radius if unbalanced, else infinity.
        """
        if is_balanced(signs):
            return float('inf')
        S = build_signed(signs)
        ev = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(ev)))

    # Exhaustive search for small m (num_edges <= 15)
    if num_edges <= 15:
        best_obj = float('inf')
        best_signs = None
        # Iterate over all 2^num_edges combinations
        for mask in range(1 << num_edges):
            if time.time() > deadline - 5.0:
                break
            signs = np.array([1 if (mask >> i) & 1 else -1 for i in range(num_edges)], dtype=np.int8)
            # Optimization: Quick check for balance?
            # is_balanced is O(E). eigvalsh is O(V^3).
            # For small E, is_balanced is very fast.
            if is_balanced(signs):
                continue
            S = build_signed(signs)
            ev = np.linalg.eigvalsh(S)
            obj = float(np.max(np.abs(ev)))
            if obj < best_obj:
                best_obj = obj
                best_signs = signs.copy()
        
        if best_signs is not None:
            return best_signs.astype(np.int8)
        # Fallback if no unbalanced found (should not happen for prism graphs m>=4)
        return np.ones(num_edges, dtype=np.int8)

    # ILS for larger m
    best_signs = None
    best_obj = float('inf')
    
    def local_search(signs):
        cur_signs = signs.copy()
        # Ensure we start with an unbalanced state?
        # If start is balanced, objective is inf.
        # We need to find a neighbor that is unbalanced.
        # For now, assume we are called with a valid (unbalanced) start or handle inf.
        
        S_cur = build_signed(cur_signs)
        cur_obj = objective(cur_signs)
        
        if cur_obj == float('inf'):
            # Try to find an unbalanced neighbor
            for e in range(num_edges):
                cur_signs[e] *= -1
                if not is_balanced(cur_signs):
                    break
                cur_signs[e] *= -1
            S_cur = build_signed(cur_signs)
            cur_obj = objective(cur_signs)
            if cur_obj == float('inf'):
                return cur_signs, cur_obj

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            best_flip = -1
            best_flip_obj = cur_obj
            
            # Best improvement single flip
            for e in range(num_edges):
                u, v = edges[e]
                old_s = cur_signs[e]
                new_s = -old_s
                cur_signs[e] = new_s
                S_cur[u, v] = new_s
                S_cur[v, u] = new_s
                
                # Check balance first to avoid eigvalsh if balanced
                if is_balanced(cur_signs):
                    o = float('inf')
                else:
                    o = float(np.max(np.abs(np.linalg.eigvalsh(S_cur))))
                
                # Undo
                cur_signs[e] = old_s
                S_cur[u, v] = old_s
                S_cur[v, u] = old_s
                
                if o < best_flip_obj - 1e-12:
                    best_flip_obj = o
                    best_flip = e
            
            if best_flip >= 0:
                e = best_flip
                u, v = edges[e]
                cur_signs[e] = -cur_signs[e]
                S_cur[u, v] = cur_signs[e]
                S_cur[v, u] = cur_signs[e]
                cur_obj = best_flip_obj
                improved = True
        
        return cur_signs, cur_obj

    def perturb(signs, k):
        p_signs = signs.copy()
        if num_edges < k:
            k = num_edges
        idxs = np.random.choice(num_edges, k, replace=False)
        p_signs[idxs] *= -1
        return p_signs

    # Initial solutions
    initial_solutions = []
    rng = np.random.default_rng(42)
    
    # Generate random unbalanced initial solutions
    for _ in range(20):
        s = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        if not is_balanced(s):
            initial_solutions.append(s)
    
    # If no random unbalanced found, force one
    if not initial_solutions:
        s = np.ones(num_edges, dtype=np.int8)
        s[0] = -1 # Flip one edge to make it unbalanced (assuming m>=2)
        initial_solutions.append(s)

    for init in initial_solutions:
        if time.time() > deadline - 5.0:
            break
        ls_signs, ls_obj = local_search(init)
        if ls_obj < best_obj:
            best_obj = ls_obj
            best_signs = ls_signs.copy()
        
        # ILS loop
        cur_signs = ls_signs.copy()
        cur_obj = ls_obj
        while time.time() < deadline - 10.0:
            # Perturb
            k = 3 + int(np.random.rand() * 3) # 3-5
            new_signs = perturb(cur_signs, k)
            
            # Ensure perturbed state is unbalanced?
            # If it becomes balanced, local_search will handle it by flipping to unbalanced.
            
            # Local Search
            ls_signs, ls_obj = local_search(new_signs)
            if ls_obj < cur_obj - 1e-12:
                cur_signs = ls_signs.copy()
                cur_obj = ls_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_signs = cur_signs.copy()
            # If not improved, continue from current best (cur_signs)
            # Or accept with probability? Simple ILS: reject and perturb again from current.
        
        # Move to next initial solution if time permits

    if best_signs is None:
        # Fallback: Find any unbalanced pattern
        s = np.ones(num_edges, dtype=np.int8)
        s[0] = -1
        best_signs = s
        
    return best_signs.astype(np.int8)
