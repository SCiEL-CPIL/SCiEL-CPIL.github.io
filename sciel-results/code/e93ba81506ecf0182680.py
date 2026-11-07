"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    deadline = time.time() + 55.0
    
    # Pre-compute adjacency list for fast connectivity checks
    # adj[u] = list of (v, edge_idx)
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    def is_balanced(signs):
        """
        Check if the signed graph is balanced.
        If balanced, the 2-lift is disconnected (two copies of G).
        If not balanced, the 2-lift is connected.
        Returns True if balanced (invalid lift), False if not balanced (valid lift).
        """
        color = [-1] * n
        color[0] = 0
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v, idx in adj[u]:
                s = signs[idx]
                # Constraint: color[v] == color[u] if s==1, else 1-color[u]
                target_color = color[u] if s == 1 else 1 - color[u]
                if color[v] == -1:
                    color[v] = target_color
                    queue.append(v)
                else:
                    if color[v] != target_color:
                        return False # Not balanced
        return True # Balanced
    
    def build_signed_matrix(signs):
        """Build the signed adjacency matrix of the base graph."""
        A = np.zeros((n, n), dtype=float)
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = build_signed_matrix(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def evaluate(signs):
        """Evaluate a sign vector: return spectral radius if valid, else None."""
        if is_balanced(signs):
            return None
        return spectral_radius(signs)
    
    # Local search with 2-flip moves for better exploration
    def local_search(signs, max_iterations=5000):
        current_signs = signs.copy()
        current_val = evaluate(current_signs)
        if current_val is None:
            return current_signs, None
        
        improved = True
        iteration = 0
        while improved and iteration < max_iterations and time.time() < deadline:
            improved = False
            best_move = None
            best_val = current_val
            
            # Try single flips
            for idx in range(num_edges):
                if time.time() >= deadline:
                    break
                trial_signs = current_signs.copy()
                trial_signs[idx] = -trial_signs[idx]
                if not is_balanced(trial_signs):
                    val = spectral_radius(trial_signs)
                    if val < best_val - 1e-12:
                        best_val = val
                        best_move = (idx, None)
            
            # Try double flips (pairs of edges)
            if best_move is None:
                for i in range(num_edges):
                    if time.time() >= deadline:
                        break
                    for j in range(i + 1, num_edges):
                        trial_signs = current_signs.copy()
                        trial_signs[i] = -trial_signs[i]
                        trial_signs[j] = -trial_signs[j]
                        if not is_balanced(trial_signs):
                            val = spectral_radius(trial_signs)
                            if val < best_val - 1e-12:
                                best_val = val
                                best_move = (i, j)
                        if time.time() >= deadline:
                            break
            
            if best_move is not None:
                i, j = best_move
                current_signs[i] = -current_signs[i]
                if j is not None:
                    current_signs[j] = -current_signs[j]
                current_val = best_val
                improved = True
            
            iteration += 1
        
        return current_signs, current_val

    best_signs = None
    best_val = float('inf')
    
    # Strategy 1: Brute force for small m
    if m <= 5:
        # 2^E possibilities
        for mask in range(1 << num_edges):
            if time.time() >= deadline:
                break
            signs = np.array([1 if (mask >> i) & 1 else -1 for i in range(num_edges)], dtype=int)
            if not is_balanced(signs):
                val = spectral_radius(signs)
                if val < best_val:
                    best_val = val
                    best_signs = signs.copy()
    else:
        # Strategy 2: Random restarts with local search for larger m
        rng = np.random.default_rng(3)
        num_starts = 50
        
        for start in range(num_starts):
            if time.time() >= deadline:
                break
            
            signs = rng.choice([-1, 1], size=num_edges)
            
            # Ensure it's valid
            if not is_balanced(signs):
                signs, val = local_search(signs)
                if val is not None and val < best_val:
                    best_val = val
                    best_signs = signs.copy()
            else:
                # If invalid, try to fix it by flipping a random edge
                for attempt in range(10):
                    trial = signs.copy()
                    idx = rng.integers(0, num_edges)
                    trial[idx] = -trial[idx]
                    if not is_balanced(trial):
                        signs = trial
                        break
            
            if not is_balanced(signs):
                signs, val = local_search(signs)
                if val is not None and val < best_val:
                    best_val = val
                    best_signs = signs.copy()

        # Strategy 3: Intensive local search from best solution
        if best_signs is not None and time.time() < deadline - 1.0:
            signs, val = local_search(best_signs, max_iterations=10000)
            if val is not None and val < best_val:
                best_val = val
                best_signs = signs.copy()

    if best_signs is None:
        # Fallback: Find any valid solution
        rng = np.random.default_rng(42)
        for _ in range(1000):
            signs = rng.choice([-1, 1], size=num_edges)
            if not is_balanced(signs):
                best_signs = signs
                best_val = spectral_radius(signs)
                break
    
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=int)
    
    return best_signs.astype(int)
# EVOLVE-BLOCK-END
