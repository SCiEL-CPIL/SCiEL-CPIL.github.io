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
    
    # Classify edges by type
    edge_cycle1 = []  # edges within 0..m-1
    edge_cycle2 = []  # edges within m..2m-1
    edge_matching = []  # edges between i and i+m
    
    for idx, (u, v) in enumerate(edges):
        u_min = min(u, v)
        u_max = max(u, v)
        if u_max < m:
            edge_cycle1.append(idx)
        elif u_min >= m:
            edge_cycle2.append(idx)
        else:
            edge_matching.append(idx)
    
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
    
    def is_lift_connected(signs):
        """Check if the 2-lift is connected using BFS."""
        lift_adj = [[] for _ in range(2 * n)]
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                lift_adj[u * 2].append(v * 2)
                lift_adj[v * 2].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2 + 1)
            else:
                lift_adj[u * 2].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2)
                lift_adj[v * 2].append(u * 2 + 1)
        
        visited = [False] * (2 * n)
        queue = deque([0])
        visited[0] = True
        count = 1
        while queue:
            node = queue.popleft()
            for neighbor in lift_adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    queue.append(neighbor)
        return count == 2 * n
    
    def evaluate(signs):
        """Evaluate a sign vector: return spectral radius if valid, else None."""
        if not is_lift_connected(signs):
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
                val = evaluate(trial_signs)
                if val is not None and val < best_val - 1e-12:
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
                        val = evaluate(trial_signs)
                        if val is not None and val < best_val - 1e-12:
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
    
    # Systematic search over gauge-equivalence classes
    # For prism graphs, we can fix the gauge by setting the first edge of each cycle to +1
    # The remaining degrees of freedom are limited
    
    best_signs = None
    best_val = float('inf')
    
    # Strategy 1: Try structured sign patterns
    # Pattern: all +1
    signs = np.ones(num_edges, dtype=int)
    val = evaluate(signs)
    if val is not None:
        signs, val = local_search(signs)
        if val is not None and val < best_val:
            best_val = val
            best_signs = signs.copy()
    
    # Pattern: alternating signs on cycles
    for alt_pattern in [1, -1]:
        signs = np.ones(num_edges, dtype=int)
        for idx in edge_cycle1:
            pos_in_cycle = edge_cycle1.index(idx)
            if pos_in_cycle % 2 == 1:
                signs[idx] = alt_pattern
        for idx in edge_cycle2:
            pos_in_cycle = edge_cycle2.index(idx)
            if pos_in_cycle % 2 == 1:
                signs[idx] = alt_pattern
        val = evaluate(signs)
        if val is not None:
            signs, val = local_search(signs)
            if val is not None and val < best_val:
                best_val = val
                best_signs = signs.copy()
    
    # Strategy 2: Random restarts with local search
    rng = np.random.default_rng(3)
    num_starts = 30
    
    for start in range(num_starts):
        if time.time() >= deadline:
            break
        
        signs = rng.choice([-1, 1], size=num_edges)
        
        val = evaluate(signs)
        if val is not None:
            signs, val = local_search(signs)
            if val is not None and val < best_val:
                best_val = val
                best_signs = signs.copy()
    
    # Strategy 3: Simulated annealing for more thorough exploration
    if time.time() < deadline - 5.0:
        current_signs = best_signs.copy() if best_signs is not None else np.ones(num_edges, dtype=int)
        current_val = evaluate(current_signs)
        
        if current_val is None:
            # Find a valid starting point
            for attempt in range(100):
                trial = rng.choice([-1, 1], size=num_edges)
                v = evaluate(trial)
                if v is not None:
                    current_signs = trial
                    current_val = v
                    break
        
        if current_val is not None:
            T = 0.5
            T_min = 0.01
            cooling = 0.995
            sa_iterations = 0
            max_sa = 50000
            
            while T > T_min and sa_iterations < max_sa and time.time() < deadline - 2.0:
                # Pick a random edge to flip
                idx = rng.integers(0, num_edges)
                trial_signs = current_signs.copy()
                trial_signs[idx] = -trial_signs[idx]
                
                trial_val = evaluate(trial_signs)
                
                if trial_val is not None:
                    if trial_val < current_val:
                        current_signs = trial_signs
                        current_val = trial_val
                        if current_val < best_val:
                            best_val = current_val
                            best_signs = current_signs.copy()
                    else:
                        delta = trial_val - current_val
                        if np.random.rand() < np.exp(-delta / T):
                            current_signs = trial_signs
                            current_val = trial_val
                else:
                    # Invalid lift - penalize heavily
                    delta = 2.0
                    if np.random.rand() < np.exp(-delta / T):
                        current_signs = trial_signs
                        current_val = None
                
                T *= cooling
                sa_iterations += 1
            
            # Final local search from SA result
            if current_val is not None:
                signs, val = local_search(current_signs)
                if val is not None and val < best_val:
                    best_val = val
                    best_signs = signs.copy()
    
    # Final polish: intensive local search from best solution
    if best_signs is not None and time.time() < deadline - 1.0:
        signs, val = local_search(best_signs, max_iterations=20000)
        if val is not None and val < best_val:
            best_val = val
            best_signs = signs.copy()
    
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=int)
    
    return best_signs.astype(int)
# EVOLVE-BLOCK-END
