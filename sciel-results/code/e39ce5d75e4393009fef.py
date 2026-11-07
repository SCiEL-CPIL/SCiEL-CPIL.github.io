import numpy as np
import time
from itertools import product

def sign_edges(graph, budget):
    m = graph['n'] // 2
    edges = graph['edges']
    num_edges = len(edges)
    n = graph['n']
    
    # Pre-compute indices for faster matrix construction
    u_idx = np.array([e[0] for e in edges])
    v_idx = np.array([e[1] for e in edges])
    
    def compute_radius(signs):
        S = np.zeros((n, n))
        S[u_idx, v_idx] = signs
        S[v_idx, u_idx] = signs
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))
    
    def is_connected_lift(signs):
        """Check if the 2-lift is connected."""
        # Build lift adjacency
        # Vertices in lift: (v, 0) and (v, 1) for v in 0..n-1
        # Lift vertex id: v * 2 + side
        adj = [[] for _ in range(2 * n)]
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                # +1 joins equal sides
                adj[u * 2 + 0].append(v * 2 + 0)
                adj[v * 2 + 0].append(u * 2 + 0)
                adj[u * 2 + 1].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2 + 1)
            else:
                # -1 joins opposite sides
                adj[u * 2 + 0].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2 + 0)
                adj[u * 2 + 1].append(v * 2 + 0)
                adj[v * 2 + 0].append(u * 2 + 1)
        
        # BFS from (0, 0)
        visited = [False] * (2 * n)
        queue = [0]
        visited[0] = True
        while queue:
            node = queue.pop(0)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        
        return all(visited)
    
    def is_valid(signs):
        """Check if the lift is connected and 3-regular."""
        if not is_connected_lift(signs):
            return False
        # 3-regularity is guaranteed by the prism graph structure
        return True
    
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    best_signs = None
    best_radius = float('inf')
    
    # Strategy: For small m, do exhaustive search with connectivity check
    # For larger m, use simulated annealing with connectivity constraint
    
    if m <= 5:
        # Exhaustive search for small m
        for mask in range(2 ** num_edges):
            if time.time() > deadline:
                break
            signs = np.array([1 if (mask >> i) & 1 else -1 for i in range(num_edges)])
            if not is_valid(signs):
                continue
            r = compute_radius(signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
    else:
        # Simulated annealing with connectivity constraint
        rng = np.random.default_rng(3)
        
        # Try multiple restarts
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Random initial assignment
            current_signs = rng.choice([-1, 1], size=num_edges)
            
            # Ensure connectivity; if not, try a few random flips
            valid = False
            for attempt in range(100):
                if is_valid(current_signs):
                    valid = True
                    break
                # Random flip
                i = rng.integers(0, num_edges)
                current_signs[i] = -current_signs[i]
            
            if not valid:
                continue
            
            current_radius = compute_radius(current_signs)
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_signs = current_signs.copy()
            
            # Simulated annealing
            T = 2.0
            cooling = 0.99
            min_T = 0.01
            no_improve_count = 0
            
            while T > min_T and time.time() < deadline:
                i = rng.integers(0, num_edges)
                old_sign = current_signs[i]
                current_signs[i] = -old_sign
                
                if not is_valid(current_signs):
                    # Revert if invalid
                    current_signs[i] = old_sign
                    T *= cooling
                    continue
                
                new_radius = compute_radius(current_signs)
                delta = new_radius - current_radius
                
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_radius = new_radius
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_signs = current_signs.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    current_signs[i] = old_sign
                
                T *= cooling
                
                # Early stop if no improvement for many iterations
                if no_improve_count > 500:
                    break
            
            # Local search polish
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(num_edges):
                    old_sign = current_signs[i]
                    current_signs[i] = -old_sign
                    
                    if not is_valid(current_signs):
                        current_signs[i] = old_sign
                        continue
                    
                    new_radius = compute_radius(current_signs)
                    if new_radius < current_radius:
                        current_radius = new_radius
                        improved = True
                        if current_radius < best_radius:
                            best_radius = current_radius
                            best_signs = current_signs.copy()
                    else:
                        current_signs[i] = old_sign
    
    # Final verification
    if best_signs is None:
        # Fallback: all +1 signs
        best_signs = np.ones(num_edges, dtype=int)
    
    return best_signs
