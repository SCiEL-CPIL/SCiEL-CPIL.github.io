import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    """
    Returns an array of signs (-1 or +1) for each edge in the graph.
    Strategy: Simulated Annealing with connectivity constraints.
    """
    
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Precompute edge endpoints as arrays for fast matrix construction
    u_arr = np.array([e[0] for e in edges])
    v_arr = np.array([e[1] for e in edges])
    
    # Precompute adjacency list for connectivity check
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
        
    def is_lift_connected(signs):
        """
        Returns True if the 2-lift is connected.
        The lift is disconnected if and only if there exists a labeling f: V -> {0,1}
        such that signs[i] == (-1)^(f[u] XOR f[v]) for all edges (u,v).
        """
        f = [-1] * n
        f[0] = 0
        queue = deque([0])
        
        while queue:
            u = queue.popleft()
            for v, idx in adj[u]:
                # signs[idx] is +1 or -1
                # If signs[idx] is +1, f[v] should be same as f[u]
                # If signs[idx] is -1, f[v] should be opposite of f[u]
                # f[v] = f[u] XOR (0 if +1 else 1)
                bit = 0 if signs[idx] == 1 else 1
                required_f_v = f[u] ^ bit
                
                if f[v] == -1:
                    f[v] = required_f_v
                    queue.append(v)
                else:
                    if f[v] != required_f_v:
                        return True  # Conflict found, so it's NOT a cut, so lift is connected
        
        return False # No conflict, it IS a cut, so lift is disconnected

    def compute_spectral_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        S = np.zeros((n, n), dtype=np.float64)
        S[u_arr, v_arr] = signs.astype(np.float64)
        S[v_arr, u_arr] = signs.astype(np.float64)
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))
    
    def get_random_valid_signs(rng):
        """Generate a random sign vector that results in a connected lift."""
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        # Ensure connectivity
        # If not connected, flip a random edge and check again
        # This is a simple rejection sampling
        for _ in range(100):
            if is_lift_connected(signs):
                return signs
            # Flip a random edge
            idx = rng.integers(0, num_edges)
            signs[idx] *= -1
        # Fallback: all -1 is usually connected for non-bipartite base? 
        # Prism graphs are bipartite. 
        # For a bipartite graph, all +1 is disconnected (two components).
        # A single -1 on a matching edge makes it connected.
        signs[:] = 1
        # Find a matching edge
        for i, (u, v) in enumerate(edges):
            if abs(u - v) == m:
                signs[i] = -1
                break
        if not is_lift_connected(signs):
            # Try another
            for i, (u, v) in enumerate(edges):
                if abs(u - v) == m:
                    if i != signs.index(-1):
                        signs[i] = -1
                        break
        return signs

    best_signs = None
    best_radius = np.inf
    
    rng = np.random.default_rng(3)
    
    # Simulated Annealing
    def simulated_annealing(initial_signs, max_time=5.0):
        current_signs = initial_signs.copy()
        current_radius = compute_spectral_radius(current_signs)
        
        if not is_lift_connected(current_signs):
            # Should not happen if initialized correctly, but just in case
            return None, np.inf
            
        best_local_signs = current_signs.copy()
        best_local_radius = current_radius
        
        start_time = time.time()
        T = 1.0  # Initial temperature
        cooling_rate = 0.999
        
        while time.time() - start_time < max_time:
            if time.time() > deadline - 2.0:
                break
                
            # Pick a random edge to flip
            idx = rng.integers(0, num_edges)
            old_sign = current_signs[idx]
            current_signs[idx] = -old_sign
            
            new_radius = compute_spectral_radius(current_signs)
            delta = new_radius - current_radius
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                # Accept move
                current_radius = new_radius
                if new_radius < best_local_radius:
                    best_local_radius = new_radius
                    best_local_signs = current_signs.copy()
            else:
                # Reject move
                current_signs[idx] = old_sign
            
            T *= cooling_rate
            if T < 1e-4:
                break
        
        return best_local_signs, best_local_radius

    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline - 10.0:
        num_restarts += 1
        initial_signs = get_random_valid_signs(rng)
        
        # Run SA for a fixed duration, reducing as we approach deadline
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
            
        time_per_restart = min(5.0, remaining / 2.0)
        
        sa_signs, sa_radius = simulated_annealing(initial_signs, max_time=time_per_restart)
        
        if sa_signs is not None and sa_radius < best_radius:
            best_radius = sa_radius
            best_signs = sa_signs.copy()
            
    # Final Polish: Local search on the best solution found
    if best_signs is not None and time.time() < deadline - 2.0:
        current_signs = best_signs.copy()
        current_radius = best_radius
        
        # Hill climbing
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(num_edges):
                if time.time() > deadline - 2.0:
                    break
                old_sign = current_signs[i]
                current_signs[i] = -old_sign
                if is_lift_connected(current_signs):
                    new_radius = compute_spectral_radius(current_signs)
                    if new_radius < current_radius - 1e-12:
                        current_radius = new_radius
                        improved = True
                        if new_radius < best_radius:
                            best_radius = new_radius
                            best_signs = current_signs.copy()
                    else:
                        current_signs[i] = old_sign
                else:
                    current_signs[i] = old_sign

    if best_signs is None:
        # Fallback
        best_signs = np.ones(num_edges, dtype=np.int8)
        # Ensure connectivity
        for i, (u, v) in enumerate(edges):
            if abs(u - v) == m:
                best_signs[i] = -1
                break
        if not is_lift_connected(best_signs):
            # Just in case
            best_signs[:] = -1
            if not is_lift_connected(best_signs):
                # This shouldn't happen for prism graphs
                best_signs[0] = 1

    return best_signs
