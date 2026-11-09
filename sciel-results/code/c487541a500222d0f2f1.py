import numpy as np
import time
import sys

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Build adjacency matrix for fast spectral radius calculation
    # We only need the signed adjacency matrix of the base graph
    # A[u,v] = sign, A[v,u] = sign
    # Since it's undirected, A is symmetric.
    
    # Precompute edge indices
    # edges is a list of (u, v)
    
    def calculate_spectral_radius(signs):
        # signs is a numpy array of shape (num_edges,) with -1, 1
        # Construct A
        # For speed, we can update A incrementally, but for simplicity and given small n,
        # constructing from scratch might be fast enough if vectorized.
        # However, np.linalg.eigvalsh is the bottleneck.
        # n is max 18. 18x18 eigvalsh is very fast (~microseconds).
        
        A = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
            
        # Check connectivity of the lift?
        # The task says: "The lift must be connected and 3-regular, or the whole result is invalid."
        # 3-regular is guaranteed by the structure (prism lift is always 3-regular if base is 3-regular? 
        # Base is 3-regular. Lift vertices have degree 3. Yes.)
        # Connectivity of the lift:
        # The lift has 2n vertices.
        # It is connected if and only if the "twist" (product of signs around cycles) 
        # allows reaching both sides.
        # Specifically, if all cycles have product +1, the lift is disconnected (two components).
        # If there is at least one cycle with product -1, the lift is connected.
        # We MUST ensure the lift is connected.
        # So we need at least one "frustrated" cycle (product -1).
        
        # How to check connectivity quickly?
        # We can check if the product of signs on any cycle is -1.
        # Easier: Build the lift graph and check connectivity?
        # Or: Check if the signed graph is "balanced" (all cycles +1). If balanced, invalid.
        # If not balanced, is it always connected?
        # For a connected base graph, the lift is connected iff the signed graph is not balanced.
        # Yes, this is a standard theorem.
        
        # So we need to ensure the signed graph is NOT balanced.
        # Balanced means: there exists v -> {-1,1} such that s_u s_v = sign(u,v).
        # We can check balance by BFS on the base graph.
        
        # But wait, if we just minimize spectral radius, we might end up with a balanced graph (radius might be low but invalid).
        # We must penalize or reject balanced graphs.
        
        # Let's check balance.
        # Assign potential signs to vertices.
        # v[0] = 1
        # For each edge (u,v) with sign s, v[v] must be s * v[u].
        # If we reach a contradiction, it's unbalanced (valid).
        # If no contradiction, it's balanced (invalid).
        
        # Since n is small, we can do this check.
        
        vertex_signs = np.zeros(n, dtype=int)
        vertex_signs[0] = 1
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        queue = [0]
        
        is_balanced = True
        
        while queue:
            u = queue.pop(0)
            for i, (eu, ev) in enumerate(edges):
                s = signs[i]
                if eu == u:
                    v = ev
                elif ev == u:
                    v = eu
                else:
                    continue
                
                if not visited[v]:
                    visited[v] = True
                    vertex_signs[v] = int(vertex_signs[u] * s)
                    queue.append(v)
                else:
                    # Check consistency
                    expected = int(vertex_signs[u] * s)
                    if vertex_signs[v] != expected:
                        is_balanced = False
                        break
            if not is_balanced:
                break
        
        if is_balanced:
            return np.inf # Invalid lift (disconnected)
        
        # Calculate spectral radius
        # A is symmetric
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    # Initialize signs
    # Strategy: Start with a "frustrated" pattern to ensure validity.
    # For m odd, alternating signs on the cycle creates frustration.
    # For m even, all +1 is balanced (invalid). We need at least one -1.
    
    # Let's identify edge types:
    # Edges in cycle 0: (i, i+1) for i=0..m-2, (m-1, 0)
    # Edges in cycle 1: (m+i, m+i+1) ...
    # Matching: (i, m+i)
    
    # We'll just use a random valid initialization or a heuristic one.
    # Heuristic: Set all to 1, then flip one edge in the first cycle to -1 to break balance.
    # Then use local search.
    
    def get_initial_signs():
        signs = np.ones(num_edges, dtype=int)
        # Find an edge in the first cycle to flip
        # First cycle edges are likely the first m edges or interspersed.
        # Let's just flip the first edge.
        signs[0] = -1
        return signs

    current_signs = get_initial_signs()
    current_radius = calculate_spectral_radius(current_signs)
    
    best_signs = current_signs.copy()
    best_radius = current_radius
    
    # If initial is invalid (inf), try random until valid
    while current_radius == np.inf:
        current_signs = np.random.choice([-1, 1], size=num_edges)
        current_radius = calculate_spectral_radius(current_signs)
        if current_radius != np.inf:
            best_signs = current_signs.copy()
            best_radius = current_radius
            break
    
    # Local Search / Simulated Annealing
    # Temperature schedule
    T = 0.5
    T_min = 0.01
    cooling = 0.99
    
    # Precompute edge neighbors for faster update?
    # Since n is small, full recalculation is fine.
    
    while time.time() < deadline:
        # Pick a random edge to flip
        i = np.random.randint(num_edges)
        new_signs = current_signs.copy()
        new_signs[i] *= -1
        
        new_radius = calculate_spectral_radius(new_signs)
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if new_radius < current_radius:
            current_signs = new_signs
            current_radius = new_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_signs = current_signs.copy()
        else:
            delta = new_radius - current_radius
            if delta < 0:
                current_signs = new_signs
                current_radius = new_radius
            else:
                # Accept worse with probability
                if np.random.rand() < np.exp(-delta / T):
                    current_signs = new_signs
                    current_radius = new_radius
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Optional: restart with random perturbation if stuck
            # If no improvement in last X steps, restart
            # For simplicity, just keep going with low T (greedy)
            
    return best_signs
