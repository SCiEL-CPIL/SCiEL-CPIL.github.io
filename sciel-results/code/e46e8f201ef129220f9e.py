"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def sign_edges(graph, budget):
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Precompute edge endpoints as arrays for fast matrix construction
    edge_u = np.array([e[0] for e in edges], dtype=int)
    edge_v = np.array([e[1] for e in edges], dtype=int)
    
    # Precompute adjacency structure for fast connectivity checks
    # For each vertex, store which edges are incident
    vertex_edges = [[] for _ in range(n)]
    for i in range(num_edges):
        u, v = edges[i]
        vertex_edges[u].append(i)
        vertex_edges[v].append(i)
    
    def spectral_radius(signs):
        A = np.zeros((n, n))
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_connected_lift(signs):
        # Fast BFS on the 2-lift using precomputed structure
        # Lift vertex (v, side) has index v*2 + side
        # Edge i connects (u, s) to (v, s*signs[i]) where s in {0,1}
        # Represent side as 0 or 1, and s*sign means: if sign=+1, same side; if sign=-1, flip side
        # side2 = side if signs[i]==1 else 1-side = side ^ (1 if signs[i]==-1 else 0)
        
        # Precompute flip mask: flip[i] = 0 if signs[i]==1, 1 if signs[i]==-1
        flip = (signs == -1).astype(np.int8)
        
        visited = np.zeros(2 * n, dtype=bool)
        visited[0] = True
        queue = [0]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            v = node // 2
            side = node % 2
            for ei in vertex_edges[v]:
                u, w = edges[ei]
                if u == v:
                    continue  # self-loop, shouldn't happen
                other = w if u == v else u  # the other endpoint
                # Wait, need to be careful: edges[ei] = (u, w), and v is one of them
                # Let me redo this
                pass
            # Better approach: just iterate all edges incident to v
            for ei in vertex_edges[v]:
                eu, ev = edges[ei]
                if eu == v:
                    other = ev
                else:
                    other = eu
                other_side = side ^ flip[ei]
                other_node = other * 2 + other_side
                if not visited[other_node]:
                    visited[other_node] = True
                    queue.append(other_node)
        
        return visited.all()
    
    def is_valid(signs):
        return is_connected_lift(signs)
    
    rng = np.random.default_rng(42)
    
    best_signs = None
    best_radius = float('inf')
    
    # Simulated annealing with multiple restarts
    num_restarts = 80
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initial signs
        signs = rng.choice([-1, 1], size=num_edges)
        
        # Ensure valid start
        if not is_valid(signs):
            valid = False
            for _ in range(300):
                if time.time() > deadline:
                    break
                flip_idx = rng.integers(num_edges)
                signs[flip_idx] *= -1
                if is_valid(signs):
                    valid = True
                    break
            if not valid:
                continue
        
        cur_radius = spectral_radius(signs)
        if cur_radius < best_radius:
            best_radius = cur_radius
            best_signs = signs.copy()
        
        # Simulated annealing with adaptive step count based on remaining time
        T_start = 0.05
        T_end = 0.0005
        num_steps = 3000
        step = 0
        
        for step in range(num_steps):
            if time.time() > deadline:
                break
            
            T = T_start * (T_end / T_start) ** (step / num_steps)
            
            # Pick a random edge to flip
            idx = rng.integers(num_edges)
            old_sign = signs[idx]
            signs[idx] = -old_sign
            
            if is_valid(signs):
                new_radius = spectral_radius(signs)
                delta = new_radius - cur_radius
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    cur_radius = new_radius
                    if cur_radius < best_radius:
                        best_radius = cur_radius
                        best_signs = signs.copy()
                else:
                    signs[idx] = old_sign
            else:
                signs[idx] = old_sign
    
    # If no valid solution found, return all +1
    if best_signs is None:
        signs = np.ones(num_edges, dtype=int)
        if not is_valid(signs):
            signs = -np.ones(num_edges, dtype=int)
            if not is_valid(signs):
                for _ in range(1000):
                    signs = rng.choice([-1, 1], size=num_edges)
                    if is_valid(signs):
                        break
        best_signs = signs
    
    return best_signs
# EVOLVE-BLOCK-END
