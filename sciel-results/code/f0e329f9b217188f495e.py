"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = 16
    target_edges = 65
    
    def build_adj(edges):
        """Build adjacency bitmasks and degrees from edge list."""
        adj = [0] * n
        deg = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg
    
    def compute_score_from_adj(adj, deg):
        """Compute max triangle degree sum using bitmasks."""
        max_sum = 0
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    continue
                # Common neighbors of i and j, with index > j
                common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
                while common:
                    k = common & (-common)
                    k_idx = k.bit_length() - 1
                    s = deg[i] + deg[j] + deg[k_idx]
                    if s > max_sum:
                        max_sum = s
                    common &= common - 1
        return max_sum
    
    def compute_score(edges):
        """Compute the maximum triangle degree sum."""
        adj, deg = build_adj(edges)
        return compute_score_from_adj(adj, deg)

    def is_valid(edges):
        """Check if edges form a simple graph with exactly target_edges edges."""
        if len(edges) != target_edges:
            return False
        seen = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in seen:
                return False
            seen.add(key)
        return True

    # Base Construction: K_{6,10} + 5 matching edges in the 10-part
    # Vertices 0-5 (Small), 6-15 (Large)
    base_edges = [[u, v] for u in range(6) for v in range(6, 16)]
    # Add 5 disjoint edges in Large part
    internal_edges = [[6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    best_edges = base_edges + internal_edges
    best_score = compute_score(best_edges) # Should be 24
    
    # Local Search
    def local_search(edges, max_iters=20000):
        current = edges[:]
        adj, deg = build_adj(current)
        current_score = compute_score_from_adj(adj, deg)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 5.0:
                break
            
            # Randomly pick an edge to remove
            idx = random.randint(0, len(current) - 1)
            u1, v1 = current[idx]
            
            # Try to add a random non-edge
            found_swap = False
            for _ in range(20):
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 == v2:
                    continue
                u2, v2 = min(u2, v2), max(u2, v2)
                
                # Check if (u2, v2) is already an edge
                if (adj[u2] >> v2) & 1:
                    continue
                
                # Check if (u2, v2) is the same as (u1, v1)
                if (u2, v2) == (min(u1,v1), max(u1,v1)):
                    continue

                # Perform swap
                new_adj = adj[:]
                new_deg = deg[:]
                
                # Remove (u1, v1)
                new_adj[u1] &= ~(1 << v1)
                new_adj[v1] &= ~(1 << u1)
                new_deg[u1] -= 1
                new_deg[v1] -= 1
                
                # Add (u2, v2)
                new_adj[u2] |= (1 << v2)
                new_adj[v2] |= (1 << u2)
                new_deg[u2] += 1
                new_deg[v2] += 1
                
                new_score = compute_score_from_adj(new_adj, new_deg)
                
                if new_score < current_score:
                    # Accept
                    adj = new_adj
                    deg = new_deg
                    current_score = new_score
                    current[idx] = [u2, v2]
                    improved = True
                    found_swap = True
                    break
        
        return current, current_score

    # Run local search from the base
    ls_edges, ls_score = local_search(best_edges)
    if ls_score < best_score:
        best_score = ls_score
        best_edges = ls_edges[:]

    # Try multiple restarts with random perturbations of the base
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        
        perturbed = base_edges[:]
        # Randomly swap a few internal edges
        for _ in range(random.randint(1, 3)):
            idx = random.randint(len(base_edges), len(base_edges) + 4) # Only swap internal ones? Or any?
            # Let's swap any edge for more diversity
            idx = random.randint(0, len(perturbed) - 1)
            u1, v1 = perturbed[idx]
            
            # Find a random non-edge
            while True:
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 == v2:
                    continue
                u2, v2 = min(u2, v2), max(u2, v2)
                # Check if edge exists in perturbed
                exists = False
                for a, b in perturbed:
                    if (a == u2 and b == v2) or (a == v2 and b == u2):
                        exists = True
                        break
                if not exists:
                    break
            perturbed[idx] = [u2, v2]
        
        if not is_valid(perturbed):
            continue
            
        p_score = compute_score(perturbed)
        
        # Local search from perturbation
        ls_p_edges, ls_p_score = local_search(perturbed, max_iters=5000)
        
        if ls_p_score < best_score:
            best_score = ls_p_score
            best_edges = ls_p_edges[:]
            
        if p_score < best_score:
            best_score = p_score
            best_edges = perturbed[:]

    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
