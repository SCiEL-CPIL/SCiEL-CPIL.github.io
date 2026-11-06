import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    n = 18
    
    def count_bits(x):
        return bin(x).count('1')
    
    def is_triangle_free(adj_mask, mask):
        """Check if the induced subgraph on mask is triangle-free."""
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            for j in range(i+1, n):
                if (mask & (1 << j)) and (adj_mask[i] & (1 << j)):
                    # Edge (i,j) exists in mask
                    common = adj_mask[i] & adj_mask[j] & mask
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    if common:
                        return False
        return True
    
    def largest_triangle_free(adj_mask):
        """Find largest triangle-free induced subgraph using branch and bound."""
        best = 0
        all_vertices = (1 << n) - 1
        
        def branch_and_bound(remaining, current_mask, current_size):
            nonlocal best
            if current_size + count_bits(remaining) <= best:
                return
            if remaining == 0:
                if current_size > best:
                    best = current_size
                return
            
            # Pick a vertex from remaining
            v = None
            min_connections = n + 1
            for i in range(n):
                if remaining & (1 << i):
                    conns = count_bits(adj_mask[i] & current_mask)
                    if conns < min_connections:
                        min_connections = conns
                        v = i
            
            if v is None:
                return
            
            # Option 1: Don't include v
            branch_and_bound(remaining & ~(1 << v), current_mask, current_size)
            
            # Option 2: Include v if it doesn't create a triangle
            neighbors_in_current = adj_mask[v] & current_mask
            creates_triangle = False
            for a in range(n):
                if not (neighbors_in_current & (1 << a)):
                    continue
                for b in range(a+1, n):
                    if (neighbors_in_current & (1 << b)) and (adj_mask[a] & (1 << b)):
                        creates_triangle = True
                        break
                if creates_triangle:
                    break
            
            if not creates_triangle:
                branch_and_bound(remaining & ~(1 << v), current_mask | (1 << v), current_size + 1)
        
        branch_and_bound(all_vertices, 0, 0)
        return best
    
    def is_k4_free(adj_mask):
        """Check if graph is K4-free."""
        for combo in itertools.combinations(range(n), 4):
            u, v, w, x = combo
            if (adj_mask[u] & (1 << v)) and (adj_mask[u] & (1 << w)) and (adj_mask[u] & (1 << x)) and \
               (adj_mask[v] & (1 << w)) and (adj_mask[v] & (1 << x)) and (adj_mask[w] & (1 << x)):
                return False
        return True
    
    def can_add_edge(adj_mask, u, v):
        """Check if adding edge (u,v) keeps graph K4-free."""
        if adj_mask[u] & (1 << v):
            return False
        common = adj_mask[u] & adj_mask[v] & ~((1 << u) | (1 << v))
        for i in range(n):
            if common & (1 << i):
                for j in range(i+1, n):
                    if (common & (1 << j)) and (adj_mask[i] & (1 << j)):
                        return False
        return True
    
    best_score = n
    best_adj_mask = [0] * n
    
    # Random restarts with local search
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Initialize with random sparse graph
        adj_mask = [0] * n
        edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if random.random() < 0.5 and can_add_edge(adj_mask, u, v):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        
        current_score = largest_triangle_free(adj_mask)
        
        if current_score < best_score:
            best_score = current_score
            best_adj_mask = adj_mask[:]
        
        # Local search: try to improve by adding edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_score = largest_triangle_free(adj_mask)
            
            # Find a triangle-free set of size current_score
            target_set = None
            for combo in itertools.combinations(range(n), current_score):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                if is_triangle_free(adj_mask, mask):
                    target_set = list(combo)
                    break
            
            if target_set is None:
                break
            
            # Try adding edges within target_set
            for i in range(len(target_set)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(target_set)):
                    u = target_set[i]
                    v = target_set[j]
                    if not (adj_mask[u] & (1 << v)) and can_add_edge(adj_mask, u, v):
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        
                        new_score = largest_triangle_free(adj_mask)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            break
                        else:
                            adj_mask[u] &= ~(1 << v)
                            adj_mask[v] &= ~(1 << u)
                if improved:
                    break
            
            if current_score < best_score:
                best_score = current_score
                best_adj_mask = adj_mask[:]
        
        # Random perturbation: remove some edges
        if time.time() > deadline:
            break
        
        adj_mask_perturbed = adj_mask[:]
        for _ in range(random.randint(1, 5)):
            existing_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj_mask_perturbed[i] & (1 << j):
                        existing_edges.append((i, j))
            if existing_edges:
                u, v = random.choice(existing_edges)
                adj_mask_perturbed[u] &= ~(1 << v)
                adj_mask_perturbed[v] &= ~(1 << u)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_score = largest_triangle_free(adj_mask_perturbed)
            
            target_set = None
            for combo in itertools.combinations(range(n), current_score):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                if is_triangle_free(adj_mask_perturbed, mask):
                    target_set = list(combo)
                    break
            
            if target_set is None:
                break
            
            for i in range(len(target_set)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(target_set)):
                    u = target_set[i]
                    v = target_set[j]
                    if not (adj_mask_perturbed[u] & (1 << v)) and can_add_edge(adj_mask_perturbed, u, v):
                        adj_mask_perturbed[u] |= (1 << v)
                        adj_mask_perturbed[v] |= (1 << u)
                        
                        new_score = largest_triangle_free(adj_mask_perturbed)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            break
                        else:
                            adj_mask_perturbed[u] &= ~(1 << v)
                            adj_mask_perturbed[v] &= ~(1 << u)
                if improved:
                    break
            
            if current_score < best_score:
                best_score = current_score
                best_adj_mask = adj_mask_perturbed[:]
    
    # Convert best_adj_mask to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj_mask[i] & (1 << j):
                edges.append([i, j])
    
    return edges
