import time
import random

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit (evaluator kills at 150s)

    # Build initial Clebsch graph using bitsets for adjacency
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            xor = u ^ v
            dist = bin(xor).count('1')
            if dist == 1 or dist == 4:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

    def count_edges():
        total = 0
        for u in range(n):
            total += bin(adj[u]).count('1')
        return total // 2

    def is_triangle_free():
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    if adj[u] & adj[v]:
                        return False
        return True

    def is_3_colorable(adj_state):
        # Fast backtracking 3-coloring with bitsets and vertex ordering
        colors = [-1] * n
        # Order by degree descending for faster pruning
        degrees = [bin(adj_state[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: degrees[v], reverse=True)
        
        # Precompute neighbor masks for each vertex in the ordering
        # For speed, we check conflicts via bitsets
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Try each color
            for c in range(3):
                # Check if any colored neighbor has color c
                conflict = False
                neighbors = adj_state[v]
                # Iterate through set bits in neighbors
                nb = neighbors
                while nb:
                    lsb = nb & (-nb)
                    u = lsb.bit_length() - 1
                    if colors[u] == c:
                        conflict = True
                        break
                    nb ^= lsb
                if not conflict:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def can_add_edge(u, v):
        """Check if adding edge (u,v) keeps graph triangle-free."""
        if (adj[u] >> v) & 1:
            return False  # already present
        # Triangle-free check: u and v must have no common neighbor
        if adj[u] & adj[v]:
            return False
        return True

    # Phase 1: Greedy edge addition
    # Try to add every possible missing edge that maintains triangle-freeness and 4-chromaticity
    for u in range(n):
        for v in range(u + 1, n):
            if time.time() > deadline:
                break
            if (adj[u] >> v) & 1:
                continue
            if not can_add_edge(u, v):
                continue
            add_edge(u, v)
            if is_3_colorable(adj):
                remove_edge(u, v)
            # else keep it
        if time.time() > deadline:
            break

    best_edges = count_edges()
    best_adj = adj[:]

    # Phase 2: Local search with simulated annealing-like restarts
    # Structural change: use multi-restart with random perturbations to escape local optima
    # This is inspired by the multi-restart strategy that improved score from 2.3 to 2.35
    
    for restart in range(50):  # up to 50 restarts
        if time.time() > deadline - 10:
            break
        
        # Random perturbation: remove k random edges, then try to re-add
        k = random.randint(1, 3)
        current_edges = count_edges()
        
        # Store current state
        saved_adj = adj[:]
        
        # Remove k random edges
        for _ in range(k):
            # Find a random edge
            all_present = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        all_present.append((u, v))
            if not all_present:
                break
            u, v = random.choice(all_present)
            remove_edge(u, v)
        
        # Try to re-add edges greedily
        improved = False
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline - 10:
                    break
                if (adj[u] >> v) & 1:
                    continue
                if not can_add_edge(u, v):
                    continue
                add_edge(u, v)
                if is_3_colorable(adj):
                    remove_edge(u, v)
                else:
                    improved = True
            if time.time() > deadline - 10:
                break
        
        if improved:
            new_edges = count_edges()
            if new_edges > best_edges:
                best_edges = new_edges
                best_adj = adj[:]
        else:
            # Restore saved state if no improvement
            adj = saved_adj[:]

    # Phase 3: Edge swap local search on the best found
    adj = best_adj[:]
    
    while time.time() < deadline - 5:
        improved = False
        # Try removing each edge and adding a different one
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline - 5:
                    break
                if not ((adj[u] >> v) & 1):
                    continue
                # Try removing this edge
                remove_edge(u, v)
                # Try adding each missing edge
                for a in range(n):
                    for b in range(a + 1, n):
                        if time.time() > deadline - 5:
                            break
                        if (adj[a] >> b) & 1:
                            continue
                        if not can_add_edge(a, b):
                            continue
                        add_edge(a, b)
                        if not is_3_colorable(adj):
                            # Good swap
                            if count_edges() > best_edges:
                                best_edges = count_edges()
                                best_adj = adj[:]
                                improved = True
                        else:
                            remove_edge(a, b)
                # Restore removed edge
                add_edge(u, v)
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    # Use best found
    if best_edges > count_edges():
        adj = best_adj

    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
