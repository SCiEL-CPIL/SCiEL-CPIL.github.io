import time
import random
import numpy as np

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    N = 40
    adj = np.zeros((N, N), dtype=np.int8)

    def edge_count():
        return int(np.sum(adj)) // 2

    def add_edge(u, v):
        adj[u, v] = 1
        adj[v, u] = 1

    def remove_edge(u, v):
        adj[u, v] = 0
        adj[v, u] = 0

    def has_c6_fast(u, v):
        """Check if there's a simple path of length 5 from u to v using BFS with bitmasks."""
        if u == v or adj[u, v]:
            return False
        
        # BFS: frontier[d] = bitmask of vertices at distance d from u (simple path)
        # We need to track visited to ensure simplicity
        visited_mask = 1 << u
        frontier = [0] * 5
        frontier[0] = 1 << u
        
        for d in range(4):
            next_mask = 0
            m = frontier[d]
            while m:
                lsb = m & (-m)
                x = lsb.bit_length() - 1
                m ^= lsb
                # Get neighbors of x
                nbrs = 0
                for nb in range(N):
                    if adj[x, nb] and not (visited_mask >> nb) & 1 and nb != v:
                        nbrs |= (1 << nb)
                next_mask |= nbrs
            visited_mask |= next_mask
            frontier[d+1] = next_mask
        
        # Check if any vertex in frontier[4] is adjacent to v
        m = frontier[4]
        while m:
            lsb = m & (-m)
            x = lsb.bit_length() - 1
            m ^= lsb
            if adj[x, v]:
                return True
        return False

    def can_add(u, v):
        if adj[u, v]:
            return False
        return not has_c6_fast(u, v)

    def get_edges():
        es = []
        for u in range(N):
            for v in range(u+1, N):
                if adj[u, v]:
                    es.append([u, v])
        return es

    best_count = 0
    best_edges = []

    def update_best():
        nonlocal best_count, best_edges
        c = edge_count()
        if c > best_count:
            best_count = c
            best_edges = get_edges()

    def initialize_random():
        """Initialize with a random sparse graph."""
        adj[:] = 0
        num_initial = random.randint(30, 60)
        all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
        random.shuffle(all_pairs)
        added = 0
        for u, v in all_pairs:
            if added >= num_initial:
                break
            if can_add(u, v):
                add_edge(u, v)
                added += 1

    def greedy_add():
        """Try to add as many edges as possible greedily."""
        improved = True
        while improved:
            improved = False
            for u in range(N):
                for v in range(u+1, N):
                    if not adj[u, v]:
                        if can_add(u, v):
                            add_edge(u, v)
                            improved = True
        update_best()

    def simulated_annealing(initial_temp, final_temp, steps):
        """Run simulated annealing."""
        nonlocal best_count, best_edges
        
        current_edges = edge_count()
        temp = initial_temp
        if steps <= 0:
            return
        cooling = (final_temp / initial_temp) ** (1.0 / steps)
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Random move: add or remove a random edge
            u = random.randint(0, N-1)
            v = random.randint(0, N-1)
            if u == v:
                continue
            
            if adj[u, v]:
                # Remove edge
                remove_edge(u, v)
                new_count = edge_count()
                delta = new_count - current_edges
                if delta >= 0 or random.random() < np.exp(delta / temp):
                    current_edges = new_count
                    update_best()
                else:
                    add_edge(u, v)
            else:
                # Try to add edge
                if can_add(u, v):
                    add_edge(u, v)
                    new_count = edge_count()
                    delta = new_count - current_edges
                    if delta >= 0 or random.random() < np.exp(delta / temp):
                        current_edges = new_count
                        update_best()
                    else:
                        remove_edge(u, v)
            
            temp *= cooling

    # Multiple restarts
    while time.time() < deadline:
        initialize_random()
        greedy_add()
        
        if time.time() > deadline:
            break
        
        # Run simulated annealing
        remaining = deadline - time.time()
        steps = min(3000, int(remaining * 30))
        if steps > 100:
            simulated_annealing(10.0, 0.1, steps)
        
        # Try greedy add again after SA
        greedy_add()
    
    # Final greedy pass
    greedy_add()
    
    return best_edges
