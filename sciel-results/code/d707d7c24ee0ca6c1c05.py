import time
import random
import numpy as np

def construct():
    d = 7
    n_vertices = 1 << d  # 128
    n_edges_total = n_vertices * d // 2  # 448
    
    start_time = time.time()
    time_limit = 118.0  # safety margin before 120s (evaluator kills at 150s)
    
    # Precompute all edges as (u, i) where u < u^(1<<i)
    edges = []
    edge_index = {}  # (u, i) -> index in edges list, where u < u^(1<<i)
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, i))
                edge_index[(u, i)] = idx
                idx += 1
    
    n_edges = len(edges)  # 448
    
    # State: edge_present[u][i] = True if edge from u in direction i is present
    # Note: edge_present[u][i] == edge_present[u^(1<<i)][i]
    edge_present = np.zeros((n_vertices, d), dtype=bool)
    
    # Precompute for fast C4 checking:
    # For edge (u, i), a C4 is formed if for some j != i:
    #   edge_present[u][j] AND edge_present[u^(1<<i)][j] AND edge_present[u^(1<<j)][i]
    # We can precompute the list of (u, i) pairs to check.
    
    def can_add(u, i):
        """Check if adding edge from u in direction i creates a C4."""
        v = u ^ (1 << i)
        for j in range(d):
            if j == i:
                continue
            if edge_present[u][j] and edge_present[v][j] and edge_present[u ^ (1 << j)][i]:
                return False
        return True
    
    def add_edge(u, i):
        v = u ^ (1 << i)
        edge_present[u][i] = True
        edge_present[v][i] = True
    
    def remove_edge(u, i):
        v = u ^ (1 << i)
        edge_present[u][i] = False
        edge_present[v][i] = False
    
    def count_edges():
        return int(np.sum(edge_present)) // 2
    
    def greedy_fill():
        """Greedily add all possible edges in a fixed order."""
        for u, i in edges:
            if not edge_present[u][i]:
                if can_add(u, i):
                    add_edge(u, i)
    
    def greedy_random():
        """Greedy with random edge ordering."""
        edge_present[:] = False
        order = list(range(n_edges))
        random.shuffle(order)
        for idx in order:
            u, i = edges[idx]
            if can_add(u, i):
                add_edge(u, i)
        return count_edges()
    
    # Phase 1: Multiple random greedy starts
    best_count = 0
    best_state = None
    
    n_greedy_starts = 20
    for _ in range(n_greedy_starts):
        if time.time() - start_time > time_limit - 30:
            break
        c = greedy_random()
        if c > best_count:
            best_count = c
            best_state = edge_present.copy()
    
    # Restore best
    if best_state is not None:
        edge_present[:] = best_state
    
    # Phase 2: Local search with simulated annealing
    # Use a more sophisticated approach: 
    # - Try flipping individual edges (add or remove)
    # - Try 2-edge moves (remove one, add another)
    # - Track a "conflict count" to guide search
    
    current_count = best_count
    current_state = edge_present.copy()
    edge_present[:] = current_state
    
    best_overall = current_count
    best_overall_state = current_state.copy()
    
    # Temperature schedule for simulated annealing
    t0 = 5.0
    t_min = 0.001
    cooling_rate = 0.9998
    
    temperature = t0
    
    # Precompute for each edge its "conflict partners" for faster checking
    # Actually, let's use a different approach: maintain a count of how many C4s
    # each edge is part of, to guide the search.
    
    # For efficiency, let's try a different local search:
    # 1. Pick a random edge to remove
    # 2. Then greedily refill all possible edges
    # 3. Accept if improvement
    
    iteration = 0
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() - start_time < time_limit - 2:
        iteration += 1
        
        if time.time() - start_time > time_limit - 1:
            break
        
        # Strategy: remove 1-3 random edges, then greedy refill
        num_remove = random.choice([1, 1, 2, 2, 3])
        removed_edges = []
        attempts = 0
        while len(removed_edges) < num_remove and attempts < 30:
            attempts += 1
            u = random.randint(0, n_vertices - 1)
            i = random.randint(0, d - 1)
            if edge_present[u][i]:
                v = u ^ (1 << i)
                if u < v:
                    remove_edge(u, i)
                    removed_edges.append((u, i))
        
        if not removed_edges:
            continue
        
        # Greedy refill
        greedy_fill()
        
        new_count = count_edges()
        delta = new_count - current_count
        
        # Accept if better, or with probability exp(delta/T) if worse
        if delta >= 0:
            current_count = new_count
            if current_count > best_overall:
                best_overall = current_count
                best_overall_state = edge_present.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Accept with probability
            if temperature > t_min:
                prob = np.exp(delta / temperature)
                if random.random() < prob:
                    current_count = new_count
                    no_improve += 1
                else:
                    # Revert: restore from a saved state
                    # We need to save state before modification
                    # For simplicity, just skip
                    pass
            else:
                # Revert
                pass
        
        # Cool down
        temperature = max(t_min, temperature * cooling_rate)
        
        if best_overall >= 308:
            break
        
        if no_improve > max_no_improve and temperature <= t_min:
            # Restart with a new random greedy
            c = greedy_random()
            if c > current_count:
                current_count = c
            if c > best_overall:
                best_overall = c
                best_overall_state = edge_present.copy()
            no_improve = 0
            temperature = t0 * 0.5
    
    # Restore best state
    edge_present[:] = best_overall_state
    
    # Convert to output format
    out = []
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v and edge_present[u][i]:
                out.append([u, v])
    
    return out
