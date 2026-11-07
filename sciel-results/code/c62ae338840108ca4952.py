"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

def construct():
    n = 14
    m = 35
    start_time = time.time()
    time_limit = 55  # Leave 5s buffer for evaluation overhead
    
    # Initial random state
    # Values should be distinct positive integers <= 10000
    # Edges are 35 unique pairs from 14 vertices (max 91 edges)
    
    def get_score(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def random_state():
        vals = random.sample(range(1, 10001), n)
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        eds = all_edges[:m]
        return vals, eds
    
    # Initialize
    current_vals, current_edges = random_state()
    current_score = get_score(current_vals, current_edges)
    
    best_vals, best_edges, best_score = current_vals[:], current_edges[:], current_score
    
    # Local Search
    while time.time() - start_time < time_limit:
        # Decide whether to perturb values or edges
        if random.random() < 0.5:
            # Perturb values
            new_vals = current_vals[:]
            idx = random.randint(0, n-1)
            new_val = random.randint(1, 10000)
            # Ensure distinctness
            if new_val not in new_vals:
                new_vals[idx] = new_val
            else:
                continue
            
            new_score = get_score(new_vals, current_edges)
            if new_score < current_score:
                current_vals = new_vals
                current_score = new_score
                if current_score < best_score:
                    best_vals = current_vals[:]
                    best_edges = current_edges[:]
                    best_score = current_score
        else:
            # Perturb edges: swap one edge with a random non-edge
            new_edges = current_edges[:]
            # Pick a random edge to remove
            rem_idx = random.randint(0, m-1)
            rem_edge = new_edges[rem_idx]
            
            # Generate a random non-edge
            all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
            # Filter out existing edges
            existing_set = set(tuple(e) for e in new_edges)
            non_edges = [e for e in all_possible if tuple(e) not in existing_set]
            if not non_edges:
                continue
            add_edge = random.choice(non_edges)
            
            new_edges[rem_idx] = add_edge
            new_score = get_score(current_vals, new_edges)
            
            if new_score < current_score:
                current_edges = new_edges
                current_score = new_score
                if current_score < best_score:
                    best_vals = current_vals[:]
                    best_edges = current_edges[:]
                    best_score = current_score
            # Accept equal score sometimes to escape plateaus? 
            # Or just strict improvement. Let's add a small probability of accepting equal to explore.
            elif new_score == current_score and random.random() < 0.1:
                current_edges = new_edges
    
    return {'values': best_vals, 'edges': best_edges}
# EVOLVE-BLOCK-END
