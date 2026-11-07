import time
import random
import math

def construct():
    """
    Constructs a graph with 14 vertices and 35 edges to minimize the max of 
    distinct edge sums and distinct edge products.
    """
    N = 14
    M = 35
    MAX_VAL = 10000
    DEADLINE = time.time() + 55.0  # 55s limit to be safe against 60s eval + 90s kill
    
    # Helper to calculate cost
    def get_cost(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    # Helper to generate random edges
    def random_edges(n, m):
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        return random.sample(all_edges, m)
    
    # Helper to generate initial values
    def initial_values():
        # Start with small integers to encourage collisions
        return list(range(1, N + 1))
        
    best_values = initial_values()
    best_edges = random_edges(N, M)
    best_cost = get_cost(best_values, best_edges)
    
    # Simulated Annealing
    current_values = list(best_values)
    current_edges = list(best_edges)
    current_cost = best_cost
    
    T = 10.0
    T_min = 0.01
    alpha = 0.999  # Cooling rate
    
    start_time = time.time()
    
    while time.time() < DEADLINE:
        # Generate a neighbor
        move_type = random.random()
        new_values = list(current_values)
        new_edges = list(current_edges)
        
        if move_type < 0.5:
            # Change a single value
            idx = random.randint(0, N - 1)
            # Try to find a new value that is distinct
            # Random value in a reasonable range
            # Small values are often better for collisions
            new_val = random.randint(1, 200) 
            # Ensure distinctness
            if new_val in new_values:
                # Retry a few times or pick a random distinct one
                while new_val in new_values:
                    new_val = random.randint(1, MAX_VAL)
            new_values[idx] = new_val
        elif move_type < 0.75:
            # Swap two values
            i, j = random.sample(range(N), 2)
            new_values[i], new_values[j] = new_values[j], new_values[i]
        else:
            # Change an edge
            # Remove a random edge
            edge_idx = random.randint(0, M - 1)
            removed_edge = tuple(current_edges[edge_idx])
            new_edges.pop(edge_idx)
            
            # Add a random non-existent edge
            all_possible = [(i, j) for i in range(N) for j in range(i+1, N)]
            existing = set(tuple(e) for e in new_edges)
            # Filter out existing
            candidates = [e for e in all_possible if tuple(e) not in existing]
            if candidates:
                new_edge = random.choice(candidates)
                new_edges.append(list(new_edge))
            else:
                # If no candidates (shouldn't happen with M < Max), restore
                new_edges.insert(edge_idx, list(removed_edge))
        
        new_cost = get_cost(new_values, new_edges)
        
        # Acceptance criterion
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_values = new_values
            current_edges = new_edges
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_values = list(current_values)
                best_edges = list(current_edges)
                best_cost = current_cost
                
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            # Optional: Restart with a new random start if stuck
            # But let's just keep going with low T for fine-tuning
            
    # Final verification and type conversion
    # Ensure values are distinct ints
    # Ensure edges are valid
    # The logic above ensures distinctness for value changes, but swaps preserve distinctness.
    # Edges are generated from valid samples.
    
    return {
        'values': [int(v) for v in best_values],
        'edges': [[int(u), int(v)] for u, v in best_edges]
    }
