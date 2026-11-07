import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 14
    m = 35
    max_val = 10000
    all_edges = list(combinations(range(n), 2))
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def simulated_annealing(initial_values, initial_edges, rng, time_limit):
        """
        Runs simulated annealing to minimize the energy.
        """
        values = list(initial_values)
        edges = list(initial_edges)
        
        # Initial energy
        current_energy = evaluate(values, edges)
        best_energy = current_energy
        best_values = list(values)
        best_edges = list(edges)
        
        T = 10.0  # Initial temperature
        T_min = 0.01
        cooling_rate = 0.9995 # Adjust based on expected iterations
        
        # Pre-calculate all possible edges for quick lookup
        edge_set = set(tuple(e) for e in edges)
        
        # We will run until time limit
        while time.time() < time_limit:
            # Perform a move
            if rng.random() < 0.5:
                # Change a value
                idx = rng.randint(0, n - 1)
                old_val = values[idx]
                
                # Generate candidate values
                # Bias towards small values and neighbors
                candidates = set()
                for delta in range(-5, 6):
                    v = old_val + delta
                    if 1 <= v <= max_val:
                        candidates.add(v)
                
                # Add some random small values
                for _ in range(5):
                    candidates.add(rng.randint(1, 50))
                
                # Add some random values up to max_val
                for _ in range(2):
                    candidates.add(rng.randint(1, max_val))
                
                # Filter distinctness
                new_val = None
                for v in candidates:
                    if v == old_val:
                        continue
                    if any(v == values[i] for i in range(n) if i != idx):
                        continue
                    new_val = v
                    break
                
                if new_val is None:
                    # Fallback: pick a random distinct value
                    while True:
                        new_val = rng.randint(1, max_val)
                        if all(new_val != values[i] for i in range(n)):
                            break
                
                values[idx] = new_val
                new_energy = evaluate(values, edges)
                
                # Accept or reject
                delta_e = new_energy - current_energy
                if delta_e < 0 or rng.random() < pow(2.718, -delta_e / T):
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_values = list(values)
                        best_edges = list(edges)
                else:
                    values[idx] = old_val
            else:
                # Swap an edge
                # Pick an edge to remove
                i = rng.randint(0, m - 1)
                old_edge = edges[i]
                
                # Pick a new edge to add
                # Randomly pick from all_edges not in edge_set
                # To make this efficient, we can just pick a random edge from all_edges
                # and check if it's already there. If so, retry.
                new_edge = None
                attempts = 0
                while attempts < 20:
                    cand_edge = rng.choice(all_edges)
                    if tuple(cand_edge) not in edge_set:
                        new_edge = cand_edge
                        break
                    attempts += 1
                
                if new_edge is None:
                    # Fallback: find any missing edge
                    for e in all_edges:
                        if tuple(e) not in edge_set:
                            new_edge = e
                            break
                
                if new_edge is not None:
                    edges[i] = new_edge
                    edge_set.discard(tuple(old_edge))
                    edge_set.add(tuple(new_edge))
                    
                    new_energy = evaluate(values, edges)
                    
                    delta_e = new_energy - current_energy
                    if delta_e < 0 or rng.random() < pow(2.718, -delta_e / T):
                        current_energy = new_energy
                        if new_energy < best_energy:
                            best_energy = new_energy
                            best_values = list(values)
                            best_edges = list(edges)
                    else:
                        edges[i] = old_edge
                        edge_set.discard(tuple(new_edge))
                        edge_set.add(tuple(old_edge))
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        return best_values, best_edges, best_energy

    def local_search(values, edges, rng, time_limit):
        """
        Fast local search (steepest descent / first improvement) to polish the result.
        """
        current_energy = evaluate(values, edges)
        edge_set = set(tuple(e) for e in edges)
        improved = True
        no_improve_count = 0
        
        while improved and time.time() < time_limit:
            improved = False
            no_improve_count += 1
            if no_improve_count > 100:
                break
                
            # Try value changes
            for idx in range(n):
                old_val = values[idx]
                # Try small range
                for delta in range(-10, 11):
                    new_val = old_val + delta
                    if new_val < 1 or new_val > max_val:
                        continue
                    if any(new_val == values[i] for i in range(n) if i != idx):
                        continue
                    
                    values[idx] = new_val
                    new_energy = evaluate(values, edges)
                    if new_energy < current_energy:
                        current_energy = new_energy
                        improved = True
                        no_improve_count = 0
                        break
                    else:
                        values[idx] = old_val
                else:
                    continue
                break # Restart loop if improved
            
            # Try edge swaps
            if not improved:
                for i in range(m):
                    old_edge = edges[i]
                    # Try swapping with a random few edges
                    for _ in range(5):
                        new_edge = rng.choice(all_edges)
                        if tuple(new_edge) in edge_set:
                            continue
                        
                        edges[i] = new_edge
                        edge_set.discard(tuple(old_edge))
                        edge_set.add(tuple(new_edge))
                        
                        new_energy = evaluate(values, edges)
                        if new_energy < current_energy:
                            current_energy = new_energy
                            improved = True
                            no_improve_count = 0
                            break
                        else:
                            edges[i] = old_edge
                            edge_set.discard(tuple(new_edge))
                            edge_set.add(tuple(old_edge))
                    else:
                        continue
                    break

        return values, edges, current_energy

    rng = random.Random(42)
    
    best_overall_energy = float('inf')
    best_result = None
    
    # Starting strategies
    # 1. Small integers
    init_1 = list(range(1, 15))
    # 2. Multiples of 2
    init_2 = [2*i for i in range(1, 15)]
    # 3. Multiples of 3
    init_3 = [3*i for i in range(1, 15)]
    # 4. Highly composite-like
    init_4 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    # 5. Random small
    init_5 = list(rng.sample(range(1, 100), 14))
    
    initial_sets = [init_1, init_2, init_3, init_4, init_5]
    
    for init_values in initial_sets:
        if time.time() > deadline - 10.0:
            break
            
        # Generate initial edges: Random 35 edges
        init_edges = rng.sample(all_edges, m)
        
        # Run SA
        # Give each strategy a slice of time
        slice_end = min(deadline - 5.0, time.time() + 15.0)
        
        v, e, energy = simulated_annealing(init_values, init_edges, rng, slice_end)
        
        # Polish with local search
        polish_end = min(deadline - 1.0, time.time() + 5.0)
        v, e, energy = local_search(v, e, rng, polish_end)
        
        if energy < best_overall_energy:
            best_overall_energy = energy
            best_result = {
                'values': list(v),
                'edges': [list(edge) for edge in e]
            }
            
        # If we found a very good score, do more polishing
        if best_overall_energy <= 15:
             if time.time() < deadline - 1.0:
                 v, e, energy = local_search(v, e, rng, deadline - 0.5)
                 if energy < best_overall_energy:
                     best_overall_energy = energy
                     best_result = {
                         'values': list(v),
                         'edges': [list(edge) for edge in e]
                     }

    if best_result is None:
        # Fallback
        best_result = {
            'values': list(range(1, n+1)),
            'edges': [list(e) for e in all_edges[:m]]
        }
    
    values = [int(v) for v in best_result['values']]
    edges = [[int(u), int(v)] for u, v in best_result['edges']]
    
    # Final validation
    assert len(set(values)) == 14
    assert all(1 <= v <= 10000 for v in values)
    assert len(edges) == 35
    assert len(set(tuple(sorted(e)) for e in edges)) == 35
    assert all(u != v for u, v in edges)
    assert all(0 <= u < 14 and 0 <= v < 14 for u, v in edges)
    
    return {'values': values, 'edges': edges}
