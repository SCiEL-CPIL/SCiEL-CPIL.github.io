"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _raw_score(values, edges):
    """Compute the raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        su = values[u]
        sv = values[v]
        sums.add(su + sv)
        prods.add(su * sv)
    return max(len(sums), len(prods))

def _evaluate_state(values, edges):
    """
    Efficiently evaluate the state.
    Returns (score, sums_set, prods_set) to allow incremental updates if needed,
    though for SA we often recompute or do delta checks.
    For simplicity and robustness in the mutation step, we return the components.
    """
    sums = set()
    prods = set()
    for u, v in edges:
        su = values[u]
        sv = values[v]
        sums.add(su + sv)
        prods.add(su * sv)
    return max(len(sums), len(prods)), sums, prods

def _initial_solution():
    """
    Generate a random initial solution.
    Values: random distinct positive integers <= 10000.
    Edges: random 35 edges.
    """
    values = random.sample(range(1, 10001), 14)
    all_edges = list(combinations(range(14), 2))
    random.shuffle(all_edges)
    edges = [list(e) for e in all_edges[:35]]
    return values, edges

def _mutate(values, edges):
    """
    Perform a random mutation on the state.
    1. Swap two values or change one value.
    2. Swap an edge in with an edge out.
    """
    new_values = values[:]
    new_edges = [e[:] for e in edges]
    
    # 1. Mutate values (30% chance per call, or just always do one value mutation)
    if random.random() < 0.5:
        # Change one value
        idx = random.randint(0, 13)
        # Ensure it stays distinct and within bounds
        while True:
            new_val = random.randint(1, 10000)
            if new_val not in new_values:
                new_values[idx] = new_val
                break
    
    # 2. Mutate edges (always do one edge swap)
    # Pick an existing edge to remove
    edge_idx = random.randint(0, 34)
    removed_edge = tuple(sorted(new_edges[edge_idx]))
    
    # Pick a new edge to add that is not currently in the graph
    all_edges = list(combinations(range(14), 2))
    current_edge_set = set(tuple(sorted(e)) for e in new_edges)
    
    # Candidate edges are those not in current_edge_set
    candidates = [e for e in all_edges if e not in current_edge_set]
    if candidates:
        new_edge = random.choice(candidates)
        new_edges[edge_idx] = list(new_edge)
    else:
        # If we have all edges (impossible for 14 nodes, max 91), just keep
        pass
        
    return new_values, new_edges

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best_score = float('inf')
    best_result = None
    
    # Initial solution
    values, edges = _initial_solution()
    current_score, _, _ = _evaluate_state(values, edges)
    
    if current_score < best_score:
        best_score = current_score
        best_result = {
            'values': values,
            'edges': edges
        }
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    while time.time() < deadline and T > T_min:
        new_values, new_edges = _mutate(values, edges)
        
        # Validate new state (distinct values, valid edges)
        if len(set(new_values)) != 14:
            continue
        if any(v < 1 or v > 10000 for v in new_values):
            continue
            
        edge_tuples = set(tuple(sorted(e)) for e in new_edges)
        if len(edge_tuples) != 35:
            continue
        if any(u == v for u, v in new_edges):
            continue
            
        new_score, _, _ = _evaluate_state(new_values, new_edges)
        
        delta = new_score - current_score
        
        if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
            # Accept
            values = new_values
            edges = new_edges
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_result = {
                    'values': values,
                    'edges': edges
                }
                
                # Early exit if we hit a very low score
                if best_score <= 10:
                    break
        
        T *= cooling_rate
        
    # If we didn't find anything better than initial, return initial
    if best_result is None:
        # Fallback to a deterministic seed-like structure if SA failed completely
        values = list(range(1, 15))
        all_edges = list(combinations(range(14), 2))
        edges = [list(e) for e in all_edges[:35]]
        best_result = {
            'values': values,
            'edges': edges
        }
        
    # Ensure types are correct for return
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return best_result
# EVOLVE-BLOCK-END
