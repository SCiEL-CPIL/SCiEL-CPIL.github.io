import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 sec safety margin
    
    best_raw = float('inf')
    best_result = None
    
    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    
    # Precompute edge indices for fast lookup
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods)), len(sums), len(prods)

    def simulated_annealing(values, initial_edges, max_iter=2000, T_start=1.0, T_end=0.01):
        """
        Optimizes the set of edges to minimize max(distinct_sums, distinct_prods).
        """
        current_edges = set(initial_edges)
        current_sums, current_prods = set(), set()
        for u, v in current_edges:
            current_sums.add(values[u] + values[v])
            current_prods.add(values[u] * values[v])
        current_score = max(len(current_sums), len(current_prods))
        
        best_edges = set(current_edges)
        best_score = current_score
        best_sums = current_sums
        best_prods = current_prods
        
        # Precompute sum/prod for all edges
        edge_props = {}
        for u, v in all_edges:
            edge_props[(u, v)] = (values[u] + values[v], values[u] * values[v])
            
        # List of edges in current set for fast random choice
        current_list = list(current_edges)
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
                
            T = T_start * (T_end / T_start) ** (i / max_iter)
            
            # Pick a random edge to remove
            idx_out = random.randint(0, len(current_list) - 1)
            e_out = current_list[idx_out]
            s_out, p_out = edge_props[e_out]
            
            # Pick a random edge to add (not in current set)
            # Optimization: Use a random edge from all_edges and check if in set
            # Since m=35, N=91, probability of collision is ~38%. 
            # Better: maintain a list of available edges? 
            # For simplicity and speed, just pick random from all_edges and retry if in set.
            while True:
                e_in = random.choice(all_edges)
                if e_in not in current_edges:
                    break
            
            s_in, p_in = edge_props[e_in]
            
            # Calculate new score incrementally
            # We need to know if s_out was unique or if s_in was already present
            # This requires maintaining counts. 
            # For simplicity in this short budget, we recalculate sets if the move is promising?
            # No, that's slow. 
            # Let's maintain counts.
            
            # Actually, maintaining counts for SA is complex to implement correctly in a short time.
            # Let's use a simpler local search: Hill Climbing with random restarts.
            # Or, just re-evaluate if the move *might* be good?
            # Given the small N (14), re-evaluation of 35 edges is ~35 ops.
            # 2000 iters * 35 ops = 70k ops. Very fast.
            
            new_edges = (current_edges - {e_out}) | {e_in}
            
            # Fast re-evaluation
            new_sums = set()
            new_prods = set()
            for u, v in new_edges:
                new_sums.add(values[u] + values[v])
                new_prods.add(values[u] * values[v])
            
            new_score = max(len(new_sums), len(new_prods))
            
            delta = new_score - current_score
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                # Accept
                current_edges = new_edges
                current_score = new_score
                # Update list for e_out selection
                # Remove e_out, add e_in
                current_list[idx_out] = e_in
                
                if new_score < best_score:
                    best_score = new_score
                    best_edges = set(new_edges)
                    best_sums = new_sums
                    best_prods = new_prods
        
        return best_score, best_edges

    # Value Candidates
    value_candidates = []
    
    # 1. Arithmetic Progressions (Good for sum collisions)
    for start in range(1, 50):
        for step in [1, 2, 3]:
            vals = [start + i * step for i in range(n)]
            if max(vals) <= 10000:
                value_candidates.append(vals)
                
    # 2. Geometric Progressions (Good for product collisions)
    for base in [2, 3]:
        vals = [base ** i for i in range(n)]
        if max(vals) <= 10000:
            value_candidates.append(vals)
            
    # 3. Random sets with small range (to encourage collisions)
    random.seed(42)
    for _ in range(20):
        # Pick 14 distinct random numbers from 1..100
        vals = random.sample(range(1, 101), n)
        value_candidates.append(vals)

    # Remove duplicates
    value_candidates = [list(t) for t in {tuple(v) for v in value_candidates}]
    
    # Run SA for each candidate
    for values in value_candidates:
        if time.time() > deadline:
            break
            
        # Initial edges: Random 35 edges
        initial_edges = set(random.sample(all_edges, m))
        
        # Run SA with a few restarts
        for restart in range(3):
            if time.time() > deadline:
                break
                
            # Vary initial edges slightly for restarts
            if restart == 0:
                init = set(random.sample(all_edges, m))
            else:
                init = set(random.sample(all_edges, m))
                
            score, edges = simulated_annealing(values, init, max_iter=1000)
            
            if score < best_raw:
                best_raw = score
                # Ensure types are correct
                best_result = {
                    'values': [int(v) for v in values],
                    'edges': [[int(u), int(v)] for u, v in edges]
                }

    # Fallback if no result found (should not happen)
    if best_result is None:
        vals = list(range(1, 15))
        edges = [list(e) for e in all_edges[:35]]
        return {'values': vals, 'edges': edges}
        
    return best_result
