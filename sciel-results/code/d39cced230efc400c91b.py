import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 sec safety margin
    
    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    num_edges_total = len(all_edges)
    
    # Precompute edge properties for a given set of values
    # We will maintain these dynamically or recompute if values change significantly
    # For joint optimization, we need fast evaluation.
    
    def get_initial_values():
        """Generate diverse initial value sets."""
        candidates = []
        
        # 1. Arithmetic Progressions (High sum collision)
        for start in range(1, 50):
            for step in [1, 2, 3, 5]:
                vals = [start + i * step for i in range(n)]
                if max(vals) <= 10000:
                    candidates.append(vals)
        
        # 2. Geometric Progressions (High product collision)
        for base in [2, 3]:
            vals = [base ** i for i in range(n)]
            if max(vals) <= 10000:
                candidates.append(vals)
        
        # 3. Random Small Range (High collision probability)
        random.seed(42)
        for _ in range(30):
            # Pick 14 distinct random numbers from 1..100
            vals = random.sample(range(1, 101), n)
            candidates.append(vals)
            
        # 4. "Prime-like" or Coprime sets (To test product uniqueness vs sum)
        # Actually, we want collisions, so maybe multiples?
        # Let's try sets with many divisors
        for base in [6, 12, 24]:
            vals = []
            for i in range(n):
                # Random multiples of base
                mult = random.randint(1, 10000 // base)
                v = base * mult
                if v not in vals and v <= 10000:
                    vals.append(v)
            # Fill if needed
            while len(vals) < n:
                v = random.randint(1, 10000)
                if v not in vals:
                    vals.append(v)
            candidates.append(vals)

        # Remove duplicates
        unique_candidates = [list(t) for t in {tuple(v) for v in candidates}]
        return unique_candidates

    def evaluate_state(values, edges_set):
        """
        Returns (score, sum_counts, prod_counts, sum_set, prod_set)
        """
        sum_counts = {}
        prod_counts = {}
        for u, v in edges_set:
            s = values[u] + values[v]
            p = values[u] * values[v]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        
        distinct_sums = len(sum_counts)
        distinct_prods = len(prod_counts)
        score = max(distinct_sums, distinct_prods)
        
        return score, sum_counts, prod_counts

    def joint_simulated_annealing(values, initial_edges, max_iter=5000, T_start=1.0, T_end=0.001):
        """
        Jointly optimizes values and edges.
        Moves:
        1. Swap an edge (remove one, add one)
        2. Swap a value (replace one value with a new random valid value)
        3. Swap two values (if distinctness is maintained)
        """
        current_values = list(values)
        current_edges = set(initial_edges)
        
        # Initial evaluation
        current_score, sum_counts, prod_counts = evaluate_state(current_values, current_edges)
        
        best_score = current_score
        best_values = list(current_values)
        best_edges = set(current_edges)
        
        # Precompute all edges for fast access
        # edge_props is not static because values change. 
        # We must calculate s, p on the fly for the specific edge involved in the move.
        
        # To make value moves fast, we need to know which edges are incident to a vertex.
        # Let's maintain an adjacency list or just calculate on the fly since N=14 is small.
        # For an edge move (u,v) removed and (u',v') added:
        # We need to update sum_counts and prod_counts.
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
                
            T = T_start * (T_end / T_start) ** (i / max_iter)
            
            move_type = random.random()
            
            if move_type < 0.6:
                # Edge Move: Remove random edge, Add random non-edge
                if len(current_edges) == 0:
                    continue
                    
                # Remove
                e_out = random.choice(list(current_edges))
                u_out, v_out = e_out
                s_out = current_values[u_out] + current_values[v_out]
                p_out = current_values[u_out] * current_values[v_out]
                
                # Add
                while True:
                    e_in = random.choice(all_edges)
                    if e_in not in current_edges:
                        break
                u_in, v_in = e_in
                s_in = current_values[u_in] + current_values[v_in]
                p_in = current_values[u_in] * current_values[v_in]
                
                # Apply move tentatively
                current_edges.discard(e_out)
                current_edges.add(e_in)
                
                # Update counts
                # Decrement out
                sum_counts[s_out] -= 1
                if sum_counts[s_out] == 0:
                    del sum_counts[s_out]
                prod_counts[p_out] -= 1
                if prod_counts[p_out] == 0:
                    del prod_counts[p_out]
                
                # Increment in
                sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
                prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
                
                new_score = max(len(sum_counts), len(prod_counts))
                delta = new_score - current_score
                
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_values = list(current_values)
                        best_edges = set(current_edges)
                else:
                    # Revert
                    current_edges.discard(e_in)
                    current_edges.add(e_out)
                    
                    sum_counts[s_in] -= 1
                    if sum_counts[s_in] == 0:
                        del sum_counts[s_in]
                    prod_counts[p_in] -= 1
                    if prod_counts[p_in] == 0:
                        del prod_counts[p_in]
                    
                    sum_counts[s_out] = sum_counts.get(s_out, 0) + 1
                    prod_counts[p_out] = prod_counts.get(p_out, 0) + 1
                    
            else:
                # Value Move: Change one value to a new random distinct value
                idx = random.randint(0, n - 1)
                old_val = current_values[idx]
                
                # Find a new value
                # To maintain distinctness, we can pick from a pool or retry
                new_val = None
                for _ in range(10):
                    candidate = random.randint(1, 10000)
                    if candidate != old_val and candidate not in current_values:
                        new_val = candidate
                        break
                
                if new_val is None:
                    continue
                
                # Identify all edges incident to idx
                incident_edges = []
                for u, v in current_edges:
                    if u == idx or v == idx:
                        incident_edges.append((u, v))
                
                # Save old properties of incident edges
                old_props = []
                for u, v in incident_edges:
                    s = current_values[u] + current_values[v]
                    p = current_values[u] * current_values[v]
                    old_props.append((s, p))
                    
                # Apply value change
                current_values[idx] = new_val
                
                # Update counts for incident edges
                for (s, p) in old_props:
                    sum_counts[s] -= 1
                    if sum_counts[s] == 0:
                        del sum_counts[s]
                    prod_counts[p] -= 1
                    if prod_counts[p] == 0:
                        del prod_counts[p]
                
                new_props = []
                for u, v in incident_edges:
                    s = current_values[u] + current_values[v]
                    p = current_values[u] * current_values[v]
                    new_props.append((s, p))
                    sum_counts[s] = sum_counts.get(s, 0) + 1
                    prod_counts[p] = prod_counts.get(p, 0) + 1
                    
                new_score = max(len(sum_counts), len(prod_counts))
                delta = new_score - current_score
                
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_values = list(current_values)
                        best_edges = set(current_edges)
                else:
                    # Revert value
                    current_values[idx] = old_val
                    
                    # Revert counts
                    for (s, p) in new_props:
                        sum_counts[s] -= 1
                        if sum_counts[s] == 0:
                            del sum_counts[s]
                        prod_counts[p] -= 1
                        if prod_counts[p] == 0:
                            del prod_counts[p]
                            
                    for (s, p) in old_props:
                        sum_counts[s] = sum_counts.get(s, 0) + 1
                        prod_counts[p] = prod_counts.get(p, 0) + 1

        return best_score, best_values, best_edges

    value_candidates = get_initial_values()
    
    best_raw = float('inf')
    best_result = None
    
    # Shuffle candidates to ensure diversity in early time
    random.shuffle(value_candidates)
    
    for values in value_candidates:
        if time.time() > deadline:
            break
            
        # Run joint SA with a few restarts
        # We use fewer restarts but longer runs because joint SA is more powerful per step
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initial edges: Random 35 edges
            init_edges = set(random.sample(all_edges, m))
            
            # Run SA
            # Max iter adjusted based on remaining time? No, fixed high number, break on deadline.
            score, vals, edges = joint_simulated_annealing(values, init_edges, max_iter=2000)
            
            if score < best_raw:
                best_raw = score
                # Ensure types are correct
                best_result = {
                    'values': [int(v) for v in vals],
                    'edges': [[int(u), int(v)] for u, v in edges]
                }
                
                # If we find a very good score (e.g., < 30), we might want to refine further
                # But the deadline guard handles the stop condition.

    # Fallback if no result found
    if best_result is None:
        vals = list(range(1, 15))
        edges = [list(e) for e in all_edges[:35]]
        return {'values': vals, 'edges': edges}
        
    return best_result

if __name__ == "__main__":
    res = construct()
    # Simple validation check
    vals = res['values']
    edges = res['edges']
    assert len(vals) == 14
    assert len(set(vals)) == 14
    assert all(isinstance(v, int) and v > 0 and v <= 10000 for v in vals)
    assert len(edges) == 35
    edge_set = set()
    for e in edges:
        u, v = e
        assert isinstance(u, int) and isinstance(v, int)
        assert u != v
        assert 0 <= u < 14 and 0 <= v < 14
        key = (min(u,v), max(u,v))
        assert key not in edge_set
        edge_set.add(key)
    
    sums = set()
    prods = set()
    for u, v in edges:
        sums.add(vals[u] + vals[v])
        prods.add(vals[u] * vals[v])
    raw = max(len(sums), len(prods))
    print(f"Raw Score: {raw}")
