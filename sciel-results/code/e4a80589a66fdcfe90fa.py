import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin for 90s limit
    
    n = 14
    m = 35
    
    def score(vals, edges):
        """Calculate the objective function: max(distinct_sums, distinct_products)."""
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(vals[i] + vals[j])
            prods.add(vals[i] * vals[j])
        return max(len(sums), len(prods))

    def greedy_and_local_search_edges(vals, time_limit):
        """
        Select m edges to minimize score.
        Uses greedy initialization followed by local search (swaps).
        """
        all_edges = list(itertools.combinations(range(n), 2))
        # Pre-compute sums and products for all potential edges
        edge_data = {}
        for i, j in all_edges:
            edge_data[(i, j)] = (vals[i] + vals[j], vals[i] * vals[j])
        
        # Greedy Initialization
        selected_edges = []
        selected_set = set()
        current_sums = set()
        current_prods = set()
        
        # Sort edges initially? No, random order with scoring is better for diversity
        random.shuffle(all_edges)
        
        for edge in all_edges:
            if len(selected_edges) >= m:
                break
            if time.time() > time_limit:
                break
            i, j = edge
            s, p = edge_data[edge]
            
            # Heuristic: prefer edges that add fewer new distinct sums/products
            new_sums_count = 0 if s in current_sums else 1
            new_prods_count = 0 if p in current_prods else 1
            
            # We want to minimize the potential increase in max(sums, prods)
            # A simple proxy: minimize (new_sums_count + new_prods_count)
            # If tie, maybe prefer smaller sum/product?
            # For now, just pick the first one that fits, but we can do better.
            # Actually, a proper greedy would evaluate all remaining edges and pick the best one.
            # With 91 edges, evaluating all remaining at each step (35 steps) is 91*35 ~ 3000 ops. Very fast.
            pass

        # Let's do a proper greedy: at each step, pick the edge that minimizes the score increase
        selected_edges = []
        selected_set = set()
        current_sums = set()
        current_prods = set()
        
        available_edges = list(all_edges)
        random.shuffle(available_edges)
        
        for _ in range(m):
            if time.time() > time_limit:
                break
            best_edge = None
            best_score_increase = float('inf')
            best_s, best_p = None, None
            
            for edge in available_edges:
                if edge in selected_set:
                    continue
                i, j = edge
                s, p = edge_data[edge]
                
                # Calculate potential new score
                new_sums_size = len(current_sums) + (0 if s in current_sums else 1)
                new_prods_size = len(current_prods) + (0 if p in current_prods else 1)
                
                # We want to minimize max(new_sums_size, new_prods_size)
                # But since we don't know future edges, we minimize the immediate penalty.
                # Penalty = max(new_sums_size, new_prods_size) - max(len(current_sums), len(current_prods))
                # Actually, just minimizing the number of new distinct items is a good proxy.
                # Let's minimize (new_sums_size + new_prods_size) primarily, 
                # and break ties by favoring edges that keep max low.
                
                penalty = (0 if s in current_sums else 1) + (0 if p in current_prods else 1)
                
                # Tie breaker: prefer edges that result in lower max
                current_max = max(len(current_sums), len(current_prods))
                new_max = max(new_sums_size, new_prods_size)
                
                # Composite score: Primary is penalty, secondary is new_max
                if penalty < best_score_increase or (penalty == best_score_increase and new_max < best_score_increase): 
                     # Note: best_score_increase is being used for penalty, let's use separate vars
                     pass
                
                # Let's refine the comparison
                if best_edge is None:
                    best_edge = edge
                    best_score_increase = penalty
                    best_new_max = new_max
                    best_s, best_p = s, p
                else:
                    if penalty < best_score_increase:
                        best_edge = edge
                        best_score_increase = penalty
                        best_new_max = new_max
                        best_s, best_p = s, p
                    elif penalty == best_score_increase:
                        if new_max < best_new_max:
                            best_edge = edge
                            best_score_increase = penalty
                            best_new_max = new_max
                            best_s, best_p = s, p
                        # Optional: random tie break
                        elif random.random() < 0.1:
                            best_edge = edge
                            best_score_increase = penalty
                            best_new_max = new_max
                            best_s, best_p = s, p

            if best_edge:
                selected_edges.append(best_edge)
                selected_set.add(best_edge)
                current_sums.add(best_s)
                current_prods.add(best_p)
                available_edges.remove(best_edge)

        # Local Search: Swaps
        current_edges = list(selected_edges)
        current_edge_set = set(selected_edges)
        current_score = score(vals, current_edges)
        
        # Pre-compute score components for faster updates?
        # For 35 edges, full re-calc is fast enough.
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try swapping each edge with a random non-edge
            # Or iterate through all possible swaps? 35 * (91-35) ~ 2000 swaps. Fast.
            for idx in range(len(current_edges)):
                if time.time() > time_limit:
                    break
                edge_to_remove = current_edges[idx]
                for edge_to_add in all_edges:
                    if edge_to_add in current_edge_set:
                        continue
                    
                    # Simulate swap
                    new_edges = current_edges[:idx] + [edge_to_add] + current_edges[idx+1:]
                    # Check validity (no duplicates, which is guaranteed by set check)
                    
                    new_score = score(vals, new_edges)
                    if new_score < current_score:
                        current_edges = new_edges
                        current_edge_set.remove(edge_to_remove)
                        current_edge_set.add(edge_to_add)
                        current_score = new_score
                        improved = True
                        # Restart inner loop or continue? 
                        # Continuing with the updated state is fine, but we need to be careful with indices.
                        # Breaking out to restart the outer loop is safer and often faster convergence.
                        break
                if improved:
                    break
        
        return current_edges, current_score

    def local_search_values(vals, edges, time_limit):
        """
        Try to improve values via small perturbations.
        """
        vals = list(vals)
        current_score = score(vals, edges)
        
        # Only perturb if we have time
        if time.time() > time_limit - 2:
            return vals, current_score
            
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for idx in range(n):
                if time.time() > time_limit:
                    break
                # Try small changes
                # Focus on changes that might create collisions
                # E.g., if v[idx] is large, try reducing it to match another sum/product
                for delta in range(-3, 4):
                    if delta == 0:
                        continue
                    new_val = vals[idx] + delta
                    if new_val <= 0 or new_val > 10000:
                        continue
                    if new_val in vals and vals.index(new_val) != idx:
                        continue
                    
                    new_vals = vals[:idx] + [new_val] + vals[idx+1:]
                    new_score = score(new_vals, edges)
                    if new_score < current_score:
                        vals = new_vals
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
        return vals, current_score

    def generate_candidate_values():
        """
        Generate diverse candidate value sets.
        """
        candidates = []
        
        # 1. Small integers with high divisibility
        # Multiples of 1, 2, 3...
        for base in range(1, 10):
            vals = [base * (i + 1) for i in range(n)]
            if max(vals) <= 10000 and len(set(vals)) == n:
                candidates.append(vals)
                
        # 2. Highly composite numbers (many factors)
        # 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 30, 36, 48
        hc = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 30, 36, 48]
        if len(set(hc)) == n and max(hc) <= 10000:
            candidates.append(hc)
            
        # 3. Scaled highly composite
        for scale in [2, 3, 5]:
            vals = [v * scale for v in hc]
            if max(vals) <= 10000 and len(set(vals)) == n:
                candidates.append(vals)
                
        # 4. Random sets with small range
        for _ in range(20):
            max_val = random.randint(20, 100)
            vals = random.sample(range(1, max_val + 1), n)
            candidates.append(vals)
            
        # 5. Random sets with shared factor
        for _ in range(20):
            base = random.randint(2, 10)
            vals = []
            while len(vals) < n:
                v = base * random.randint(1, 1000)
                if v <= 10000 and v not in vals:
                    vals.append(v)
            candidates.append(vals)
            
        return candidates

    best_score = float('inf')
    best_result = None
    
    candidates = generate_candidate_values()
    random.shuffle(candidates)
    
    # Add some random restarts to the candidates list if time permits
    # But first, process the structured ones
    
    for vals in candidates:
        if time.time() > deadline:
            break
            
        # Validate
        if not all(isinstance(v, int) for v in vals):
            continue
        if len(set(vals)) != n or min(vals) <= 0 or max(vals) > 10000:
            continue
            
        # Run edge optimization
        edges, sc = greedy_and_local_search_edges(vals, time_limit=deadline - 5)
        
        if len(edges) < m:
            continue
            
        # Run value optimization
        vals, sc = local_search_values(vals, edges, time_limit=deadline - 2)
        
        if sc < best_score:
            best_score = sc
            best_result = (list(vals), [list(e) for e in edges])
            
            # If we get a very good score, we might stop early?
            # Theoretical minimum is likely around 10-20.
            if best_score <= 12:
                break

    # If we still have time, do a few random restarts
    while time.time() < deadline - 10:
        vals = random.sample(range(1, 1001), n)
        if len(set(vals)) != n:
            continue
            
        edges, sc = greedy_and_local_search_edges(vals, time_limit=deadline - 5)
        if len(edges) < m:
            continue
            
        vals, sc = local_search_values(vals, edges, time_limit=deadline - 2)
        
        if sc < best_score:
            best_score = sc
            best_result = (list(vals), [list(e) for e in edges])

    # Fallback
    if best_result is None:
        vals = list(range(1, n + 1))
        edges = [list(e) for e in list(itertools.combinations(range(n), 2))[:m]]
        best_result = (vals, edges)
        
    vals, edges = best_result
    
    # Ensure types are correct
    vals = [int(v) for v in vals]
    edges = [[int(i), int(j)] for i, j in edges]
    
    return {'values': vals, 'edges': edges}

if __name__ == '__main__':
    res = construct()
    print(res)
