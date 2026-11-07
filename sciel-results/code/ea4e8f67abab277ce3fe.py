"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s limit
    
    n = P['n']
    m = P['m']
    max_val = P['maximum']
    
    # All possible edges (91 total for 14 vertices)
    all_edges = list(combinations(range(n), 2))
    
    best_score = float('inf')
    best_result = None
    
    # Strategy: Try to find values and edges that minimize the max of distinct sums and distinct products
    # 
    # Key insight: If we use small numbers, sums will be small and products will be small.
    # But we need 35 edges out of 91 possible edges.
    #
    # Let's try a structured approach:
    # 1. Generate candidate value sets
    # 2. For each, select edges greedily to minimize the objective
    
    def evaluate(values, edges):
        """Compute the objective: max(distinct sums, distinct products)"""
        sums = set()
        products = set()
        for u, v in edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            sums.add(s)
            products.add(p)
        return max(len(sums), len(products))
    
    def greedy_edge_selection(values, target_edges=35):
        """Greedy: add edges that minimize the increase in distinct sums/products"""
        edges = []
        sums = set()
        products = set()
        
        # Precompute sum and product for each edge
        edge_data = []
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_data.append((u, v, s, p))
        
        # Sort by "rarity" - prefer edges whose sum/product are already common
        # Actually, let's be smarter: try to add edges that reuse existing sums/products
        
        remaining = list(range(len(edge_data)))
        
        while len(edges) < target_edges and remaining:
            best_idx = -1
            best_score = float('inf')
            
            for idx in remaining:
                u, v, s, p = edge_data[idx]
                # Score: how many new distinct values this edge introduces
                new_s = 0 if s in sums else 1
                new_p = 0 if p in products else 1
                score = new_s + new_p
                # Tie-break: prefer smaller sums and products
                if score < best_score or (score == best_score and (s + p) < best_score):
                    if score < best_score:
                        best_score = score
                        best_idx = idx
                    elif score == best_score:
                        # Tie-break by sum+product
                        if best_idx == -1 or (s + p) < (edge_data[best_idx][2] + edge_data[best_idx][3]):
                            best_idx = idx
            
            if best_idx == -1:
                break
            
            u, v, s, p = edge_data[best_idx]
            edges.append((u, v))
            sums.add(s)
            products.add(p)
            remaining.remove(best_idx)
        
        return edges
    
    def random_value_set(rng):
        """Generate a random set of 14 distinct positive integers <= 10000"""
        # Mix of strategies
        strategy = rng.random()
        
        if strategy < 0.3:
            # Small consecutive numbers
            start = rng.randint(1, 20)
            vals = []
            for i in range(14):
                vals.append(start + i * rng.randint(1, 3))
            # Ensure distinct and <= 10000
            vals = sorted(set(vals))
            while len(vals) < 14:
                candidate = rng.randint(1, 10000)
                if candidate not in vals:
                    vals.append(candidate)
            vals = vals[:14]
            
        elif strategy < 0.6:
            # Geometric progression
            start = rng.randint(1, 5)
            ratio = rng.uniform(1.2, 2.0)
            vals = []
            for i in range(14):
                val = int(start * (ratio ** i))
                vals.append(min(val, 10000))
            vals = sorted(set(vals))
            while len(vals) < 14:
                candidate = rng.randint(1, 10000)
                if candidate not in vals:
                    vals.append(candidate)
            vals = vals[:14]
            
        else:
            # Random
            vals = random.sample(range(1, 10001), 14)
        
        return sorted(vals)
    
    # Try several random value sets with greedy edge selection
    rng = random.Random(42)
    attempts = 0
    
    while time.time() < deadline and attempts < 2000:
        attempts += 1
        values = random_value_set(rng)
        edges = greedy_edge_selection(values)
        
        if len(edges) == 35:
            score = evaluate(values, edges)
            if score < best_score:
                best_score = score
                best_result = {'values': values, 'edges': [list(e) for e in edges]}
        
        # If we've found a very good score, do local search
        if best_score <= 12 and time.time() < deadline - 10:
            # Local search: try swapping some values
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(n):
                    if time.time() >= deadline - 5:
                        break
                    old_val = values[i]
                    for new_val in [old_val - 1, old_val + 1, old_val - 2, old_val + 2, old_val // 2, old_val * 2]:
                        if new_val < 1 or new_val > 10000:
                            continue
                        if new_val in values and new_val != old_val:
                            continue
                        # Check distinctness
                        test_values = values.copy()
                        test_values[i] = new_val
                        if len(set(test_values)) != 14:
                            continue
                        
                        edges_test = greedy_edge_selection(test_values)
                        if len(edges_test) == 35:
                            score_test = evaluate(test_values, edges_test)
                            if score_test < best_score:
                                best_score = score_test
                                best_result = {'values': test_values, 'edges': [list(e) for e in edges_test]}
                                values = test_values
                                improved = True
                                break
                    if improved:
                        break
    
    # If we didn't find a good solution, use a fallback
    if best_result is None:
        values = list(range(1, 15))
        edges = [list(e) for e in all_edges[:35]]
        best_result = {'values': values, 'edges': edges}
    
    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return best_result
# EVOLVE-BLOCK-END
