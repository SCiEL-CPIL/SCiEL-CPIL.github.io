import time
import random
from itertools import combinations

def _score(values, edges):
    sums = set()
    products = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        products.add(values[i] * values[j])
    return len(sums), len(products), max(len(sums), len(products))

def _greedy_select_edges(values, num_edges=35, deadline=None):
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    selected_edges = []
    current_sums = set()
    current_products = set()
    
    start_time = time.time()
    
    for _ in range(num_edges):
        if deadline and time.time() > deadline:
            break
            
        best_edge = None
        best_score = float('inf')
        best_tie_breaker = float('inf')
        
        for e in all_edges:
            if e in selected_edges:
                continue
                
            i, j = e
            s = values[i] + values[j]
            p = values[i] * values[j]
            
            is_new_sum = s not in current_sums
            is_new_prod = p not in current_products
            
            new_sum_count = len(current_sums) + (1 if is_new_sum else 0)
            new_prod_count = len(current_products) + (1 if is_new_prod else 0)
            
            candidate_score = max(new_sum_count, new_prod_count)
            tie_breaker = (is_new_sum + is_new_prod)
            
            if candidate_score < best_score or (candidate_score == best_score and tie_breaker < best_tie_breaker):
                best_score = candidate_score
                best_tie_breaker = tie_breaker
                best_edge = e

        if best_edge is None:
            break
            
        selected_edges.append(best_edge)
        i, j = best_edge
        current_sums.add(values[i] + values[j])
        current_products.add(values[i] * values[j])
        
    return selected_edges

def _local_search_values(values, edges, deadline):
    n = len(values)
    current_score, _, _ = _score(values, edges)
    best_score = current_score
    best_values = values[:]
    
    while time.time() < deadline:
        i = random.randint(0, n - 1)
        old_val = values[i]
        
        if random.random() < 0.3:
            new_val = random.randint(1, 20)
        elif random.random() < 0.6:
            new_val = random.randint(1, 100)
        else:
            new_val = random.randint(1, 10000)
            
        if new_val == old_val:
            continue
            
        # Check distinctness
        is_distinct = True
        for idx, v in enumerate(values):
            if idx != i and v == new_val:
                is_distinct = False
                break
                
        if not is_distinct:
            continue

        values[i] = new_val
        new_score, _, _ = _score(values, edges)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_values = values[:]
        else:
            values[i] = old_val
            
    return best_values, best_score

def construct():
    deadline = time.time() + 55
    
    best_overall_score = float('inf')
    best_overall_values = None
    best_overall_edges = None
    
    n = 14
    
    def get_initial_values():
        strat = random.randint(0, 3)
        if strat == 0:
            return list(range(1, n + 1))
        elif strat == 1:
            return random.sample(range(1, 50), n)
        elif strat == 2:
            return random.sample(range(1, 200), n)
        else:
            # Mix of small and large
            return random.sample(range(1, 1000), n)

    while time.time() < deadline:
        values = get_initial_values()
        
        # Phase 1: Greedy edges
        edges = _greedy_select_edges(values, 35, deadline)
        
        # Phase 2: Optimize values
        # Limit time for this phase to allow more restarts
        # If we have plenty of time, spend more here.
        time_left = deadline - time.time()
        val_search_time = min(3.0, max(0.1, time_left * 0.2))
        val_search_deadline = time.time() + val_search_time
        
        values, score = _local_search_values(values, edges, val_search_deadline)
        
        # Phase 3: Re-select edges with new values
        edges = _greedy_select_edges(values, 35, deadline)
        
        # Phase 4: Short value polish
        if time.time() < deadline:
            short_deadline = time.time() + min(1.0, max(0.05, (deadline - time.time()) * 0.1))
            values, score = _local_search_values(values, edges, short_deadline)
            edges = _greedy_select_edges(values, 35, deadline)
        
        _, _, final_score = _score(values, edges)
        
        if final_score < best_overall_score:
            best_overall_score = final_score
            best_overall_values = values[:]
            best_overall_edges = [list(e) for e in edges]
            
    if best_overall_values is None:
        # Fallback
        best_overall_values = list(range(1, 15))
        best_overall_edges = [list(e) for e in combinations(range(14), 2)[:35]]
        
    return {
        'values': [int(x) for x in best_overall_values],
        'edges': [[int(i), int(j)] for i, j in best_overall_edges]
    }
