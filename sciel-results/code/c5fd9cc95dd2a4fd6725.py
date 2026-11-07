import heapq
import itertools
import random
import time

def construct():
    """
    Constructs a graph with 14 vertices and 35 edges to minimize the max of distinct sums and products.
    Uses a priority-queue-based greedy edge selection followed by local search refinement.
    """
    n = 14
    m = 35
    
    # Candidate value sets
    candidates = []
    
    # 1. Multiples of small bases (high collision potential)
    for base in [2, 3, 4, 5, 6, 8, 10, 12]:
        for start in range(1, 15):
            vals = [base * (start + i) for i in range(n)]
            if max(vals) <= 10000 and min(vals) >= 1:
                candidates.append(vals)
                
    # 2. Specific structured sets
    candidates.append(list(range(2, 2*n+2, 2))) # 2, 4, ..., 28
    candidates.append(list(range(3, 3*n+3, 3))) # 3, 6, ..., 42
    candidates.append(list(range(1, n + 1)))
    candidates.append(list(range(2, n + 2)))
    
    # 3. Random sets with bias towards small numbers
    random.seed(42)
    for _ in range(100):
        vals = random.sample(range(1, 40), n)
        candidates.append(vals)
        
    # 4. Random multiples
    for _ in range(100):
        base = random.choice([2, 3, 4, 6])
        vals = [base * x for x in random.sample(range(1, 25), n)]
        if len(set(vals)) == n and max(vals) <= 10000:
             candidates.append(vals)

    best_score = float('inf')
    best_edges = None
    best_values = None
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))

    def greedy_select_edges_pq(values):
        all_edges = list(itertools.combinations(range(n), 2))
        
        # Precompute sum and product for each edge
        edge_props = {}
        for i, j in all_edges:
            s = values[i] + values[j]
            p = values[i] * values[j]
            edge_props[(i, j)] = (s, p)
            
        selected_edges = []
        selected_set = set()
        current_sums = set()
        current_prods = set()
        
        for _ in range(m):
            heap = []
            
            # Build heap with current state
            # Priority: (new_max_distinct, total_new_items, edge)
            for edge in all_edges:
                if edge in selected_set:
                    continue
                s, p = edge_props[edge]
                ds = 1 if s not in current_sums else 0
                dp = 1 if p not in current_prods else 0
                
                new_sums_count = len(current_sums) + ds
                new_prods_count = len(current_prods) + dp
                new_max = max(new_sums_count, new_prods_count)
                
                heapq.heappush(heap, (new_max, ds + dp, edge))
            
            if not heap:
                break
                
            new_max, total_new, best_edge = heapq.heappop(heap)
            
            i, j = best_edge
            s, p = edge_props[(i, j)]
            current_sums.add(s)
            current_prods.add(p)
            selected_edges.append((i, j))
            selected_set.add((i, j))
            
        return selected_edges, max(len(current_sums), len(current_prods))

    for values in candidates:
        if len(set(values)) != n or max(values) > 10000 or min(values) < 1:
            continue
            
        edges, score = greedy_select_edges_pq(values)
        
        if score < best_score:
            best_score = score
            best_edges = edges
            best_values = values
            
        if best_score <= 10:
            break

    # Local search refinement on the best found
    if best_edges is not None:
        current_edges = set(best_edges)
        current_score = best_score
        all_edges = list(itertools.combinations(range(n), 2))
        
        start_time = time.time()
        improved = True
        while improved and time.time() - start_time < 10:
            improved = False
            edges_list = list(current_edges)
            random.shuffle(edges_list)
            
            for edge_in in edges_list:
                # Find a random edge to swap in
                edge_out = None
                for _ in range(20):
                    candidate_out = tuple(sorted(random.sample(range(n), 2)))
                    if candidate_out not in current_edges:
                        edge_out = candidate_out
                        break
                if edge_out is None:
                    continue
                
                new_edges = set(current_edges)
                new_edges.remove(edge_in)
                new_edges.add(edge_out)
                
                new_score = evaluate(best_values, new_edges)
                
                if new_score < current_score:
                    current_edges = new_edges
                    current_score = new_score
                    improved = True
                    break
        
        if current_score < best_score:
            best_score = current_score
            best_edges = list(current_edges)

    if best_edges is None:
        best_values = list(range(1, n + 1))
        all_edges = list(itertools.combinations(range(n), 2))
        best_edges = all_edges[:m]

    return {
        'values': [int(v) for v in best_values],
        'edges': [[int(i), int(j)] for i, j in best_edges]
    }
