import random
import itertools
import time
import heapq

def construct():
    """
    Constructs a graph with 14 vertices and 35 edges to minimize the max of distinct sums and products.
    Uses a priority-queue based greedy edge selection and local search refinement.
    """
    n = 14
    m = 35
    
    # Candidate value sets
    candidates = []
    
    # 1. Small consecutive integers
    candidates.append(list(range(1, n + 1)))
    candidates.append(list(range(2, n + 2)))
    candidates.append(list(range(5, n + 5)))
    candidates.append(list(range(10, n + 10)))
    
    # 2. Multiples
    for k in range(2, 10):
        candidates.append([k * i for i in range(1, n + 1)])
        
    # 3. Structured set with many factor collisions (powers and multiples)
    candidates.append([2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48])
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48])
    candidates.append([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    
    # 4. Random sets with bias towards small numbers and specific structures
    random.seed(42)
    for _ in range(50):
        # Try to create sets with many additive relations
        base = random.randint(1, 5)
        vals = []
        while len(vals) < n:
            v = random.randint(1, 100)
            if v not in vals:
                vals.append(v)
        candidates.append(sorted(vals))
        
    # 5. Specific sets known to have good properties
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    candidates.append([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]) # Powers of 2 (bad for sums, good for products? No, distinct products)
    candidates.append([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) # Invalid (distinct required), but let's check constraints later. Actually constraint is "fourteen distinct".
    
    # Filter candidates
    valid_candidates = []
    for vals in candidates:
        if len(set(vals)) != n:
            continue
        if any(v < 1 or v > 10000 for v in vals):
            continue
        valid_candidates.append(vals)
        
    if not valid_candidates:
        valid_candidates = [list(range(1, n+1))]

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
        
        current_sums = set()
        current_prods = set()
        selected_edges = []
        remaining_edges = set(all_edges)
        
        # Priority queue: (cost, edge)
        # Cost is the marginal increase in max(distinct sums, distinct products)
        # We want to minimize this cost.
        
        # Initialize heap with all edges
        heap = []
        for edge in all_edges:
            i, j = edge
            s = values[i] + values[j]
            p = values[i] * values[j]
            
            new_sums_count = len(current_sums) + (1 if s not in current_sums else 0)
            new_prods_count = len(current_prods) + (1 if p not in current_prods else 0)
            
            cost = max(new_sums_count, new_prods_count)
            # Add a small random tie-breaker to avoid bias in ties? Or just use edge index.
            heapq.heappush(heap, (cost, edge))
            
        for _ in range(m):
            if not heap:
                break
                
            cost, edge = heapq.heappop(heap)
            
            # Check if edge is still remaining
            if edge not in remaining_edges:
                continue
                
            i, j = edge
            s = values[i] + values[j]
            p = values[i] * values[j]
            
            # Update sets
            current_sums.add(s)
            current_prods.add(p)
            selected_edges.append(edge)
            remaining_edges.remove(edge)
            
            # Recalculate costs for remaining edges?
            # This is O(M * E) which is 35 * 91 ~ 3000 ops per step, total ~ 100k. Fast enough.
            # However, simply re-pushing all remaining edges with new costs is simpler but potentially slower if done naively.
            # Let's just re-evaluate all remaining edges and rebuild heap for simplicity and correctness.
            # Since n=14, E=91, M=35. Rebuilding heap 35 times is 35 * 91 log 91 ~ 35 * 600 ~ 21000 ops. Very fast.
            
            new_heap = []
            for rem_edge in remaining_edges:
                ri, rj = rem_edge
                rs = values[ri] + values[rj]
                rp = values[ri] * values[rj]
                
                new_sums_count = len(current_sums) + (1 if rs not in current_sums else 0)
                new_prods_count = len(current_prods) + (1 if rp not in current_prods else 0)
                
                new_cost = max(new_sums_count, new_prods_count)
                heapq.heappush(new_heap, (new_cost, rem_edge))
                
            heap = new_heap
            
        return selected_edges, max(len(current_sums), len(current_prods))

    for values in valid_candidates:
        edges, score = greedy_select_edges_pq(values)
        
        if score < best_score:
            best_score = score
            best_edges = edges
            best_values = values
            
        if best_score <= 15:
            break

    # Local search refinement on the best found
    if best_edges is not None:
        current_edges = set(best_edges)
        current_score = best_score
        all_edges = list(itertools.combinations(range(n), 2))
        
        start_time = time.time()
        improved = True
        while improved and time.time() - start_time < 5:
            improved = False
            edges_list = list(current_edges)
            random.shuffle(edges_list)
            for edge_in in edges_list:
                # Try swapping with a random non-edge
                edge_out = tuple(sorted(random.sample(range(n), 2)))
                if edge_out in current_edges:
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
