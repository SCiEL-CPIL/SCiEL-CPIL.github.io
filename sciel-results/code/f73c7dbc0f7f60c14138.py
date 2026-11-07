"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _raw_score(values, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        s = values[u] + values[v]
        p = values[u] * values[v]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _all_pairs(n):
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j))
    return pairs

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    n = 14
    m = 35
    max_val = 10000
    all_pairs = _all_pairs(n)
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Initial value sets to try
    initial_sets = [
        list(range(1, n+1)),
        [2**i for i in range(n)],  # powers of 2
        [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
        [1,3,5,7,9,11,13,15,17,19,21,23,25,27],  # odd numbers
        [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192],
        [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
    ]
    
    def _evaluate(values, edges):
        return _raw_score(values, edges)
    
    def _sa_optimize(values, edges, time_limit):
        """Simulated annealing on both values and edges."""
        nonlocal best_score, best_values, best_edges
        
        cur_score = _evaluate(values, edges)
        if cur_score < best_score:
            best_score = cur_score
            best_values = list(values)
            best_edges = [list(e) for e in edges]
        
        cur_values = list(values)
        cur_edges = set(tuple(e) for e in edges)
        
        T = 5.0
        cooling = 0.999
        min_T = 0.01
        
        while T > min_T:
            if time.time() > time_limit:
                break
            
            # Random move
            move_type = random.random()
            
            if move_type < 0.5:
                # Change a value
                idx = random.randint(0, n-1)
                old_val = cur_values[idx]
                new_val = random.randint(1, max_val)
                if new_val == old_val:
                    continue
                # Check distinctness
                if new_val in cur_values:
                    continue
                cur_values[idx] = new_val
                
                new_score = _evaluate(cur_values, list(cur_edges))
                delta = new_score - cur_score
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_values = list(cur_values)
                        best_edges = [list(e) for e in cur_edges]
                else:
                    cur_values[idx] = old_val
                    
            elif move_type < 0.75:
                # Swap two values
                i, j = random.sample(range(n), 2)
                cur_values[i], cur_values[j] = cur_values[j], cur_values[i]
                
                new_score = _evaluate(cur_values, list(cur_edges))
                delta = new_score - cur_score
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_values = list(cur_values)
                        best_edges = [list(e) for e in cur_edges]
                else:
                    cur_values[i], cur_values[j] = cur_values[j], cur_values[i]
                    
            else:
                # Change an edge: remove one, add one
                if len(cur_edges) < m:
                    # Add an edge
                    candidates = [p for p in all_pairs if p not in cur_edges]
                    if candidates:
                        new_edge = tuple(random.choice(candidates))
                        cur_edges.add(new_edge)
                        
                        new_score = _evaluate(cur_values, list(cur_edges))
                        delta = new_score - cur_score
                        if delta <= 0 or random.random() < math.exp(-delta / T):
                            cur_score = new_score
                            if cur_score < best_score:
                                best_score = cur_score
                                best_values = list(cur_values)
                                best_edges = [list(e) for e in cur_edges]
                        else:
                            cur_edges.discard(new_edge)
                else:
                    # Swap an edge
                    cur_edge_list = list(cur_edges)
                    old_edge = random.choice(cur_edge_list)
                    candidates = [p for p in all_pairs if p not in cur_edges and p != old_edge]
                    if candidates:
                        new_edge = tuple(random.choice(candidates))
                        cur_edges.discard(old_edge)
                        cur_edges.add(new_edge)
                        
                        new_score = _evaluate(cur_values, list(cur_edges))
                        delta = new_score - cur_score
                        if delta <= 0 or random.random() < math.exp(-delta / T):
                            cur_score = new_score
                            if cur_score < best_score:
                                best_score = cur_score
                                best_values = list(cur_values)
                                best_edges = [list(e) for e in cur_edges]
                        else:
                            cur_edges.discard(new_edge)
                            cur_edges.add(old_edge)
            
            T *= cooling
        
        return best_score, best_values, best_edges
    
    # Try multiple initial configurations
    random.seed(42)
    
    for init_values in initial_sets:
        if time.time() > deadline:
            break
        
        # For each initial value set, try different edge selections
        # Greedy edge selection: pick edges that minimize distinct sums/products
        def _greedy_edges(values, num_edges):
            """Select edges greedily to minimize distinct sums and products."""
            edges = set()
            sums = set()
            prods = set()
            
            # Score each potential edge by how "new" its sum and product are
            def edge_score(u, v):
                s = values[u] + values[v]
                p = values[u] * values[v]
                score = 0
                if s not in sums:
                    score += 10
                if p not in prods:
                    score += 10
                return score
            
            for _ in range(num_edges):
                best_edge = None
                best_s = float('inf')
                for u, v in all_pairs:
                    if (u, v) in edges:
                        continue
                    s = edge_score(u, v)
                    if s < best_s:
                        best_s = s
                        best_edge = (u, v)
                if best_edge:
                    edges.add(best_edge)
            
            return list(edges)
        
        edges = _greedy_edges(init_values, m)
        
        # Run SA from this starting point
        remaining = deadline - time.time()
        if remaining > 5:
            sa_time = min(remaining - 2, 30)
            _sa_optimize(init_values, edges, start_time + sa_time)
    
    # Final fallback
    if best_values is None or best_edges is None:
        best_values = list(range(1, n+1))
        best_edges = [list(e) for e in all_pairs[:m]]
    
    # Ensure valid return types
    return {'values': [int(v) for v in best_values], 'edges': [[int(u), int(v)] for u, v in best_edges]}

# EVOLVE-BLOCK-END
