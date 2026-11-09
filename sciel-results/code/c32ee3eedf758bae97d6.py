"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

_DEADLINE = time.time() + 80  # safety margin before 90s kill

def _score_fast(values, edges):
    """Compute raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _precompute_all_edges(values, n):
    """Precompute sum and product for all possible edges."""
    edge_data = []
    for i in range(n):
        for j in range(i + 1, n):
            s = values[i] + values[j]
            p = values[i] * values[j]
            edge_data.append((i, j, s, p))
    return edge_data

def _edge_index_map(n):
    """Map (i,j) to index in edge_data."""
    m = n * (n - 1) // 2
    edge_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map[(i, j)] = idx
            idx += 1
    return edge_map

def _generate_structured_values(n, max_val=10000):
    """Generate diverse candidate value sets with collision-promoting patterns."""
    candidates = []
    
    # 1. Consecutive small integers
    candidates.append(list(range(1, n + 1)))
    
    # 2. Even numbers
    candidates.append(list(range(2, 2 * n + 1, 2)))
    
    # 3. Pairs that share sums (additive structure)
    # e.g., values where many pairs sum to same value
    for base in range(10, 30):
        vals = []
        for i in range(n):
            if i < n // 2:
                vals.append(i + 1)
            else:
                vals.append(base - (i - n // 2))
        if len(set(vals)) == n and min(vals) >= 1 and max(vals) <= max_val:
            candidates.append(sorted(vals))
    
    # 4. Multiplicative structure: factors of highly composite numbers
    # Many pairs multiply to same value
    for base in [12, 24, 36, 48, 60, 72, 90, 108, 120, 144]:
        vals = []
        for d in range(1, n + 1):
            if base % d == 0:
                vals.append(d)
            else:
                vals.append(d + base)
        if len(set(vals)) == n and min(vals) >= 1 and max(vals) <= max_val:
            candidates.append(sorted(vals))
    
    # 5. Arithmetic progressions
    for start in range(1, 10):
        for step in range(1, 6):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= max_val and min(vals) >= 1:
                candidates.append(vals)
    
    # 6. Random small values with seed
    random.seed(42)
    for _ in range(50):
        vals = sorted(random.sample(range(1, 80), n))
        candidates.append(vals)
    
    # 7. Powers and multiples
    candidates.append([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    
    # 8. Fibonacci-like
    candidates.append([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610])
    
    # 9. Narrow range
    for lo in range(1, 30):
        hi = lo + 10
        if hi <= max_val:
            vals = random.sample(range(lo, hi + 1), min(n, hi - lo + 1))
            if len(vals) == n:
                candidates.append(sorted(vals))
    
    # 10. Mixed small/medium
    candidates.append([2, 3, 4, 6, 8, 12, 1, 5, 7, 9, 10, 11, 13, 14])
    
    return candidates

def _simulated_annealing(values, edges, n, m, max_iter=50000):
    """
    Simulated annealing over both values and edges simultaneously.
    State: (values, edge_set)
    Move: change one value, or swap one edge
    Objective: max(distinct_sums, distinct_prods)
    """
    random.seed(12345)
    
    # Precompute all possible edge indices
    all_edges = list(combinations(range(n), 2))
    edge_map = {(i, j): idx for idx, (i, j) in enumerate(all_edges)}
    
    # Current state
    cur_values = list(values)
    cur_edges = set(edges)
    
    # Compute initial score
    def compute_score(vals, edge_set):
        sums = set()
        prods = set()
        for (u, v) in edge_set:
            a = vals[u]
            b = vals[v]
            sums.add(a + b)
            prods.add(a * b)
        return max(len(sums), len(prods))
    
    cur_score = compute_score(cur_values, cur_edges)
    best_score = cur_score
    best_values = list(cur_values)
    best_edges = set(cur_edges)
    
    # Temperature schedule
    T = 50.0
    T_min = 0.01
    cooling_rate = 0.999
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE:
            break
        if cur_score <= 2:
            break
        
        # Choose move type: 70% edge swap, 30% value change
        move_type = random.random()
        
        if move_type < 0.7:
            # Edge swap: remove one edge, add one
            if len(cur_edges) < m:
                # Add an edge
                candidate_edges = [e for e in all_edges if e not in cur_edges]
                if not candidate_edges:
                    break
                new_edge = random.choice(candidate_edges)
                new_edges = cur_edges | {new_edge}
                new_score = compute_score(cur_values, new_edges)
                
                delta = new_score - cur_score
                if delta <= 0 or random.random() < pow(2.718281828, -delta / max(T, T_min)):
                    cur_edges = new_edges
                    cur_score = new_score
            else:
                # Swap: remove one, add one
                edge_to_remove = random.choice(list(cur_edges))
                candidate_edges = [e for e in all_edges if e not in cur_edges]
                if not candidate_edges:
                    break
                edge_to_add = random.choice(candidate_edges)
                new_edges = cur_edges - {edge_to_remove} | {edge_to_add}
                new_score = compute_score(cur_values, new_edges)
                
                delta = new_score - cur_score
                if delta <= 0 or random.random() < pow(2.718281828, -delta / max(T, T_min)):
                    cur_edges = new_edges
                    cur_score = new_score
        else:
            # Value change: modify one value
            idx = random.randint(0, n - 1)
            old_val = cur_values[idx]
            
            # Try a random new value
            new_val = random.randint(1, 10000)
            # Ensure distinctness
            while new_val in cur_values:
                new_val = random.randint(1, 10000)
            
            cur_values[idx] = new_val
            new_score = compute_score(cur_values, cur_edges)
            
            delta = new_score - cur_score
            if delta <= 0 or random.random() < pow(2.718281828, -delta / max(T, T_min)):
                pass  # Accept
            else:
                # Reject: revert
                cur_values[idx] = old_val
            
            # Update best
            if cur_score < best_score:
                best_score = cur_score
                best_values = list(cur_values)
                best_edges = set(cur_edges)
        
        # Update best for edge moves
        if cur_score < best_score:
            best_score = cur_score
            best_values = list(cur_values)
            best_edges = set(cur_edges)
        
        # Cool down
        T = max(T * cooling_rate, T_min)
    
    return best_score, best_values, [list(e) for e in best_edges]

def _try_candidate(values, n, m, max_iter=20000):
    """Try a candidate value set with simulated annealing."""
    if time.time() > _DEADLINE:
        return None
    
    # Generate initial edges: most popular sums/products
    edge_data = _precompute_all_edges(values, n)
    sum_counts = defaultdict(int)
    prod_counts = defaultdict(int)
    for i, j, s, p in edge_data:
        sum_counts[s] += 1
        prod_counts[p] += 1
    
    edge_popularity = []
    for idx, (i, j, s, p) in enumerate(edge_data):
        edge_popularity.append(sum_counts[s] + prod_counts[p])
    
    sorted_edges = sorted(range(len(edge_data)), key=lambda x: -edge_popularity[x])
    initial_edges = [list(edge_data[idx][:2]) for idx in sorted_edges[:m]]
    
    score, values_out, edges = _simulated_annealing(values, initial_edges, n, m, max_iter)
    return score, values_out, edges

def construct():
    n = 14
    m = 35
    max_val = 10000
    
    best_score = m + 1
    best_result = None
    
    candidates = _generate_structured_values(n, max_val)
    
    for vals in candidates:
        if time.time() > _DEADLINE:
            break
        
        # Validate
        if len(set(vals)) != n or min(vals) < 1 or max(vals) > max_val:
            continue
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            continue
        
        # Determine max iterations based on remaining time
        remaining = _DEADLINE - time.time()
        if remaining < 3:
            break
        max_iter = min(30000, int(remaining * 3000))
        max_iter = max(max_iter, 2000)
        
        result = _try_candidate(vals, n, m, max_iter)
        if result is None:
            continue
        score, values, edges = result
        if score < best_score:
            best_score = score
            best_result = (values, edges)
            if best_score <= 2:
                break
    
    if best_result is None:
        # Fallback
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    else:
        values, edges = best_result
    
    # Ensure types are correct
    values = [int(v) for v in values]
    edges = [[int(u), int(v)] for u, v in edges]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
