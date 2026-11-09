import time
import random
import math
from itertools import combinations
from collections import defaultdict, Counter

_DEADLINE = time.time() + 85

def _score_detailed(values, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        sums.add(values[u] + values[v])
        prods.add(values[u] * values[v])
    return sums, prods, max(len(sums), len(prods))

def _optimize_edges(values, n, m, time_budget=10.0):
    """
    Optimizes edge selection to minimize max(distinct_sums, distinct_prods).
    """
    start_time = time.time()
    all_edges = list(combinations(range(n), 2))
    edge_sp = {}
    for e in all_edges:
        u, v = e
        edge_sp[e] = (values[u] + values[v], values[u] * values[v])
    
    # Greedy Initialization
    current_edges = set()
    sum_counts = Counter()
    prod_counts = Counter()
    sums_set = set()
    prods_set = set()
    
    # Sort edges by "commonness" of sum and product
    sum_freq = Counter(s for s, p in edge_sp.values())
    prod_freq = Counter(p for s, p in edge_sp.values())
    
    edge_scores = []
    for e in all_edges:
        s, p = edge_sp[e]
        score = sum_freq[s] * prod_freq[p]
        edge_scores.append((score, e))
    edge_scores.sort(key=lambda x: -x[0])
    
    for score, e in edge_scores:
        if len(current_edges) >= m:
            break
        current_edges.add(e)
        s, p = edge_sp[e]
        sum_counts[s] += 1
        prod_counts[p] += 1
        sums_set.add(s)
        prods_set.add(p)
        
    current_score = max(len(sums_set), len(prods_set))
    best_score = current_score
    best_edges = set(current_edges)
    
    # Simulated Annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start_time < time_budget and T > T_min:
        if time.time() > _DEADLINE - 2:
            break
            
        # Pick a random edge to remove
        rem = random.choice(list(current_edges))
        rs, rp = edge_sp[rem]
        
        # Pick a random edge to add
        add = random.choice(all_edges)
        if add in current_edges:
            continue
            
        as_, ap = edge_sp[add]
        
        # Calculate delta
        delta_s = 0
        if sum_counts[rs] == 1: delta_s -= 1
        if sum_counts[as_] == 0: delta_s += 1
        
        delta_p = 0
        if prod_counts[rp] == 1: delta_p -= 1
        if prod_counts[ap] == 0: delta_p += 1
        
        new_sums = len(sums_set) + delta_s
        new_prods = len(prods_set) + delta_p
        new_score = max(new_sums, new_prods)
        
        delta_score = new_score - current_score
        
        if delta_score < 0 or random.random() < math.exp(-delta_score / T):
            # Accept
            current_edges.discard(rem)
            current_edges.add(add)
            
            sum_counts[rs] -= 1
            if sum_counts[rs] == 0:
                del sum_counts[rs]
                sums_set.discard(rs)
            prod_counts[rp] -= 1
            if prod_counts[rp] == 0:
                del prod_counts[rp]
                prods_set.discard(rp)
                
            sum_counts[as_] += 1
            if sum_counts[as_] == 1:
                sums_set.add(as_)
            prod_counts[ap] += 1
            if prod_counts[ap] == 1:
                prods_set.add(ap)
                
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
                
        T *= cooling
        
    return best_score, best_edges

def _generate_candidates():
    candidates = []
    seen = set()
    
    def add(vals):
        if len(vals) != 14: return
        if len(set(vals)) != 14: return
        if any(v < 1 or v > 10000 for v in vals): return
        key = tuple(sorted(vals))
        if key not in seen:
            seen.add(key)
            candidates.append(sorted(vals))
            
    # 1. Pair-based candidates
    # Pick P, then pick pairs (x, P//x)
    for P in [100, 200, 300, 500, 1000, 2000, 5000, 10000]:
        for k in range(6, 8): # 6 to 7 pairs
            pairs = []
            possible_x = [x for x in range(1, int(math.sqrt(P)) + 1) if P % x == 0]
            random.shuffle(possible_x)
            for x in possible_x:
                if len(pairs) >= k: break
                y = P // x
                if x != y and x not in [p[0] for p in pairs] and y not in [p[1] for p in pairs]:
                    pairs.append((x, y))
            if len(pairs) == k:
                vals = []
                for x, y in pairs:
                    vals.append(x)
                    vals.append(y)
                # Add dummy values
                while len(vals) < 14:
                    v = random.randint(1000, 10000)
                    if v not in vals:
                        vals.append(v)
                add(vals)
                
    # 2. Smooth numbers
    smooth_sets = [
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
    ]
    for s in smooth_sets:
        add(s)
        
    # 3. Consecutive
    add(list(range(1, 15)))
    
    # 4. Random
    for _ in range(20):
        vals = sorted(random.sample(range(1, 100), 14))
        add(vals)
        
    return candidates

def construct():
    n = 14
    m = 35
    
    best_score = m
    best_result = None
    
    candidates = _generate_candidates()
    
    for vals in candidates:
        if time.time() > _DEADLINE - 5:
            break
            
        score, edges = _optimize_edges(vals, n, m, time_budget=10.0)
        
        if score < best_score:
            best_score = score
            best_result = (vals, [list(e) for e in edges])
            
    if best_result is None:
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    else:
        values, edges = best_result
        
    return {'values': [int(v) for v in values], 'edges': [[int(u), int(v)] for u, v in edges]}
