"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return sums, prods, max(len(sums), len(prods))

def _greedy_init_edges(n, m, edge_sp):
    all_edges = list(combinations(range(n), 2))
    total = len(all_edges)
    sum_counts = Counter()
    prod_counts = Counter()
    chosen = set()
    
    first = all_edges[random.randint(0, total - 1)]
    chosen.add(first)
    s0, p0 = edge_sp[first]
    sum_counts[s0] += 1
    prod_counts[p0] += 1
    
    for _ in range(m - 1):
        best_edge = None
        best_key = (0, 0, 0)
        if total <= 200:
            candidates = all_edges
        else:
            candidates = random.sample(all_edges, 80)
        for e in candidates:
            if e in chosen:
                continue
            s, p = edge_sp[e]
            sc = sum_counts.get(s, 0)
            pc = prod_counts.get(p, 0)
            key = (sc, pc, random.random())
            if key > best_key:
                best_key = key
                best_edge = e
        if best_edge is None:
            for _ in range(50):
                e = all_edges[random.randint(0, total - 1)]
                if e not in chosen:
                    best_edge = e
                    break
            if best_edge is None:
                continue
        chosen.add(best_edge)
        s, p = edge_sp[best_edge]
        sum_counts[s] += 1
        prod_counts[p] += 1
    
    return chosen

def _fast_anneal(values, n, m, edge_sp, time_budget=5.0):
    start_time = time.time()
    all_edges = list(combinations(range(n), 2))
    total_edges = len(all_edges)
    
    current_edges = _greedy_init_edges(n, m, edge_sp)
    
    current_sums_set, current_prods_set, current_score = _score_detailed(values, list(current_edges))
    best_score = current_score
    best_edges = set(current_edges)
    
    sum_counts = Counter()
    prod_counts = Counter()
    for e in current_edges:
        s, p = edge_sp[e]
        sum_counts[s] += 1
        prod_counts[p] += 1
    
    T = 6.0
    T_min = 0.01
    cooling = 0.9997
    max_iter = 150000
    
    edges_list = list(current_edges)
    
    for it in range(max_iter):
        if it % 500 == 0 and time.time() - start_time > time_budget:
            break
        if time.time() > _DEADLINE - 3:
            break
        if current_score <= 3:
            break
        
        remove_edge = edges_list[random.randint(0, len(edges_list) - 1)]
        add_edge = all_edges[random.randint(0, total_edges - 1)]
        
        if add_edge == remove_edge:
            continue
        if add_edge in current_edges:
            continue

        rs, rp = edge_sp[remove_edge]
        as_, ap = edge_sp[add_edge]
        
        delta_sums = 0
        if sum_counts[rs] == 1:
            delta_sums -= 1
        if sum_counts.get(as_, 0) == 0:
            delta_sums += 1
        
        delta_prods = 0
        if prod_counts[rp] == 1:
            delta_prods -= 1
        if prod_counts.get(ap, 0) == 0:
            delta_prods += 1
        
        new_num_sums = len(current_sums_set) + delta_sums
        new_num_prods = len(current_prods_set) + delta_prods
        new_score = max(new_num_sums, new_num_prods)
        
        delta_score = new_score - current_score
        
        if delta_score < 0 or (T > T_min and random.random() < math.exp(-delta_score / T)):
            current_edges.discard(remove_edge)
            current_edges.add(add_edge)
            
            sum_counts[rs] -= 1
            if sum_counts[rs] == 0:
                del sum_counts[rs]
                current_sums_set.discard(rs)
            prod_counts[rp] -= 1
            if prod_counts[rp] == 0:
                del prod_counts[rp]
                current_prods_set.discard(rp)
            
            sum_counts[as_] = sum_counts.get(as_, 0) + 1
            prod_counts[ap] = prod_counts.get(ap, 0) + 1
            if sum_counts[as_] == 1:
                current_sums_set.add(as_)
            if prod_counts[ap] == 1:
                current_prods_set.add(ap)
            
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
                edges_list = list(current_edges)
        
        T = max(T_min, T * cooling)
    
    return best_score, best_edges

def _polish_edges(values, n, m, edge_sp, initial_edges, time_budget=3.0):
    start_time = time.time()
    all_edges = list(combinations(range(n), 2))
    total_edges = len(all_edges)
    
    current_edges = set(initial_edges)
    current_sums_set, current_prods_set, current_score = _score_detailed(values, list(current_edges))
    best_score = current_score
    best_edges = set(current_edges)
    
    sum_counts = Counter()
    prod_counts = Counter()
    for e in current_edges:
        s, p = edge_sp[e]
        sum_counts[s] += 1
        prod_counts[p] += 1
    
    for _pass in range(15):
        if time.time() - start_time > time_budget:
            break
        if time.time() > _DEADLINE - 3:
            break
        if current_score <= 3:
            break
        
        improved = False
        for rem in list(current_edges):
            if time.time() - start_time > time_budget:
                break
            if time.time() > _DEADLINE - 3:
                break
            
            rs, rp = edge_sp[rem]
            
            sum_counts[rs] -= 1
            prod_counts[rp] -= 1
            temp_sums = len(current_sums_set) - (1 if sum_counts[rs] == 0 else 0)
            temp_prods = len(current_prods_set) - (1 if prod_counts[rp] == 0 else 0)
            
            if sum_counts[rs] == 0:
                current_sums_set.discard(rs)
            if prod_counts[rp] == 0:
                current_prods_set.discard(rp)
            
            best_add = None
            best_new_score = current_score
            
            for add in all_edges:
                if add in current_edges or add == rem:
                    continue
                as_, ap = edge_sp[add]
                
                ns = temp_sums + (1 if sum_counts.get(as_, 0) == 0 else 0)
                np_ = temp_prods + (1 if prod_counts.get(ap, 0) == 0 else 0)
                ns_score = max(ns, np_)
                
                if ns_score < best_new_score:
                    best_new_score = ns_score
                    best_add = add
            
            sum_counts[rs] += 1
            prod_counts[rp] += 1
            current_sums_set.add(rs)
            current_prods_set.add(rp)
            
            if best_add is not None and best_new_score < current_score:
                current_edges.discard(rem)
                current_edges.add(best_add)
                
                as_, ap = edge_sp[best_add]
                
                sum_counts[rs] -= 1
                if sum_counts[rs] == 0:
                    del sum_counts[rs]
                    current_sums_set.discard(rs)
                prod_counts[rp] -= 1
                if prod_counts[rp] == 0:
                    del prod_counts[rp]
                    current_prods_set.discard(rp)
                
                sum_counts[as_] = sum_counts.get(as_, 0) + 1
                prod_counts[ap] = prod_counts.get(ap, 0) + 1
                if sum_counts[as_] == 1:
                    current_sums_set.add(as_)
                if prod_counts[ap] == 1:
                    current_prods_set.add(ap)
                
                current_score = best_new_score
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(current_edges)
        
        if not improved:
            break
    
    return best_score, best_edges

def _generate_value_candidates(n, max_val=10000):
    candidates = []
    seen_keys = set()
    
    def add_candidate(vals):
        if len(vals) != n or len(set(vals)) != n:
            return
        if any(v < 1 or v > max_val for v in vals):
            return
        key = tuple(sorted(vals))
        if key not in seen_keys:
            seen_keys.add(key)
            candidates.append(sorted(vals))
    
    add_candidate(list(range(1, n + 1)))
    
    smooth_sets = [
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128],
        [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
        [1, 2, 4, 6, 8, 12, 16, 24, 36, 48, 64, 96, 144, 192],
        [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30],
        [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48],
        [1, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 30],
    ]
    for s in smooth_sets:
        if len(s) == n:
            add_candidate(s)
    
    for base in [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]:
        vals = []
        for i in range(n):
            v = max(1, int(round(base ** i)))
            vals.append(v)
        seen = set()
        unique = []
        for v in vals:
            while v in seen or v < 1:
                v += 1
            seen.add(v)
            unique.append(v)
        if len(unique) == n and all(v <= max_val for v in unique):
            add_candidate(unique)
    
    random.seed(42)
    for trial in range(40):
        base_vals = list(range(1, n + 1))
        for _ in range(random.randint(2, 6)):
            idx = random.randint(0, n - 1)
            base_vals[idx] = random.randint(1, 60)
        add_candidate(base_vals)
    
    for base_mult in [2, 3, 4, 6]:
        vals = [base_mult * (i + 1) for i in range(n)]
        if all(v <= max_val for v in vals):
            add_candidate(vals)
    
    for start in [1, 2, 3]:
        for step in [1, 2, 3, 4, 5, 6]:
            vals = [start + i * step for i in range(n)]
            if all(v <= max_val for v in vals):
                add_candidate(vals)
    
    small_primes = [2, 3, 5, 7]
    random.seed(123)
    for trial in range(25):
        vals = []
        for i in range(n):
            k = random.randint(2, 3)
            chosen = random.sample(small_primes, k)
            v = 1
            for p in chosen:
                v *= p
            vals.append(v)
        for i in range(n):
            if random.random() < 0.3:
                vals[i] = vals[i] * random.choice([2, 3, 4])
        add_candidate(vals)
    
    random.seed(999)
    for trial in range(20):
        vals = sorted(random.sample(range(1, 120), n))
        add_candidate(vals)
    
    for trial in range(15):
        vals = sorted(random.sample(range(1, 250), n))
        add_candidate(vals)
    
    add_candidate([2**i for i in range(n)])
    
    for trial in range(15):
        vals = []
        for i in range(n):
            if random.random() < 0.5:
                vals.append(2**random.randint(0, 10))
            else:
                vals.append(random.choice([1,2,3,4,5,6,7,8,9,10,12,15,16,18,20,24,25,27,30,32,36,40,48,54,60,64,72,80,81,90,96,100,108,120,144,150,160,180,192,200,216,240,256,270,288,300,320,360,400,432,480,500,512,540,576,600,640,720,800,900,960,1000]))
        add_candidate(vals)
    
    return candidates

def construct():
    n = 14
    m = 35
    max_val = 10000
    
    best_score = m
    best_result = None
    
    candidates = _generate_value_candidates(n, max_val)
    
    def candidate_heuristic(vals):
        rng = max(vals) - min(vals)
        from math import gcd
        common_div_count = 0
        for i in range(len(vals)):
            for j in range(i+1, len(vals)):
                g = gcd(vals[i], vals[j])
                if g > 1:
                    common_div_count += 1
        return (rng, -common_div_count)
    
    candidates.sort(key=candidate_heuristic)
    
    num_candidates = len(candidates)
    
    total_budget = 78.0
    weights = []
    for i in range(num_candidates):
        if i < 3:
            weights.append(2.5)
        elif i < 8:
            weights.append(1.5)
        else:
            weights.append(1.0)
    total_weight = sum(weights)
    
    for idx, vals in enumerate(candidates):
        if time.time() > _DEADLINE - 10:
            break
        
        time_per_candidate = total_budget * weights[idx] / total_weight
        
        all_edges = list(combinations(range(n), 2))
        edge_sp = {}
        for e in all_edges:
            u, v = e
            edge_sp[e] = (vals[u] + vals[v], vals[u] * vals[v])
        
        budget = time_per_candidate * 0.55
        score, edges = _fast_anneal(vals, n, m, edge_sp, time_budget=budget)
        
        polish_budget = min(4.0, time_per_candidate * 0.45)
        if time.time() < _DEADLINE - 6:
            score, edges = _polish_edges(vals, n, m, edge_sp, edges, time_budget=polish_budget)
        
        if score < best_score:
            best_score = score
            best_result = (vals, [list(e) for e in edges])
            if best_score <= 3:
                break
    
    if best_result is None:
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    else:
        values, edges = best_result
    
    values = [int(v) for v in values]
    edges = [[int(u), int(v)] for u, v in edges]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
