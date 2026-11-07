import time
import random
import itertools
import numpy as np

N = 14
M = 35
MAXV = 10000

def _raw_score(values, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _all_edges():
    return list(itertools.combinations(range(N), 2))

def _edge_features(values, all_edges):
    feats = []
    for u, v in all_edges:
        a, b = values[u], values[v]
        feats.append(((u, v), a + b, a * b))
    return feats

def _select_edges_greedy(feats, target_m):
    sums = set()
    prods = set()
    chosen = []
    remaining = list(feats)
    sum_count = {}
    prod_count = {}
    for _, s, p in feats:
        sum_count[s] = sum_count.get(s, 0) + 1
        prod_count[p] = prod_count.get(p, 0) + 1
        
    for _ in range(target_m):
        best_edge = None
        best_key = None
        for idx, (e, s, p) in enumerate(remaining):
            new_sums = len(sums) + (0 if s in sums else 1)
            new_prods = len(prods) + (0 if p in prods else 1)
            cost = max(new_sums, new_prods)
            tie = -(sum_count[s] + prod_count[p])
            key = (cost, tie, idx)
            if best_key is None or key < best_key:
                best_key = key
                best_edge = idx
        e, s, p = remaining.pop(best_edge)
        chosen.append(e)
        sums.add(s)
        prods.add(p)
    return chosen

def _select_edges_spectral(values, all_edges):
    feats = _edge_features(values, all_edges)
    E = len(feats)
    C = np.zeros((E, E), dtype=int)
    for i in range(E):
        s_i, p_i = feats[i][1], feats[i][2]
        for j in range(i+1, E):
            s_j, p_j = feats[j][1], feats[j][2]
            if s_i == s_j or p_i == p_j:
                C[i, j] = 1
                C[j, i] = 1
    w, v = np.linalg.eigh(C)
    top_vec = v[:, -1]
    idxs = np.argsort(top_vec)[::-1][:M]
    selected_edges = [feats[i][0] for i in idxs]
    return selected_edges

def _local_search_edges(values, edges, all_edges):
    edges_set = set(edges)
    all_edges_list = list(all_edges)
    non_edges = [e for e in all_edges_list if e not in edges_set]
    
    cur_score = _raw_score(values, edges)
    improved = True
    # Limit iterations to avoid infinite loops or excessive time
    for _ in range(50):
        if not improved:
            break
        improved = False
        for e_out in edges:
            for e_in in non_edges:
                new_edges = list(edges_set)
                new_edges.remove(e_out)
                new_edges.append(e_in)
                new_score = _raw_score(values, new_edges)
                if new_score < cur_score:
                    edges_set.remove(e_out)
                    edges_set.add(e_in)
                    edges = list(edges_set)
                    non_edges = [e for e in all_edges_list if e not in edges_set]
                    cur_score = new_score
                    improved = True
                    break
            if improved:
                break
    return edges, cur_score

def _score_with_best_edges(values):
    all_e = _all_edges()
    feats = _edge_features(values, all_e)
    
    # Greedy
    edges_g = _select_edges_greedy(feats, M)
    score_g = _raw_score(values, edges_g)
    
    # Local Search on Greedy
    edges_ls, score_ls = _local_search_edges(values, edges_g, all_e)
    
    best_edges = edges_ls
    best_score = score_ls
    
    # Spectral (only if time allows or for better structure)
    # Spectral is slower, so we might skip it in tight loops
    # But let's try it for the initial evaluation
    
    edges_s = _select_edges_spectral(values, all_e)
    score_s = _raw_score(values, edges_s)
    
    # Local Search on Spectral
    edges_ls_s, score_ls_s = _local_search_edges(values, edges_s, all_e)
    
    if score_ls_s < best_score:
        best_score = score_ls_s
        best_edges = edges_ls_s
        
    return best_score, best_edges

def _make_geometric_values(scale=1, base=2):
    vals = []
    v = scale
    for i in range(N):
        if v > MAXV:
            break
        vals.append(v)
        v *= base
    while len(vals) < N:
        vals.append(vals[-1] * 2)
    vals = sorted(set(vals[:N]))
    while len(vals) < N:
        vals.append(vals[-1] + 1)
    return vals[:N]

def _make_arithmetic_values(start=1, step=1):
    return [start + i * step for i in range(N)]

def _make_highly_composite():
    # Highly composite numbers
    return [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 30, 36, 48]

def _random_values():
    return sorted(random.sample(range(1, MAXV + 1), N))

def _anneal_values(initial_vals, all_e, time_budget):
    vals = initial_vals[:]
    score, edges = _score_with_best_edges(vals)
    best_vals = vals[:]
    best_score = score
    best_edges = edges[:]
    
    T = 1000.0
    cooling = 0.995
    start = time.time()
    
    while T > 1.0:
        if time.time() - start > time_budget:
            break
        new_vals = vals[:]
        idx = random.randint(0, N - 1)
        old = new_vals[idx]
        delta = random.choice([-50, -10, -5, -2, -1, 1, 2, 5, 10, 50])
        new_vals[idx] = max(1, min(MAXV, old + delta))
        if new_vals[idx] == old:
            new_vals[idx] = max(1, min(MAXV, old + random.randint(-100, 100)))
        
        if len(set(new_vals)) < N or max(new_vals) > MAXV:
            T *= cooling
            continue
        
        new_score, new_edges = _score_with_best_edges(new_vals)
        delta_s = new_score - score
        if delta_s < 0 or random.random() < pow(2.718281828, -delta_s / max(T, 1e-9)):
            vals = new_vals
            score = new_score
            edges = new_edges
            if score < best_score:
                best_score = score
                best_vals = vals[:]
                best_edges = edges[:]
        T *= cooling
    
    return best_vals, best_score, best_edges

def construct():
    start_time = time.time()
    deadline = start_time + 75
    
    all_e = _all_edges()
    
    best_result = None
    best_score = float('inf')
    
    inits = [
        _make_geometric_values(1, 2),
        _make_arithmetic_values(1, 1),
        _make_arithmetic_values(1, 2),
        _make_geometric_values(1, 3),
        list(range(1, N + 1)),
        [2 * (i + 1) for i in range(N)],
        [2 ** i for i in range(N)],
        [1, 2, 3, 4, 5, 6, 7, 8, 100, 200, 400, 800, 1600, 3200],
        _make_highly_composite()
    ]
    
    for _ in range(5):
        inits.append(_random_values())
        
    for vals in inits:
        if time.time() > deadline:
            break
        if len(set(vals)) < N or max(vals) > MAXV:
            continue
        score, edges = _score_with_best_edges(vals)
        if score < best_score:
            best_score = score
            best_result = {'values': vals[:], 'edges': [list(e) for e in edges]}
            
    if best_result is None:
        best_result = {'values': list(range(1, N + 1)), 'edges': [list(e) for e in all_e[:M]]}
        best_score = _raw_score(best_result['values'], best_result['edges'])
        
    init_scores = []
    for vals in inits[:8]:
        if len(set(vals)) < N or max(vals) > MAXV:
            continue
        s, e = _score_with_best_edges(vals)
        init_scores.append((s, vals))
    init_scores.sort(key=lambda x: x[0])
    
    for s0, v0 in init_scores[:4]:
        if time.time() > deadline - 10:
            break
        remaining = deadline - time.time()
        if remaining < 5:
            break
        budget = min(15.0, remaining - 5)
        if budget < 3:
            continue
        av, ascore, aedges = _anneal_values(v0, all_e, budget)
        if ascore < best_score:
            best_score = ascore
            best_result = {'values': av[:], 'edges': [list(e) for e in aedges]}
            
    if best_result and time.time() < deadline - 5:
        vals = best_result['values'][:]
        edges = [tuple(e) for e in best_result['edges']]
        cur_score = best_score
        remaining = deadline - time.time() - 5
        if remaining > 3:
            for _ in range(int(remaining * 50)):
                if time.time() > deadline - 5:
                    break
                new_vals = vals[:]
                idx = random.randint(0, N - 1)
                new_vals[idx] = max(1, min(MAXV, new_vals[idx] + random.randint(-20, 20)))
                if len(set(new_vals)) < N or max(new_vals) > MAXV:
                    continue
                ns, ne = _score_with_best_edges(new_vals)
                if ns < cur_score:
                    vals = new_vals
                    cur_score = ns
                    edges = ne
        if cur_score < best_score:
            best_score = cur_score
            best_result = {'values': vals[:], 'edges': [list(e) for e in edges]}
            
    if best_result is None:
        vals = list(range(1, N + 1))
        edges = [list(e) for e in all_e[:M]]
        best_result = {'values': vals, 'edges': edges}
        
    vals = best_result['values']
    edges = best_result['edges']
    best_result['values'] = [int(v) for v in vals]
    best_result['edges'] = [[int(a), int(b)] for a, b in edges]
    
    return best_result
