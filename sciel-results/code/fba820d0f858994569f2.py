"""Rewritten program using ILP-style construction with targeted value/edge optimization."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import Counter

P = {'n': 14, 'm': 35, 'maximum': 10000}

def construct():
    deadline = time.time() + 85
    n = P['n']
    m = P['m']
    all_edges = list(combinations(range(n), 2))
    random.seed(42)

    def evaluate(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))

    def compute_edge_maps(values):
        edge_sums = {}
        edge_prods = {}
        for i, j in all_edges:
            edge_sums[(i, j)] = values[i] + values[j]
            edge_prods[(i, j)] = values[i] * values[j]
        return edge_sums, edge_prods

    def greedy_edges(values, edge_sums, edge_prods):
        sum_freq = Counter(edge_sums.values())
        prod_freq = Counter(edge_prods.values())
        sum_count = Counter()
        prod_count = Counter()
        edges = []
        remaining = set(all_edges)
        for _ in range(m):
            best_e = None
            best_key = None
            for e in remaining:
                s = edge_sums[e]
                p = edge_prods[e]
                new_s = 0 if sum_count[s] > 0 else 1
                new_p = 0 if prod_count[p] > 0 else 1
                key = (new_s + new_p, -sum_freq[s], -prod_freq[p])
                if best_key is None or key < best_key:
                    best_key = key
                    best_e = e
            edges.append(list(best_e))
            sum_count[edge_sums[best_e]] += 1
            prod_count[edge_prods[best_e]] += 1
            remaining.discard(best_e)
        return edges

    def local_search(values, edges, max_iters=3000, deadline=None):
        """
        Local search with 2-edge swaps and value swaps.
        """
        edges = [list(e) for e in edges]
        edge_set = set(tuple(e) for e in edges)
        
        sum_count = Counter()
        prod_count = Counter()
        for i, j in edges:
            sum_count[values[i] + values[j]] += 1
            prod_count[values[i] * values[j]] += 1
        
        cur_score = max(len(sum_count), len(prod_count))
        best_score = cur_score
        best_edges = [list(e) for e in edges]
        best_values = list(values)
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            move_type = random.random()
            
            if move_type < 0.6:
                # 2-Edge Swap
                idx1, idx2 = random.sample(range(m), 2)
                e1 = tuple(edges[idx1])
                e2 = tuple(edges[idx2])
                
                found = False
                attempts = 0
                while not found and attempts < 30:
                    ne1 = random.choice(all_edges)
                    ne2 = random.choice(all_edges)
                    if ne1 != ne2 and ne1 not in edge_set and ne2 not in edge_set:
                        found = True
                    attempts += 1
                
                if not found:
                    continue
                
                # Remove e1, e2
                for e in [e1, e2]:
                    s = values[e[0]] + values[e[1]]
                    p = values[e[0]] * values[e[1]]
                    sum_count[s] -= 1
                    if sum_count[s] == 0: del sum_count[s]
                    prod_count[p] -= 1
                    if prod_count[p] == 0: del prod_count[p]
                
                # Add ne1, ne2
                for e in [ne1, ne2]:
                    s = values[e[0]] + values[e[1]]
                    p = values[e[0]] * values[e[1]]
                    sum_count[s] += 1
                    prod_count[p] += 1
                
                new_score = max(len(sum_count), len(prod_count))
                
                if new_score <= cur_score:
                    edges[idx1] = list(ne1)
                    edges[idx2] = list(ne2)
                    edge_set.remove(e1)
                    edge_set.remove(e2)
                    edge_set.add(ne1)
                    edge_set.add(ne2)
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_edges = [list(e) for e in edges]
                        best_values = list(values)
                else:
                    # Undo
                    for e in [ne1, ne2]:
                        s = values[e[0]] + values[e[1]]
                        p = values[e[0]] * values[e[1]]
                        sum_count[s] -= 1
                        if sum_count[s] == 0: del sum_count[s]
                        prod_count[p] -= 1
                        if prod_count[p] == 0: del prod_count[p]
                    
                    for e in [e1, e2]:
                        s = values[e[0]] + values[e[1]]
                        p = values[e[0]] * values[e[1]]
                        sum_count[s] += 1
                        prod_count[p] += 1

            elif move_type < 0.9:
                # Value Swap
                v1, v2 = random.sample(range(n), 2)
                val1, val2 = values[v1], values[v2]
                
                if val1 == val2: continue
                
                # Collect all affected edges
                affected = set()
                for idx, e in enumerate(edges):
                    if v1 in e or v2 in e:
                        affected.add(idx)
                
                old_data = []
                for idx in affected:
                    i, j = edges[idx]
                    s = values[i] + values[j]
                    p = values[i] * values[j]
                    old_data.append((idx, s, p))
                    sum_count[s] -= 1
                    if sum_count[s] == 0: del sum_count[s]
                    prod_count[p] -= 1
                    if prod_count[p] == 0: del prod_count[p]
                
                # Swap
                values[v1], values[v2] = val2, val1
                
                for idx in affected:
                    i, j = edges[idx]
                    new_s = values[i] + values[j]
                    new_p = values[i] * values[j]
                    sum_count[new_s] += 1
                    prod_count[new_p] += 1
                
                new_score = max(len(sum_count), len(prod_count))
                
                if new_score <= cur_score:
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_edges = [list(e) for e in edges]
                        best_values = list(values)
                else:
                    # Undo
                    for idx in affected:
                        i, j = edges[idx]
                        cur_s = values[i] + values[j]
                        cur_p = values[i] * values[j]
                        sum_count[cur_s] -= 1
                        if sum_count[cur_s] == 0: del sum_count[cur_s]
                        prod_count[cur_p] -= 1
                        if prod_count[cur_p] == 0: del prod_count[cur_p]
                    
                    for idx, s, p in old_data:
                        sum_count[s] += 1
                        prod_count[p] += 1
                    
                    values[v1], values[v2] = val1, val2

            else:
                # Value Perturbation
                v_idx = random.randint(0, n - 1)
                old_val = values[v_idx]
                
                nv = random.randint(1, P['maximum'])
                if nv == old_val or nv in values:
                    continue
                
                affected = set()
                for idx, e in enumerate(edges):
                    if v_idx in e:
                        affected.add(idx)
                
                old_data = []
                for idx in affected:
                    i, j = edges[idx]
                    s = values[i] + values[j]
                    p = values[i] * values[j]
                    old_data.append((idx, s, p))
                    sum_count[s] -= 1
                    if sum_count[s] == 0: del sum_count[s]
                    prod_count[p] -= 1
                    if prod_count[p] == 0: del prod_count[p]
                
                values[v_idx] = nv
                
                for idx in affected:
                    i, j = edges[idx]
                    new_s = values[i] + values[j]
                    new_p = values[i] * values[j]
                    sum_count[new_s] += 1
                    prod_count[new_p] += 1
                
                new_score = max(len(sum_count), len(prod_count))
                
                if new_score <= cur_score:
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_edges = [list(e) for e in edges]
                        best_values = list(values)
                else:
                    for idx in affected:
                        i, j = edges[idx]
                        cur_s = values[i] + values[j]
                        cur_p = values[i] * values[j]
                        sum_count[cur_s] -= 1
                        if sum_count[cur_s] == 0: del sum_count[cur_s]
                        prod_count[cur_p] -= 1
                        if prod_count[cur_p] == 0: del prod_count[cur_p]
                    
                    for idx, s, p in old_data:
                        sum_count[s] += 1
                        prod_count[p] += 1
                    
                    values[v_idx] = old_val

        return best_values, best_edges, best_score

    def perturb(values, edges):
        """Strong perturbation for ILS"""
        vals = list(values)
        eds = [list(e) for e in edges]
        
        for _ in range(random.randint(2, 3)):
            i, j = random.sample(range(n), 2)
            vals[i], vals[j] = vals[j], vals[i]
            
        for _ in range(random.randint(2, 4)):
            idx1, idx2 = random.sample(range(m), 2)
            e1 = tuple(eds[idx1])
            e2 = tuple(eds[idx2])
            edge_set = set(tuple(e) for e in eds)
            ne1, ne2 = None, None
            for _ in range(100):
                c1 = random.choice(all_edges)
                c2 = random.choice(all_edges)
                if c1 != c2 and c1 not in edge_set and c2 not in edge_set:
                    ne1, ne2 = c1, c2
                    break
            if ne1:
                eds[idx1] = list(ne1)
                eds[idx2] = list(ne2)
                
        return vals, eds

    best_overall_score = float('inf')
    best_overall_values = None
    best_overall_edges = None

    # Start with structured initializations
    structures = [
        list(range(1, 15)),
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84],
        [4, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52],
    ]

    # Add random starts
    for _ in range(4):
        vals = [random.randint(1, 10000) for _ in range(n)]
        while len(set(vals)) < n:
            idx = random.randint(0, n-1)
            vals[idx] = random.randint(1, 10000)
        structures.append(vals)

    for init_vals in structures:
        if time.time() > deadline - 5:
            break
        if len(set(init_vals)) != n or any(v < 1 or v > P['maximum'] for v in init_vals):
            continue
            
        edge_sums, edge_prods = compute_edge_maps(init_vals)
        edges = greedy_edges(init_vals, edge_sums, edge_prods)
        
        cur_vals, cur_edges, cur_score = local_search(init_vals, edges, max_iters=2000, deadline=deadline - 5)
        
        if cur_score < best_overall_score:
            best_overall_score = cur_score
            best_overall_values = cur_vals
            best_overall_edges = cur_edges
            
        if time.time() > deadline - 10:
            break
            
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            p_vals, p_edges = perturb(cur_vals, cur_edges)
            if len(set(p_vals)) != n: continue
            nv, ne, ns = local_search(p_vals, p_edges, max_iters=1000, deadline=deadline - 5)
            if ns < best_overall_score:
                best_overall_score = ns
                best_overall_values = nv
                best_overall_edges = ne
                cur_vals, cur_edges, cur_score = nv, ne, ns

    if best_overall_values is None:
        best_overall_values = list(range(1, 15))
        edge_sums, edge_prods = compute_edge_maps(best_overall_values)
        best_overall_edges = greedy_edges(best_overall_values, edge_sums, edge_prods)
        best_overall_score = evaluate(best_overall_values, best_overall_edges)

    final_values = [int(v) for v in best_overall_values]
    final_edges = [[int(i), int(j)] for i, j in best_overall_edges]

    assert len(final_values) == n
    assert len(set(final_values)) == n
    assert all(isinstance(v, int) and 1 <= v <= P['maximum'] for v in final_values)
    assert len(final_edges) == m
    edge_tuples = set()
    for e in final_edges:
        i, j = e
        assert 0 <= i < n and 0 <= j < n and i != j
        t = tuple(sorted((i, j)))
        assert t not in edge_tuples
        edge_tuples.add(t)

    return {'values': final_values, 'edges': final_edges}

# EVOLVE-BLOCK-END
