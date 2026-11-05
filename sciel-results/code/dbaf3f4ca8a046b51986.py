import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    n = 16
    
    # Precompute all 8-vertex subsets and pair-to-subset mapping
    all_subsets = list(combinations(range(n), 8))
    pair_to_subsets = {}
    for u in range(n):
        for v in range(u + 1, n):
            pair_to_subsets[(u, v)] = []
    for idx, subset in enumerate(all_subsets):
        subset_set = set(subset)
        for u in subset:
            for v in subset:
                if u < v:
                    pair_to_subsets[(u, v)].append(idx)
    
    def is_q3_in_subset(edge_set, subset):
        subset_set = set(subset)
        adj = {v: set() for v in subset}
        edge_count = 0
        for u, v in edge_set:
            if u in subset_set and v in subset_set:
                edge_count += 1
                adj[u].add(v)
                adj[v].add(u)
        if edge_count < 12:
            return False
        A_list = list(subset)
        for A in combinations(A_list, 4):
            A_set = set(A)
            B = tuple(v for v in A_list if v not in A_set)
            B_set = set(B)
            ok = True
            for v in A:
                if adj[v] & A_set:
                    ok = False
                    break
            if not ok:
                continue
            for v in B:
                if adj[v] & B_set:
                    ok = False
                    break
            if not ok:
                continue
            for v in A:
                if len(adj[v]) != 3:
                    ok = False
                    break
            if not ok:
                continue
            for v in B:
                if len(adj[v]) != 3:
                    ok = False
                    break
            if not ok:
                continue
            cycle_count = 0
            for i in range(4):
                for j in range(i + 1, 4):
                    cycle_count += len(adj[A[i]] & adj[A[j]])
            if cycle_count == 6:
                return True
        return False
    
    # Start with K3,13
    edges = set()
    for a in range(3):
        for b in range(3, 16):
            edges.add((a, b))
    
    subset_edge_counts = [0] * len(all_subsets)
    for idx, subset in enumerate(all_subsets):
        subset_set = set(subset)
        for u, v in edges:
            if u in subset_set and v in subset_set:
                subset_edge_counts[idx] += 1
    
    best_edges = edges.copy()
    best_count = len(edges)
    
    def try_add(edge_set, new_edge, counts):
        u, v = new_edge
        if (u, v) in edge_set or (v, u) in edge_set:
            return True
        for idx in pair_to_subsets[(u, v)]:
            if counts[idx] >= 12:
                if is_q3_in_subset(edge_set | {new_edge}, all_subsets[idx]):
                    return False
        return True
    
    # Greedy
    changed = True
    while changed and time.time() - start_time < DEADLINE - 20:
        changed = False
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in edges:
                    if try_add(edges, (u, v), subset_edge_counts):
                        edges.add((u, v))
                        for idx in pair_to_subsets[(u, v)]:
                            subset_edge_counts[idx] += 1
                        changed = True
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges.copy()
    
    # Local search with kicks
    random.seed(42)
    
    def local_search(es, counts):
        improved = True
        while improved and time.time() - start_time < DEADLINE - 5:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) not in es:
                        if try_add(es, (u, v), counts):
                            es.add((u, v))
                            for idx in pair_to_subsets[(u, v)]:
                                counts[idx] += 1
                            improved = True
        return es, counts
    
    for it in range(40):
        if time.time() - start_time > DEADLINE:
            break
        cur = edges.copy()
        cc = subset_edge_counts.copy()
        cur, cc = local_search(cur, cc)
        if len(cur) > best_count:
            best_count = len(cur)
            best_edges = cur.copy()
        
        kicked = cur.copy()
        kc = cc.copy()
        nr = random.randint(3, 8)
        el = list(kicked)
        random.shuffle(el)
        for e in el[:nr]:
            kicked.discard(e)
            for idx in pair_to_subsets[e]:
                kc[idx] -= 1
        kicked, kc = local_search(kicked, kc)
        if len(kicked) > best_count:
            best_count = len(kicked)
            best_edges = kicked.copy()
        if len(kicked) >= len(cur):
            edges = kicked
            subset_edge_counts = kc
    
    return [[u, v] for u, v in best_edges if u < v]
