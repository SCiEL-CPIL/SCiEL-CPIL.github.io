import time
import random
import math
import sys
from collections import defaultdict

def construct():
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def bron_kerbosch(adj, R, P, X):
        if not P and not X:
            if len(R) >= 2:
                yield tuple(sorted(R))
            return
        if P or X:
            u = max(P | X, key=lambda v: len(P & adj[v]))
            candidates = P - adj[u]
        else:
            candidates = set()
        for v in candidates:
            yield from bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)

    def get_maximal_cliques(adj):
        n = len(adj)
        return list(bron_kerbosch(adj, set(), set(range(n)), set()))

    def solve_instance(n, p, seed, time_budget):
        edges, adj = random_graph(n, p, seed)
        cliques = get_maximal_cliques(adj)
        
        if not cliques:
            return []

        cliques_list = cliques
        m = len(cliques_list)
        
        # Precompute vertex to cliques mapping
        C = [set() for _ in range(n)]
        for i, c in enumerate(cliques_list):
            for v in c:
                C[v].add(i)
        
        # Also precompute clique to vertices for quick access
        clique_vertices = [set(c) for c in cliques_list]

        def greedy_init(rng):
            uncovered = set(range(m))
            selected = set()
            
            while uncovered:
                best_v = -1
                best_count = -1
                for v in range(n):
                    if not C[v]:
                        continue
                    count = len(uncovered & C[v])
                    if count > best_count:
                        best_count = count
                        best_v = v
                    elif count == best_count and count > 0:
                        if rng.random() < 0.5:
                            best_v = v
                
                if best_v == -1:
                    break
                    
                selected.add(best_v)
                uncovered -= C[best_v]
                
            return selected

        def local_search(S, rng, time_limit):
            S = set(S)
            cover_count = [0] * m
            for v in S:
                for c in C[v]:
                    cover_count[c] += 1
            
            improved = True
            start = time.time()
            
            while improved and (time.time() - start) < time_limit:
                improved = False
                
                # 1. Try Removals
                for v in list(S):
                    can_remove = True
                    for c in C[v]:
                        if cover_count[c] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for c in C[v]:
                            cover_count[c] -= 1
                        S.remove(v)
                        improved = True
                        break
                
                if improved:
                    continue
                    
                # 2. Try 1-for-1 Swaps
                found_swap = False
                for v_out in list(S):
                    critical = [c for c in C[v_out] if cover_count[c] == 1]
                    if not critical:
                        continue
                    
                    best_v_in = -1
                    best_score = -1
                    
                    for v_in in range(n):
                        if v_in in S:
                            continue
                        valid = True
                        for c in critical:
                            if c not in C[v_in]:
                                valid = False
                                break
                        if valid:
                            score = len(C[v_in])
                            if score > best_score:
                                best_score = score
                                best_v_in = v_in
                                if score == m:
                                    break
                    
                    if best_v_in != -1:
                        for c in C[v_out]:
                            cover_count[c] -= 1
                        for c in C[best_v_in]:
                            cover_count[c] += 1
                        
                        S.remove(v_out)
                        S.add(best_v_in)
                        found_swap = True
                        break
                
                if found_swap:
                    improved = True
                    continue
                    
                # 3. Try 2-for-1 swaps: remove two vertices, add one
                found_2swap = False
                S_list = list(S)
                for i in range(len(S_list)):
                    if (time.time() - start) >= time_limit:
                        break
                    for j in range(i + 1, len(S_list)):
                        v1 = S_list[i]
                        v2 = S_list[j]
                        critical_cliques = set()
                        for c in C[v1]:
                            if cover_count[c] == 1:
                                critical_cliques.add(c)
                            elif cover_count[c] == 2 and c in C[v2]:
                                critical_cliques.add(c)
                        for c in C[v2]:
                            if cover_count[c] == 1 and c not in C[v1]:
                                critical_cliques.add(c)
                        
                        if not critical_cliques:
                            for c in C[v1]:
                                cover_count[c] -= 1
                            for c in C[v2]:
                                cover_count[c] -= 1
                            S.remove(v1)
                            S.remove(v2)
                            found_2swap = True
                            break
                        
                        best_v_in = -1
                        best_score = -1
                        for v_in in range(n):
                            if v_in in S:
                                continue
                            valid = True
                            for c in critical_cliques:
                                if c not in C[v_in]:
                                    valid = False
                                    break
                            if valid:
                                score = len(C[v_in])
                                if score > best_score:
                                    best_score = score
                                    best_v_in = v_in
                                    if score == m:
                                        break
                        
                        if best_v_in != -1:
                            for c in C[v1]:
                                cover_count[c] -= 1
                            for c in C[v2]:
                                cover_count[c] -= 1
                            for c in C[best_v_in]:
                                cover_count[c] += 1
                            
                            S.remove(v1)
                            S.remove(v2)
                            S.add(best_v_in)
                            found_2swap = True
                            break
                    
                    if found_2swap:
                        break
                
                if found_2swap:
                    improved = True

            return S

        def final_polish(S, time_limit):
            """High-precision polish: exhaustive removal, 1-for-1, 2-for-1, and 3-for-2 swaps."""
            S = set(S)
            cover_count = [0] * m
            for v in S:
                for c in C[v]:
                    cover_count[c] += 1
            
            start = time.time()
            improved = True
            
            while improved and (time.time() - start) < time_limit:
                improved = False
                
                # Pass 1: Removals
                for v in list(S):
                    can_remove = True
                    for c in C[v]:
                        if cover_count[c] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for c in C[v]:
                            cover_count[c] -= 1
                        S.remove(v)
                        improved = True
                
                if improved:
                    continue
                
                # Pass 2: 1-for-1 swaps
                for v_out in list(S):
                    critical = [c for c in C[v_out] if cover_count[c] == 1]
                    if not critical:
                        continue
                    best_v_in = -1
                    best_score = -1
                    for v_in in range(n):
                        if v_in in S:
                            continue
                        valid = True
                        for c in critical:
                            if c not in C[v_in]:
                                valid = False
                                break
                        if valid:
                            score = len(C[v_in])
                            if score > best_score:
                                best_score = score
                                best_v_in = v_in
                    if best_v_in != -1:
                        for c in C[v_out]:
                            cover_count[c] -= 1
                        for c in C[best_v_in]:
                            cover_count[c] += 1
                        S.remove(v_out)
                        S.add(best_v_in)
                        improved = True
                
                if improved:
                    continue
                
                # Pass 3: 2-for-1 swaps
                S_list = list(S)
                for i in range(len(S_list)):
                    if (time.time() - start) >= time_limit:
                        break
                    for j in range(i + 1, len(S_list)):
                        v1, v2 = S_list[i], S_list[j]
                        critical_cliques = set()
                        for c in C[v1]:
                            if cover_count[c] == 1:
                                critical_cliques.add(c)
                            elif cover_count[c] == 2 and c in C[v2]:
                                critical_cliques.add(c)
                        for c in C[v2]:
                            if cover_count[c] == 1 and c not in C[v1]:
                                critical_cliques.add(c)
                        if not critical_cliques:
                            for c in C[v1]:
                                cover_count[c] -= 1
                            for c in C[v2]:
                                cover_count[c] -= 1
                            S.remove(v1)
                            S.remove(v2)
                            improved = True
                            break
                        best_v_in = -1
                        best_score = -1
                        for v_in in range(n):
                            if v_in in S:
                                continue
                            valid = True
                            for c in critical_cliques:
                                if c not in C[v_in]:
                                    valid = False
                                    break
                            if valid:
                                score = len(C[v_in])
                                if score > best_score:
                                    best_score = score
                                    best_v_in = v_in
                        if best_v_in != -1:
                            for c in C[v1]:
                                cover_count[c] -= 1
                            for c in C[v2]:
                                cover_count[c] -= 1
                            for c in C[best_v_in]:
                                cover_count[c] += 1
                            S.remove(v1)
                            S.remove(v2)
                            S.add(best_v_in)
                            improved = True
                            break
                    if improved:
                        break
                
                if improved:
                    continue
                
                # Pass 4: 3-for-2 swaps (remove 3, add 2)
                S_list = list(S)
                for i in range(len(S_list)):
                    if (time.time() - start) >= time_limit:
                        break
                    for j in range(i + 1, len(S_list)):
                        if (time.time() - start) >= time_limit:
                            break
                        for k in range(j + 1, len(S_list)):
                            if (time.time() - start) >= time_limit:
                                break
                            v1, v2, v3 = S_list[i], S_list[j], S_list[k]
                            # Compute critical cliques after removing v1, v2, v3
                            critical_cliques = set()
                            for c in C[v1]:
                                if cover_count[c] == 1:
                                    critical_cliques.add(c)
                                elif cover_count[c] == 2:
                                    if c in C[v2] or c in C[v3]:
                                        critical_cliques.add(c)
                                elif cover_count[c] == 3:
                                    if c in C[v2] and c in C[v3]:
                                        critical_cliques.add(c)
                            for c in C[v2]:
                                if cover_count[c] == 1 and c not in C[v1]:
                                    critical_cliques.add(c)
                                elif cover_count[c] == 2 and c in C[v3]:
                                    critical_cliques.add(c)
                            for c in C[v3]:
                                if cover_count[c] == 1 and c not in C[v1] and c not in C[v2]:
                                    critical_cliques.add(c)
                            
                            if not critical_cliques:
                                for c in C[v1]:
                                    cover_count[c] -= 1
                                for c in C[v2]:
                                    cover_count[c] -= 1
                                for c in C[v3]:
                                    cover_count[c] -= 1
                                S.remove(v1)
                                S.remove(v2)
                                S.remove(v3)
                                improved = True
                                break
                            
                            # Try to find 2 vertices that cover all critical cliques
                            best_pair = None
                            best_score = -1
                            for a in range(n):
                                if a in S:
                                    continue
                                for b in range(a + 1, n):
                                    if b in S:
                                        continue
                                    valid = True
                                    for c in critical_cliques:
                                        if c not in C[a] and c not in C[b]:
                                            valid = False
                                            break
                                    if valid:
                                        score = len(C[a]) + len(C[b])
                                        if score > best_score:
                                            best_score = score
                                            best_pair = (a, b)
                            
                            if best_pair:
                                a, b = best_pair
                                for c in C[v1]:
                                    cover_count[c] -= 1
                                for c in C[v2]:
                                    cover_count[c] -= 1
                                for c in C[v3]:
                                    cover_count[c] -= 1
                                for c in C[a]:
                                    cover_count[c] += 1
                                for c in C[b]:
                                    cover_count[c] += 1
                                S.remove(v1)
                                S.remove(v2)
                                S.remove(v3)
                                S.add(a)
                                S.add(b)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break

            return S

        best_set = set(range(n))
        start_time = time.time()
        
        iter_count = 0
        while (time.time() - start_time) < time_budget:
            iter_seed = 1000 + iter_count
            rng = random.Random(iter_seed)
            
            if iter_count % 3 == 0:
                candidate = greedy_init(rng)
            else:
                candidate = greedy_init(rng)
                if len(candidate) > 2:
                    to_remove = rng.sample(sorted(candidate), min(2, len(candidate) // 2))
                    for v in to_remove:
                        candidate.discard(v)
                    uncovered = set()
                    for c in range(m):
                        covered = False
                        for v in candidate:
                            if c in C[v]:
                                covered = True
                                break
                        if not covered:
                            uncovered.add(c)
                    while uncovered:
                        best_v = -1
                        best_count = -1
                        for v in range(n):
                            if v in candidate or not C[v]:
                                continue
                            count = len(uncovered & C[v])
                            if count > best_count:
                                best_count = count
                                best_v = v
                        if best_v == -1:
                            break
                        candidate.add(best_v)
                        uncovered -= C[best_v]
            
            remaining_ls = time_budget - (time.time() - start_time)
            ls_time = min(0.15, remaining_ls * 0.4)
            if ls_time > 0.01:
                candidate = local_search(candidate, rng, ls_time)
            
            if len(candidate) < len(best_set):
                best_set = candidate
                
            iter_count += 1
        
        # Final high-precision polish
        remaining_polish = time_budget - (time.time() - start_time)
        if remaining_polish > 0.05:
            polished = final_polish(best_set, remaining_polish)
            if len(polished) < len(best_set):
                best_set = polished
        
        return sorted(best_set)

    out = []
    start_time = time.time()
    total_deadline = 115.0

    instances = ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))
    
    for idx, (n, p, seed) in enumerate(instances):
        remaining = total_deadline - (time.time() - start_time)
        if remaining <= 0:
            out.append(sorted(range(n)))
            continue
        
        inst_budget = remaining / (len(instances) - idx)
        
        res = solve_instance(n, p, seed, inst_budget)
        out.append(res)
        
    return out
