import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    TOTAL_BUDGET = 145

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        adj_mask = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        return adj_mask

    def solve_instance(n, p, seed, time_budget):
        adj_mask_orig = get_graph(n, p, seed)
        
        if not any(adj_mask_orig):
            return []
        
        def iter_bits(x):
            while x:
                lsb = x & -x
                yield lsb.bit_length() - 1
                x ^= lsb

        def find_best_edge(adj, n, rng=None, mode='min'):
            """
            Find the best edge to start a clique from.
            mode: 'min' (min common neighbors), 'max' (max common neighbors), 
                  'min_deg' (min sum of degrees), 'max_deg' (max sum of degrees)
            """
            best_u, best_v = -1, -1
            best_score = float('inf') if mode in ('min', 'min_deg') else -1
            
            for u in range(n):
                if adj[u] == 0:
                    continue
                for v in iter_bits(adj[u]):
                    if v > u:
                        common = (adj[u] & adj[v]).bit_count()
                        deg_u = adj[u].bit_count()
                        deg_v = adj[v].bit_count()
                        
                        if mode == 'min':
                            score = common
                        elif mode == 'max':
                            score = -common
                        elif mode == 'min_deg':
                            score = deg_u + deg_v
                        elif mode == 'max_deg':
                            score = -(deg_u + deg_v)
                        else:
                            score = common
                        
                        if (mode in ('min', 'min_deg') and score < best_score) or \
                           (mode in ('max', 'max_deg') and score > best_score):
                            best_score = score
                            best_u, best_v = u, v
            
            # If rng is provided, we might want to pick from a set of good candidates
            # For now, let's stick to the deterministic best for speed, 
            # but allow randomization in the selection if we implement a candidate pool.
            # The current program uses a candidate pool in some modes. Let's implement a simple one.
            if rng is not None and mode in ('min', 'max'):
                # Collect top 10% candidates
                candidates = []
                threshold = best_score
                for u in range(n):
                    if adj[u] == 0:
                        continue
                    for v in iter_bits(adj[u]):
                        if v > u:
                            common = (adj[u] & adj[v]).bit_count()
                            deg_u = adj[u].bit_count()
                            deg_v = adj[v].bit_count()
                            if mode == 'min':
                                score = common
                            elif mode == 'max':
                                score = -common
                            else:
                                score = 0
                            
                            # Only consider candidates close to the best
                            if mode == 'min' and score <= threshold + 2:
                                candidates.append((u, v))
                            elif mode == 'max' and score >= threshold - 2:
                                candidates.append((u, v))
                
                if candidates:
                    u, v = rng.choice(candidates)
                    return u, v
            
            return best_u, best_v

        def expand_clique_topk(seed_u, seed_v, adj, rng=None, mode='greedy', top_k=3):
            """
            Expand an edge to a clique.
            Uses a top-k strategy for candidate selection to avoid local minima.
            """
            C = [seed_u, seed_v]
            U = adj[seed_u] & adj[seed_v]
            U &= ~(1 << seed_u) & ~(1 << seed_v)
            
            while U:
                candidates = list(iter_bits(U))
                if not candidates:
                    break
                
                if rng is not None and len(candidates) > 1:
                    rng.shuffle(candidates)
                
                # Calculate scores for candidates
                scored_candidates = []
                for u in candidates:
                    if mode == 'greedy':
                        # Maximize common neighbors within U
                        score = (adj[u] & U).bit_count()
                        score2 = adj[u].bit_count()
                    elif mode == 'degree':
                        score = adj[u].bit_count()
                        score2 = (adj[u] & U).bit_count()
                    else:
                        score = (adj[u] & U).bit_count()
                        score2 = adj[u].bit_count()
                    
                    scored_candidates.append((score, score2, u))
                
                # Sort by primary score descending, then secondary
                scored_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
                
                # Try top K candidates
                best_u = -1
                best_clique_size = -1
                
                k = min(top_k, len(scored_candidates))
                for i in range(k):
                    score, score2, u = scored_candidates[i]
                    # Simulate adding u
                    new_U = U & adj[u]
                    # Estimate final clique size by simple greedy expansion count
                    # Or just use the score as a proxy
                    # A better estimate: count how many more vertices can be added
                    # For speed, we just pick the one with the highest immediate score
                    # The "top-k" insight suggests checking if the expansion leads to a larger clique.
                    # Let's do a quick check: if we add u, what is the new U?
                    # We'll just pick the best score, as full simulation is expensive.
                    if score > best_clique_size:
                        best_clique_size = score
                        best_u = u
                
                if best_u == -1:
                    break
                
                C.append(best_u)
                U &= adj[best_u]
                U &= ~(1 << best_u)
                
            return C

        def remove_clique_edges(C, adj):
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v = C[i], C[j]
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

        def is_clique_union(C1, C2, adj):
            for u in C1:
                for v in C2:
                    if not (adj[u] & (1 << v)):
                        return False
            return True

        def try_merge(cliques, adj, n, deadline):
            changed = True
            while changed and (time.time() < deadline):
                changed = False
                merged_indices = set()
                new_cliques = []
                i = 0
                while i < len(cliques):
                    if i in merged_indices:
                        i += 1
                        continue
                    C_i = cliques[i]
                    merged = False
                    for j in range(i + 1, len(cliques)):
                        if j in merged_indices:
                            continue
                        C_j = cliques[j]
                        if is_clique_union(C_i, C_j, adj):
                            new_cliques.append(C_i + C_j)
                            merged_indices.add(j)
                            changed = True
                            merged = True
                            break
                    if not merged:
                        new_cliques.append(C_i)
                    i += 1
                cliques = new_cliques
            return cliques

        def greedy_partition(adj_mask_orig, n, rng=None, deadline=None, mode='min'):
            if deadline is None:
                deadline = time.time() + 10
            adj = list(adj_mask_orig)
            cliques = []
            
            while True:
                if time.time() > deadline:
                    break
                u, v = find_best_edge(adj, n, rng, mode)
                if u == -1:
                    break
                C = expand_clique_topk(u, v, adj, rng, mode='greedy', top_k=3)
                cliques.append(C)
                remove_clique_edges(C, adj)
            
            cliques = try_merge(cliques, adj, n, deadline)
            return cliques

        def validate(cliques, adj_mask_orig, n):
            for C in cliques:
                if len(C) < 2:
                    return False
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v = C[i], C[j]
                        if not (adj_mask_orig[u] & (1 << v)):
                            return False
            edge_set = set()
            for C in cliques:
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v = C[i], C[j]
                        key = (min(u, v), max(u, v))
                        if key in edge_set:
                            return False
                        edge_set.add(key)
            for u in range(n):
                for v in iter_bits(adj_mask_orig[u]):
                    if v > u:
                        if (u, v) not in edge_set:
                            return False
            return True

        def local_search(cliques, adj_mask_orig, n, rng, deadline):
            if len(cliques) < 2:
                return cliques
            
            best = cliques[:]
            best_count = len(best)
            
            changed = True
            while changed and (time.time() < deadline):
                changed = False
                merged_indices = set()
                new_cliques = []
                i = 0
                while i < len(best):
                    if i in merged_indices:
                        i += 1
                        continue
                    C_i = best[i]
                    merged = False
                    for j in range(i + 1, len(best)):
                        if j in merged_indices:
                            continue
                        C_j = best[j]
                        if is_clique_union(C_i, C_j, adj_mask_orig):
                            new_cliques.append(C_i + C_j)
                            merged_indices.add(j)
                            changed = True
                            merged = True
                            break
                    if not merged:
                        new_cliques.append(C_i)
                    i += 1
                best = new_cliques
                best_count = len(best)
            
            # Vertex moving heuristic
            for _ in range(10):
                if time.time() > deadline:
                    break
                improved = False
                for idx in range(len(best)):
                    if time.time() > deadline:
                        break
                    C = best[idx]
                    if len(C) < 3:
                        continue
                    for remove_idx in range(len(C)):
                        if time.time() > deadline:
                            break
                        v = C[remove_idx]
                        C_remaining = [x for x in C if x != v]
                        if len(C_remaining) < 2:
                            continue
                        
                        valid = True
                        for a in range(len(C_remaining)):
                            for b in range(a + 1, len(C_remaining)):
                                u, w = C_remaining[a], C_remaining[b]
                                if not (adj_mask_orig[u] & (1 << w)):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        
                        for other_idx in range(len(best)):
                            if other_idx == idx:
                                continue
                            C2 = best[other_idx]
                            if is_clique_union(C_remaining, C2, adj_mask_orig):
                                new_C = C_remaining + C2
                                new_cliques = []
                                for k in range(len(best)):
                                    if k == idx:
                                        if v not in new_C:
                                            new_cliques.append([v])
                                    elif k == other_idx:
                                        pass
                                    else:
                                        new_cliques.append(best[k])
                                new_cliques.append(new_C)
                                if validate(new_cliques, adj_mask_orig, n):
                                    if len(new_cliques) < best_count:
                                        best = new_cliques
                                        best_count = len(best)
                                        improved = True
                                        break
                            else:
                                can_add = True
                                for u in C2:
                                    if not (adj_mask_orig[v] & (1 << u)):
                                        can_add = False
                                        break
                                if can_add:
                                    new_C2 = C2 + [v]
                                    new_cliques = []
                                    for k in range(len(best)):
                                        if k == idx:
                                            new_cliques.append(C_remaining)
                                        elif k == other_idx:
                                            new_cliques.append(new_C2)
                                        else:
                                            new_cliques.append(best[k])
                                    if validate(new_cliques, adj_mask_orig, n):
                                        if len(new_cliques) < best_count:
                                            best = new_cliques
                                            best_count = len(best)
                                            improved = True
                                            break
                        if improved:
                            break
                    if improved:
                        break
                if not improved:
                    break
            
            return best

        best_cliques = None
        best_count = float('inf')
        
        # Deterministic runs
        det_deadline = start_time + time_budget * 0.15
        det_cliques = greedy_partition(adj_mask_orig, n, rng=None, deadline=det_deadline, mode='min')
        if validate(det_cliques, adj_mask_orig, n):
            best_cliques = det_cliques
            best_count = len(det_cliques)
        
        det_deadline2 = start_time + time_budget * 0.30
        det_cliques2 = greedy_partition(adj_mask_orig, n, rng=None, deadline=det_deadline2, mode='max')
        if validate(det_cliques2, adj_mask_orig, n):
            if len(det_cliques2) < best_count:
                best_count = len(det_cliques2)
                best_cliques = det_cliques2

        # Randomized restarts
        remaining = time_budget - (time.time() - start_time)
        if remaining > 3:
            num_restarts = max(8, int(remaining / 0.8))
            rng_master = random.Random(seed + 1000)
            modes = ['min', 'max', 'min_deg', 'max_deg']
            
            for r in range(num_restarts):
                if time.time() > start_time + time_budget - 2.0:
                    break
                rng = random.Random(rng_master.randint(0, 2**31))
                mode = modes[r % len(modes)]
                restart_deadline = min(start_time + time_budget - 2.0, time.time() + max(0.2, remaining / (num_restarts - r)))
                try:
                    cliques = greedy_partition(adj_mask_orig, n, rng=rng, deadline=restart_deadline, mode=mode)
                    if validate(cliques, adj_mask_orig, n):
                        ls_deadline = min(start_time + time_budget - 2.0, time.time() + 1.0)
                        improved = local_search(cliques, adj_mask_orig, n, rng, ls_deadline)
                        if validate(improved, adj_mask_orig, n):
                            if len(improved) < best_count:
                                best_count = len(improved)
                                best_cliques = improved
                        else:
                            if len(cliques) < best_count:
                                best_count = len(cliques)
                                best_cliques = cliques
                except:
                    continue
        
        if best_cliques is None:
            best_cliques = []
            for u in range(n):
                for v in iter_bits(adj_mask_orig[u]):
                    if v > u:
                        best_cliques.append([u, v])
        
        return best_cliques

    results = []
    # Instance 1
    t1 = min(50, TOTAL_BUDGET - 65)
    results.append(solve_instance(32, 0.6, 601, t1))
    
    # Instance 2
    elapsed = time.time() - start_time
    t2 = max(10, TOTAL_BUDGET - elapsed - 3)
    results.append(solve_instance(64, 0.55, 602, t2))
    
    return results
