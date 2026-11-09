import time
import random

def construct():
    start_time = time.time()
    TOTAL_BUDGET = 145  # safety margin from 150s
    
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

        def find_edge(adj, n, rng=None, mode='min'):
            best_u, best_v = -1, -1
            best_score = float('inf') if mode == 'min' else -1
            edges = []
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
                        edges.append((score, u, v))
            
            if not edges:
                return -1, -1
            
            if rng is not None:
                edges.sort(key=lambda x: x[0])
                # Pick from top 10% or so
                k = min(len(edges), max(1, len(edges) // 10))
                candidates = edges[:k]
                idx = rng.randint(0, len(candidates) - 1)
                _, best_u, best_v = candidates[idx]
            else:
                for score, u, v in edges:
                    if (mode in ['min', 'min_deg'] and score < best_score) or \
                       (mode in ['max', 'max_deg'] and score > best_score):
                        best_score = score
                        best_u, best_v = u, v
            
            return best_u, best_v

        def expand_clique(seed_u, seed_v, adj, rng=None, mode='greedy'):
            C = [seed_u, seed_v]
            U = adj[seed_u] & adj[seed_v]
            
            while U:
                candidates = list(iter_bits(U))
                if rng is not None and len(candidates) > 1:
                    rng.shuffle(candidates)
                
                best_u = -1
                best_score = -1
                best_score2 = -1
                
                for u in candidates:
                    if mode == 'greedy':
                        score = (adj[u] & U).bit_count()
                        score2 = adj[u].bit_count()
                    elif mode == 'degree':
                        score = adj[u].bit_count()
                        score2 = (adj[u] & U).bit_count()
                    else:
                        score = (adj[u] & U).bit_count()
                        score2 = adj[u].bit_count()
                    
                    if score > best_score or (score == best_score and score2 > best_score2):
                        best_score = score
                        best_score2 = score2
                        best_u = u
                
                if best_u == -1:
                    break
                    
                C.append(best_u)
                U &= adj[best_u]
                
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
                new_cliques = []
                i = 0
                while i < len(cliques):
                    C_i = cliques[i]
                    merged = False
                    for j in range(i + 1, len(cliques)):
                        C_j = cliques[j]
                        if is_clique_union(C_i, C_j, adj):
                            new_cliques = [C_i + C_j] + [cliques[k] for k in range(i + 1, len(cliques)) if k != j]
                            i = 0
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
                u, v = find_edge(adj, n, rng, mode)
                if u == -1:
                    break
                C = expand_clique(u, v, adj, rng, mode='greedy')
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
            
            for _ in range(30):
                if time.time() > deadline:
                    break
                
                idx = rng.randint(0, len(best) - 1)
                C = best[idx]
                if len(C) < 3:
                    continue
                
                for other_idx in range(len(best)):
                    if other_idx == idx:
                        continue
                    C2 = best[other_idx]
                    if is_clique_union(C, C2, adj_mask_orig):
                        new_cliques = [C + C2] + [best[k] for k in range(len(best)) if k != idx and k != other_idx]
                        if validate(new_cliques, adj_mask_orig, n):
                            if len(new_cliques) < best_count:
                                best = new_cliques
                                best_count = len(new_cliques)
            return best

        best_cliques = None
        best_count = float('inf')
        
        # Deterministic run first with min-common-neighbor strategy
        det_deadline = start_time + time_budget * 0.15
        det_cliques = greedy_partition(adj_mask_orig, n, rng=None, deadline=det_deadline, mode='min')
        if validate(det_cliques, adj_mask_orig, n):
            best_cliques = det_cliques
            best_count = len(det_cliques)
        
        # Try max-common-neighbor strategy
        det_deadline2 = start_time + time_budget * 0.25
        det_cliques2 = greedy_partition(adj_mask_orig, n, rng=None, deadline=det_deadline2, mode='max')
        if validate(det_cliques2, adj_mask_orig, n):
            if len(det_cliques2) < best_count:
                best_count = len(det_cliques2)
                best_cliques = det_cliques2
        
        # Randomized restarts with different modes
        remaining = time_budget - (time.time() - start_time)
        if remaining > 3:
            num_restarts = max(4, int(remaining / 1.0))
            rng_master = random.Random(seed + 1000)
            modes = ['min', 'max', 'min_deg', 'max_deg']
            
            for r in range(num_restarts):
                if time.time() > start_time + time_budget - 1.5:
                    break
                rng = random.Random(rng_master.randint(0, 2**31))
                mode = modes[r % len(modes)]
                restart_deadline = min(start_time + time_budget - 1.5, time.time() + max(0.3, remaining / (num_restarts - r)))
                try:
                    cliques = greedy_partition(adj_mask_orig, n, rng=rng, deadline=restart_deadline, mode=mode)
                    if validate(cliques, adj_mask_orig, n):
                        ls_deadline = min(start_time + time_budget - 1.5, time.time() + 0.5)
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
    t1 = min(45, TOTAL_BUDGET - 60)
    results.append(solve_instance(32, 0.6, 601, t1))
    
    # Instance 2
    elapsed = time.time() - start_time
    t2 = max(10, TOTAL_BUDGET - elapsed - 2)
    results.append(solve_instance(64, 0.55, 602, t2))
    
    return results
