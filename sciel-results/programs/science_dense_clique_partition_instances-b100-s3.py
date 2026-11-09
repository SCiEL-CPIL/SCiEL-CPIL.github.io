import time
import random
from collections import defaultdict

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

        def find_edge(adj, n, rng=None, mode='min'):
            best_u, best_v = -1, -1
            best_score = float('inf') if mode in ('min', 'min_deg') else -1
            for u in range(n):
                if adj[u] == 0:
                    continue
                deg_u = adj[u].bit_count()
                for v in iter_bits(adj[u]):
                    if v > u:
                        common = (adj[u] & adj[v]).bit_count()
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
            return best_u, best_v

        def expand_clique(seed_u, seed_v, adj, rng=None):
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
            if len(cliques) < 2:
                return cliques
            
            # Build adjacency between cliques based on whether their union is a clique
            # Two cliques can be merged if every vertex in C1 is connected to every vertex in C2
            m = len(cliques)
            merge_adj = [set() for _ in range(m)]
            
            for i in range(m):
                for j in range(i + 1, m):
                    if is_clique_union(cliques[i], cliques[j], adj):
                        merge_adj[i].add(j)
                        merge_adj[j].add(i)
            
            # Find connected components in the merge graph
            visited = [False] * m
            components = []
            
            for i in range(m):
                if not visited[i]:
                    # BFS to find component
                    component = []
                    stack = [i]
                    visited[i] = True
                    while stack:
                        node = stack.pop()
                        component.append(node)
                        for neighbor in merge_adj[node]:
                            if not visited[neighbor]:
                                visited[neighbor] = True
                                stack.append(neighbor)
                    components.append(component)
            
            # Each component becomes one merged clique
            new_cliques = []
            for comp in components:
                merged = []
                for idx in comp:
                    merged.extend(cliques[idx])
                new_cliques.append(merged)
            
            return new_cliques

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
                C = expand_clique(u, v, adj, rng)
                cliques.append(C)
                remove_clique_edges(C, adj)
            cliques = try_merge(cliques, adj, n, deadline)
            return cliques

        def validate(cliques, adj_mask_orig, n):
            # Check each clique is valid
            for C in cliques:
                if len(C) < 2:
                    return False
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v = C[i], C[j]
                        if not (adj_mask_orig[u] & (1 << v)):
                            return False
            
            # Check edge coverage and no overlaps
            edge_set = set()
            for C in cliques:
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v = C[i], C[j]
                        key = (min(u, v), max(u, v))
                        if key in edge_set:
                            return False
                        edge_set.add(key)
            
            # Check all edges are covered
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
            
            # First pass: merge any cliques that can be merged
            best = try_merge(best, adj_mask_orig, n, deadline)
            best_count = len(best)
            
            # Second pass: try to move vertices between cliques to enable more merges
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
                        
                        # Check if C_remaining is still a clique
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
                        
                        # Try to merge C_remaining with another clique
                        for other_idx in range(len(best)):
                            if other_idx == idx:
                                continue
                            C2 = best[other_idx]
                            if is_clique_union(C_remaining, C2, adj_mask_orig):
                                # Check if v can be added to the merged clique or stays alone
                                new_C = C_remaining + C2
                                new_cliques = []
                                for k in range(len(best)):
                                    if k == idx:
                                        pass
                                    elif k == other_idx:
                                        pass
                                    else:
                                        new_cliques.append(best[k])
                                new_cliques.append(new_C)
                                # v needs to form a clique with someone or be part of an existing one
                                # For now, skip if v can't be placed
                                continue
                        else:
                            # Check if v can be added to C2
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

        def triangle_priority_partition(adj_mask_orig, n, deadline):
            """
            Prioritize forming triangles (3-cliques) first using a matching-based approach,
            then expand remaining edges into larger cliques.
            """
            adj = list(adj_mask_orig)
            cliques = []

            # Build list of edges with triangle counts
            edge_triangles = {}
            for u in range(n):
                for v in iter_bits(adj[u]):
                    if v > u:
                        common = adj[u] & adj[v]
                        tri_count = common.bit_count()
                        edge_triangles[(u, v)] = tri_count

            # Sort edges by triangle count (descending) - prioritize edges in many triangles
            sorted_edges = sorted(edge_triangles.keys(), key=lambda e: edge_triangles[e], reverse=True)

            for u, v in sorted_edges:
                if time.time() > deadline:
                    break
                # Check if edge still exists
                if not (adj[u] & (1 << v)):
                    continue
                # Find common neighbors that form a triangle
                common = adj[u] & adj[v]
                if not common:
                    continue
                # Pick the common neighbor that maximizes the resulting clique size
                candidates = list(iter_bits(common))
                best_w = -1
                best_size = 0
                for w in candidates:
                    # Check if w is still connected to both u and v
                    if not (adj[w] & (1 << u)) or not (adj[w] & (1 << v)):
                        continue
                    # Estimate clique size if we add w
                    U = adj[u] & adj[v] & adj[w]
                    size = 3 + U.bit_count()
                    if size > best_size:
                        best_size = size
                        best_w = w
                if best_w != -1:
                    # Extract the triangle and try to expand
                    C = expand_clique(u, v, adj, rng=None)
                    cliques.append(C)
                    remove_clique_edges(C, adj)

            # Phase 2: Process remaining edges with standard greedy
            while True:
                if time.time() > deadline:
                    break
                u, v = find_edge(adj, n, rng=None, mode='min')
                if u == -1:
                    break
                C = expand_clique(u, v, adj, rng=None)
                cliques.append(C)
                remove_clique_edges(C, adj)

            cliques = try_merge(cliques, adj, n, deadline)
            return cliques

        def multi_start_greedy(adj_mask_orig, n, deadline, K=5):
            """
            Try multiple starting vertices (top K by degree) and select the one
            that yields the largest maximal clique, reducing total clique count.
            """
            best_cliques = None
            best_count = float('inf')

            # Get top K vertices by degree
            degs = [(adj_mask_orig[u].bit_count(), u) for u in range(n)]
            degs.sort(reverse=True)
            top_K = [u for _, u in degs[:K]]

            for start_u in top_K:
                if time.time() > deadline:
                    break
                # Find best partner for start_u
                best_v = -1
                best_common = -1
                for v in iter_bits(adj_mask_orig[start_u]):
                    common = (adj_mask_orig[start_u] & adj_mask_orig[v]).bit_count()
                    if common > best_common:
                        best_common = common
                        best_v = v
                if best_v == -1:
                    continue

                # Run greedy partition starting with this edge
                adj = list(adj_mask_orig)
                cliques = []

                # First, extract the clique containing start_u and best_v
                C = expand_clique(start_u, best_v, adj, rng=None)
                cliques.append(C)
                remove_clique_edges(C, adj)

                # Continue with standard greedy
                while True:
                    if time.time() > deadline:
                        break
                    u, v = find_edge(adj, n, rng=None, mode='min')
                    if u == -1:
                        break
                    C = expand_clique(u, v, adj, rng=None)
                    cliques.append(C)
                    remove_clique_edges(C, adj)

                cliques = try_merge(cliques, adj, n, deadline)
                if validate(cliques, adj_mask_orig, n):
                    if len(cliques) < best_count:
                        best_count = len(cliques)
                        best_cliques = cliques

            return best_cliques

        best_cliques = None
        best_count = float('inf')

        # Strategy 1: Triangle priority partition
        t1_deadline = start_time + time_budget * 0.20
        cliques1 = triangle_priority_partition(adj_mask_orig, n, t1_deadline)
        if validate(cliques1, adj_mask_orig, n):
            best_cliques = cliques1
            best_count = len(cliques1)

        # Strategy 2: Multi-start greedy
        t2_deadline = start_time + time_budget * 0.35
        cliques2 = multi_start_greedy(adj_mask_orig, n, t2_deadline, K=5)
        if cliques2 is not None and validate(cliques2, adj_mask_orig, n):
            if len(cliques2) < best_count:
                best_count = len(cliques2)
                best_cliques = cliques2

        # Strategy 3: Standard deterministic runs with different modes
        for mode in ['min', 'max', 'min_deg']:
            if time.time() > start_time + time_budget * 0.55:
                break
            t_deadline = min(start_time + time_budget * 0.55, time.time() + max(0.5, time_budget * 0.05))
            cliques = greedy_partition(adj_mask_orig, n, rng=None, deadline=t_deadline, mode=mode)
            if validate(cliques, adj_mask_orig, n):
                if len(cliques) < best_count:
                    best_count = len(cliques)
                    best_cliques = cliques

        # Strategy 4: Randomized restarts with local search
        remaining = time_budget - (time.time() - start_time)
        if remaining > 3:
            num_restarts = max(6, int(remaining / 0.8))
            rng_master = random.Random(seed + 2000)
            modes = ['min', 'max', 'min_deg', 'max_deg']

            for r in range(num_restarts):
                if time.time() > start_time + time_budget - 2.0:
                    break
                rng = random.Random(rng_master.randint(0, 2**31))
                mode = modes[r % len(modes)]
                restart_deadline = min(start_time + time_budget - 2.0, time.time() + max(0.3, remaining / (num_restarts - r)))
                try:
                    cliques = greedy_partition(adj_mask_orig, n, rng=rng, deadline=restart_deadline, mode=mode)
                    if validate(cliques, adj_mask_orig, n):
                        ls_deadline = min(start_time + time_budget - 2.0, time.time() + 1.5)
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
    t1 = min(55, TOTAL_BUDGET - 60)
    results.append(solve_instance(32, 0.6, 601, t1))

    elapsed = time.time() - start_time
    t2 = max(15, TOTAL_BUDGET - elapsed - 3)
    results.append(solve_instance(64, 0.55, 602, t2))

    return results
