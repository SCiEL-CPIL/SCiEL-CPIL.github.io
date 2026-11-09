import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s kill

    def _ok():
        return (time.time() - start_time) < DEADLINE

    def get_adj(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def solve_instance(n, p, seed, time_budget):
        adj_orig = get_adj(n, p, seed)
        # Build initial edge set
        edge_set = set()
        for u in range(n):
            mask = adj_orig[u] & ((1 << u) - 1)  # only neighbors < u
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                edge_set.add((u, v))
                mask ^= lsb
        if not edge_set:
            return []

        best_cliques = None
        best_count = float('inf')

        # Try multiple random restarts with different tie-breaking
        for run_seed in range(30):
            if not _ok():
                break

            rng = random.Random(seed * 1000 + run_seed)
            remaining = set(edge_set)
            # Work adjacency as bitmasks
            adj = [adj_orig[i] for i in range(n)]
            cliques = []

            while remaining:
                if not _ok():
                    break

                # Find hardest edge: minimum common neighbors in remaining graph
                min_common = float('inf')
                best_edges = []
                for e in remaining:
                    u, v = e
                    common = adj[u] & adj[v]
                    # Count only vertices w where (u,w) and (v,w) are in remaining
                    # For dense graphs, most common neighbors will have edges in remaining
                    # Quick count: popcount of common
                    cnt = bin(common).count('1')
                    if cnt < min_common:
                        min_common = cnt
                        best_edges = [e]
                    elif cnt == min_common:
                        best_edges.append(e)

                if not best_edges:
                    break

                u, v = rng.choice(best_edges)
                clique = [u, v]

                # Expand: candidates are common neighbors with edges in remaining
                candidates = adj[u] & adj[v]
                # Filter: w must have (u,w) and (v,w) in remaining
                cand_list = []
                m = candidates
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    e_uw = (u, w) if u < w else (w, u)
                    e_vw = (v, w) if v < w else (w, v)
                    if e_uw in remaining and e_vw in remaining:
                        cand_list.append(w)
                    m ^= lsb

                # Greedy expansion: pick candidate with max degree in candidate subgraph
                while cand_list:
                    if not _ok():
                        break

                    # Compute degree within cand_list for each candidate
                    best_w = None
                    best_deg = -1
                    ties = []
                    for w in cand_list:
                        deg = 0
                        m2 = 0
                        for x in cand_list:
                            if x == w:
                                continue
                            e_wx = (w, x) if w < x else (x, w)
                            if e_wx in remaining:
                                deg += 1
                        if deg > best_deg:
                            best_deg = deg
                            best_w = w
                            ties = [w]
                        elif deg == best_deg:
                            ties.append(w)

                    if best_w is None:
                        break

                    if len(ties) > 1:
                        best_w = rng.choice(ties)

                    clique.append(best_w)
                    cand_list.remove(best_w)
                    # Filter: remaining candidates must connect to best_w in remaining
                    new_cand = []
                    for w in cand_list:
                        e_wbw = (w, best_w) if w < best_w else (best_w, w)
                        if e_wbw in remaining:
                            new_cand.append(w)
                    cand_list = new_cand

                cliques.append(clique)

                # Remove all edges within clique from remaining
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        a, b = clique[i], clique[j]
                        e = (a, b) if a < b else (b, a)
                        remaining.discard(e)
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)

            # If we broke early, add remaining edges as K2
            if remaining:
                for e in remaining:
                    cliques.append([e[0], e[1]])

            # Local improvement: merge cliques that form a larger clique
            improved = True
            while improved and _ok():
                improved = False
                for i in range(len(cliques)):
                    if i >= len(cliques):
                        break
                    for j in range(i + 1, len(cliques)):
                        # Check if union of cliques[i] and cliques[j] is a clique
                        can_merge = True
                        for a in cliques[i]:
                            for b in cliques[j]:
                                if a == b:
                                    continue
                                if not (adj_orig[a] & (1 << b)):
                                    can_merge = False
                                    break
                            if not can_merge:
                                break
                        if can_merge:
                            cliques[i] = cliques[i] + cliques[j]
                            del cliques[j]
                            improved = True
                            break
                    if improved:
                        break

            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = [list(c) for c in cliques]

        if best_cliques is None:
            # Fallback: each edge as K2
            return [[u, v] for u, v in edge_set]

        return best_cliques

    results = []
    # Instance 1
    results.append(solve_instance(32, 0.6, 601, 50))
    # Instance 2
    results.append(solve_instance(64, 0.55, 602, 90))

    return results
