import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges_list = get_graph(n, p, seed)
        if not edges_list:
            return []

        edges = set(edges_list)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        def greedy_partition(edge_set, adj_set, use_min_common=False):
            """Greedy clique partition. Returns list of cliques."""
            remaining_edges = set(edge_set)
            rem_adj = [set(s) for s in adj_set]
            cliques = []

            while remaining_edges:
                if time.time() > deadline:
                    break

                # Choose seed edge
                if use_min_common:
                    best_key = float('inf')
                else:
                    best_key = -1
                best_u, best_v = -1, -1

                for u, v in remaining_edges:
                    common = len(rem_adj[u] & rem_adj[v])
                    if use_min_common:
                        if common < best_key:
                            best_key = common
                            best_u, best_v = u, v
                    else:
                        if common > best_key:
                            best_key = common
                            best_u, best_v = u, v

                if best_u == -1:
                    break

                # Build maximal clique from (best_u, best_v)
                C = [best_u, best_v]
                P = rem_adj[best_u] & rem_adj[best_v]
                P -= {best_u, best_v}

                while P:
                    if time.time() > deadline:
                        break
                    best_cand = None
                    best_score = -1
                    for cand in P:
                        score = len(rem_adj[cand] & set(C))
                        if score > best_score or (score == best_score and len(rem_adj[cand]) > len(rem_adj[best_cand] if best_cand is not None else set())):
                            best_score = score
                            best_cand = cand
                    if best_cand is None:
                        break
                    C.append(best_cand)
                    P = P & rem_adj[best_cand]

                cliques.append(C)

                # Remove edges in clique C
                Cset = set(C)
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v = C[i], C[j]
                        if u > v:
                            u, v = v, u
                        if (u, v) in remaining_edges:
                            remaining_edges.remove((u, v))
                            rem_adj[u].remove(v)
                            rem_adj[v].remove(u)

            return cliques, remaining_edges

        def merge_cliques(cliques, adj_set):
            """Try to merge cliques whose union is a clique."""
            changed = True
            while changed:
                changed = False
                new_cliques = []
                i = 0
                while i < len(cliques):
                    C1 = set(cliques[i])
                    merged = False
                    for j in range(i + 1, len(cliques)):
                        if time.time() > deadline:
                            return new_cliques + cliques[i:]
                        C2 = set(cliques[j])
                        union = C1 | C2
                        # Check if union is a clique
                        is_clique = True
                        for u in union:
                            for v in union:
                                if u < v and v not in adj_set[u]:
                                    is_clique = False
                                    break
                            if not is_clique:
                                break
                        if is_clique and len(union) > max(len(C1), len(C2)):
                            new_cliques.append(list(union))
                            cliques[j] = None
                            merged = True
                            changed = True
                            break
                    if not merged:
                        new_cliques.append(cliques[i])
                    i += 1
                new_cliques = [c for c in new_cliques if c is not None]
                cliques = new_cliques
            return cliques

        # Try multiple strategies and pick the best
        best_cliques = None
        best_count = float('inf')

        # Strategy 1: max common neighbors
        cliques1, leftover1 = greedy_partition(edges, adj, use_min_common=False)
        if not leftover1:
            cliques1 = merge_cliques(cliques1, adj)
            if len(cliques1) < best_count:
                best_count = len(cliques1)
                best_cliques = cliques1

        if time.time() > deadline:
            return best_cliques if best_cliques else []

        # Strategy 2: min common neighbors
        cliques2, leftover2 = greedy_partition(edges, adj, use_min_common=True)
        if not leftover2:
            cliques2 = merge_cliques(cliques2, adj)
            if len(cliques2) < best_count:
                best_count = len(cliques2)
                best_cliques = cliques2

        # Strategy 3: randomized restarts
        for restart in range(5):
            if time.time() > deadline:
                break
            rng = random.Random(seed + restart * 1000)
            shuffled_edges = list(edges)
            rng.shuffle(shuffled_edges)

            # Use shuffled order for greedy
            remaining_edges = set(edges)
            rem_adj = [set(s) for s in adj]
            cliques3 = []

            # Process edges in shuffled order
            for u, v in shuffled_edges:
                if time.time() > deadline:
                    break
                if (u, v) not in remaining_edges:
                    continue
                # Check if (u,v) is still an edge
                if v not in rem_adj[u]:
                    continue

                # Try to build a clique starting from (u,v)
                C = [u, v]
                P = rem_adj[u] & rem_adj[v]
                P -= {u, v}

                while P:
                    if time.time() > deadline:
                        break
                    best_cand = None
                    best_score = -1
                    for cand in P:
                        score = len(rem_adj[cand] & set(C))
                        if score > best_score:
                            best_score = score
                            best_cand = cand
                    if best_cand is None:
                        break
                    C.append(best_cand)
                    P = P & rem_adj[best_cand]

                cliques3.append(C)

                # Remove edges in C
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        a, b = C[i], C[j]
                        if a > b:
                            a, b = b, a
                        if (a, b) in remaining_edges:
                            remaining_edges.remove((a, b))
                            rem_adj[a].remove(b)
                            rem_adj[b].remove(a)

            if not remaining_edges:
                cliques3 = merge_cliques(cliques3, adj)
                if len(cliques3) < best_count:
                    best_count = len(cliques3)
                    best_cliques = cliques3

        return best_cliques if best_cliques else []

    results = []
    results.append(solve_instance(32, 0.6, 601))

    if time.time() > deadline:
        results.append(solve_instance(64, 0.55, 602))
        return results

    results.append(solve_instance(64, 0.55, 602))
    return results
