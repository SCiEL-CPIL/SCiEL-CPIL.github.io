import time
import random

def construct():
    n = 15
    deadline = time.time() + 85.0
    best_edges = None
    best_count = -1

    def build_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def get_neighbors(adj, u):
        """Return list of neighbors of u from adjacency bitmask."""
        nbrs = []
        for v in range(n):
            if (adj[u] >> v) & 1:
                nbrs.append(v)
        return nbrs

    def check(adj, deg, edges):
        """Check if graph with given adj/deg/edges is valid."""
        # Precompute neighbor lists
        nbrs = [get_neighbors(adj, i) for i in range(n)]
        # For each edge (u,v): forbidden if there exist s in N(u)\{v}, t in N(v)\{u} with deg[s]==deg[t]
        for u, v in edges:
            # Build set of degrees of neighbors of u excluding v
            du_set = set()
            for s in nbrs[u]:
                if s != v:
                    du_set.add(deg[s])
            if not du_set:
                continue
            for t in nbrs[v]:
                if t != u:
                    if deg[t] in du_set:
                        return False
        return True

    def check_edges(edges):
        """Check validity from edge list alone."""
        adj = build_adj(edges)
        deg = [bin(adj[i]).count('1') for i in range(n)]
        return check(adj, deg, edges)

    def try_improve(start_edges):
        edges = list(start_edges)
        adj = build_adj(edges)
        deg = [bin(adj[i]).count('1') for i in range(n)]
        present = set()
        for u, v in edges:
            a, b = (u, v) if u < v else (v, u)
            present.add((a, b))

        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for u in range(n):
                if time.time() > deadline - 2:
                    break
                for v in range(u + 1, n):
                    if (u, v) in present:
                        continue
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    deg[u] += 1
                    deg[v] += 1
                    if check(adj, deg, edges + [[u, v]]):
                        present.add((u, v))
                        edges.append([u, v])
                        improved = True
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        deg[u] -= 1
                        deg[v] -= 1
        return edges

    def local_search(start_edges):
        import math
        edges = list(start_edges)
        adj = build_adj(edges)
        deg = [bin(adj[i]).count('1') for i in range(n)]
        present = set()
        for u, v in edges:
            a, b = (u, v) if u < v else (v, u)
            present.add((a, b))

        best = len(edges)
        best_copy = [e[:] for e in edges]

        # Greedy add
        for u in range(n):
            if time.time() > deadline - 3:
                return best_copy
            for v in range(u + 1, n):
                if (u, v) in present:
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                if check(adj, deg, edges + [[u, v]]):
                    present.add((u, v))
                    edges.append([u, v])
                    if len(edges) > best:
                        best = len(edges)
                        best_copy = [e[:] for e in edges]
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    deg[u] -= 1
                    deg[v] -= 1

        # Conflict-graph based greedy independent set for packing edges
        # This is a specialized heuristic: treat each potential edge as a node,
        # two edges conflict if both being present creates a violation.
        # Greedily select edges with minimum conflict degree.
        if time.time() < deadline - 10:
            cg_edges = [e[:] for e in best_copy]
            cg_adj = build_adj(cg_edges)
            cg_deg = [bin(cg_adj[i]).count('1') for i in range(n)]
            cg_present = set()
            for u, v in cg_edges:
                a, b = (u, v) if u < v else (v, u)
                cg_present.add((a, b))

            # All potential edges not yet present
            all_potential = [(a, b) for a in range(n) for b in range(a + 1, n)]
            missing_edges = [e for e in all_potential if e not in cg_present]

            # Try to greedily add edges using a conflict-aware ordering
            # An edge e=(u,v) can be added if check passes
            # We prioritize edges that have fewer "near-conflicts"
            # (i.e., edges that would create violations with few existing edges)
            added = []
            for _ in range(50):  # up to 50 rounds of greedy
                if time.time() > deadline - 5:
                    break
                round_added = False
                # Score each missing edge by how many new violations it would create
                scored = []
                for me in missing_edges:
                    a, b = me
                    if (a, b) in cg_present:
                        continue
                    # Tentatively add
                    cg_adj[a] |= (1 << b)
                    cg_adj[b] |= (1 << a)
                    cg_deg[a] += 1
                    cg_deg[b] += 1
                    valid = check(cg_adj, cg_deg, cg_edges + [[a, b]])
                    # Revert
                    cg_adj[a] &= ~(1 << b)
                    cg_adj[b] &= ~(1 << a)
                    cg_deg[a] -= 1
                    cg_deg[b] -= 1
                    if valid:
                        scored.append((0, me))  # priority: valid edges first
                    else:
                        scored.append((1, me))

                # Sort: valid edges first, then by some heuristic (e.g., lower degree sum)
                scored.sort(key=lambda x: (x[0], cg_deg[x[1][0]] + cg_deg[x[1][1]]))
                for score, me in scored:
                    if score > 0:
                        break  # Only add valid edges in this round
                    a, b = me
                    cg_adj[a] |= (1 << b)
                    cg_adj[b] |= (1 << a)
                    cg_deg[a] += 1
                    cg_deg[b] += 1
                    cg_edges.append([a, b])
                    cg_present.add((a, b))
                    missing_edges.remove(me)
                    round_added = True
                    if len(cg_edges) > best:
                        best = len(cg_edges)
                        best_copy = [e[:] for e in cg_edges]
                if not round_added:
                    break

        # Final greedy polish on best_copy
        if time.time() < deadline - 2:
            pol_edges = [e[:] for e in best_copy]
            pol_adj = build_adj(pol_edges)
            pol_deg = [bin(pol_adj[i]).count('1') for i in range(n)]
            pol_present = set()
            for u, v in pol_edges:
                a, b = (u, v) if u < v else (v, u)
                pol_present.add((a, b))
            improved_pol = True
            while improved_pol and time.time() < deadline - 1:
                improved_pol = False
                for u in range(n):
                    if time.time() > deadline - 1:
                        break
                    for v in range(u + 1, n):
                        if (u, v) in pol_present:
                            continue
                        pol_adj[u] |= (1 << v)
                        pol_adj[v] |= (1 << u)
                        pol_deg[u] += 1
                        pol_deg[v] += 1
                        if check(pol_adj, pol_deg, pol_edges + [[u, v]]):
                            pol_present.add((u, v))
                            pol_edges.append([u, v])
                            improved_pol = True
                            if len(pol_edges) > best:
                                best = len(pol_edges)
                                best_copy = [e[:] for e in pol_edges]
                        else:
                            pol_adj[u] &= ~(1 << v)
                            pol_adj[v] &= ~(1 << u)
                            pol_deg[u] -= 1
                            pol_deg[v] -= 1

        return best_copy

    # Generate candidates
    candidates = []

    # K_{7,8} bipartite graph - known to be valid (degrees 8 and 7 are distinct)
    A = list(range(7))
    B = list(range(7, 15))
    k78_edges = []
    for x in A:
        for y in B:
            k78_edges.append([x, y])
    candidates.append(k78_edges)

    # Other bipartite graphs
    for a in range(1, 15):
        edges = []
        A = list(range(a))
        B = list(range(a, 15))
        for x in A:
            for y in B:
                edges.append([x, y])
        candidates.append(edges)

    # Complete graph
    complete = [[u, v] for u in range(15) for v in range(u + 1, 15)]
    candidates.append(complete)

    # Empty
    candidates.append([])

    # Random graphs with various densities
    for seed in range(20):
        random.seed(seed)
        density = 0.15 + 0.02 * seed
        edges = []
        for u in range(15):
            for v in range(u + 1, 15):
                if random.random() < density:
                    edges.append([u, v])
        candidates.append(edges)

    # Degree-diversity biased random: try to create varied degrees
    for seed in range(5):
        random.seed(seed + 100)
        edges = []
        for u in range(15):
            target_deg = random.randint(1, 10)
            candidates_v = [v for v in range(15) if v != u]
            random.shuffle(candidates_v)
            for v in candidates_v[:target_deg]:
                a, b = (u, v) if u < v else (v, u)
                if (a, b) not in set((min(e[0],e[1]), max(e[0],e[1])) for e in edges):
                    edges.append([u, v])
        candidates.append(edges)

    def fix_invalid(edges):
        if check_edges(edges):
            return edges
        adj = build_adj(edges)
        deg = [bin(adj[i]).count('1') for i in range(n)]
        present = set()
        for u, v in edges:
            a, b = (u, v) if u < v else (v, u)
            present.add((a, b))

        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            if check_edges(edges):
                break
            # Find all violations and count which edge appears most
            adj = build_adj(edges)
            deg = [bin(adj[i]).count('1') for i in range(n)]
            edge_violation_count = {}
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] >> v) & 1:
                        continue
                    for s in range(n):
                        if s == u or s == v:
                            continue
                        if not (adj[u] >> s) & 1:
                            continue
                        for t in range(n):
                            if t == u or t == v or t == s:
                                continue
                            if not (adj[v] >> t) & 1:
                                continue
                            if deg[s] == deg[t]:
                                edge_violation_count[(u, v)] = edge_violation_count.get((u, v), 0) + 1
            if not edge_violation_count:
                break
            # Remove the edge with most violations
            worst_edge = max(edge_violation_count, key=lambda e: edge_violation_count[e])
            edges = [e for e in edges if not ((e[0] == worst_edge[0] and e[1] == worst_edge[1]) or
                                              (e[0] == worst_edge[1] and e[1] == worst_edge[0]))]
            changed = True
        return edges

    for cand in candidates:
        if time.time() > deadline - 3:
            break
        cand = fix_invalid(cand)
        improved = try_improve(cand)
        if check_edges(improved):
            cnt = len(improved)
            if cnt > best_count:
                best_count = cnt
                best_edges = improved

        if best_edges is not None and time.time() < deadline - 15:
            ls_result = local_search(best_edges)
            if check_edges(ls_result):
                cnt = len(ls_result)
                if cnt > best_count:
                    best_count = cnt
                    best_edges = ls_result

    if best_edges is None:
        return [[0, v] for v in range(1, 15)]

    out = [[int(u), int(v)] for u, v in best_edges]
    return out
