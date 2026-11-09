import time
from random import Random

def construct():
    t_start = time.time()
    deadline = t_start + 138.0

    # Keep the instance generator exactly as specified
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def build_adj(n, k, p, seed):
        edges = instance(n, k, p, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        edge_set = set()
        for u, v in edges:
            edge_set.add((u, v))
        return adj, edge_set, edges

    def dsatur_coloring(vertices, adj):
        """DSATUR greedy coloring. Returns dict vertex->color."""
        colors = {}
        sat_deg = {v: 0 for v in vertices}
        color_sets = {v: set() for v in vertices}
        uncolored = set(vertices)
        
        while uncolored:
            best = max(uncolored, key=lambda x: (sat_deg[x], len(adj[x] & uncolored)))
            c = 0
            while c in color_sets[best]:
                c += 1
            colors[best] = c
            color_sets[best].add(c)
            uncolored.discard(best)
            for w in adj[best]:
                if w in uncolored and c not in color_sets[w]:
                    color_sets[w].add(c)
                    sat_deg[w] = len(color_sets[w])
        return colors

    def greedy_clique_cover(edge_set, vertices, adj):
        """Cover edge_set with cliques from the graph induced by vertices."""
        remaining = set(edge_set)
        cliques = []
        
        while remaining:
            best_edge = None
            best_clique = None
            best_size = 0
            
            for e in list(remaining):
                u, v = e
                current = {u, v}
                candidates = (adj[u] & adj[v])
                candidates.discard(u)
                candidates.discard(v)
                sorted_cands = sorted(candidates, key=lambda x: len(adj[x]), reverse=True)
                changed = True
                while changed:
                    changed = False
                    for w in sorted_cands:
                        if w in current:
                            continue
                        can_add = all(w in adj[x] for x in current)
                        if can_add:
                            current.add(w)
                            changed = True
                            break
                if len(current) > best_size:
                    best_size = len(current)
                    best_edge = e
                    best_clique = set(current)
            
            if best_clique is None:
                e = next(iter(remaining))
                best_clique = {e[0], e[1]}
            
            cliques.append(frozenset(best_clique))
            clique_list = list(best_clique)
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    remaining.discard((a, b))
                    remaining.discard((b, a))
        
        return cliques

    def solve_instance_v1(n, k, p, seed):
        """Process I vertices, color their C-neighborhoods (ascending degree)."""
        adj, edge_set, edges = build_adj(n, k, p, seed)
        C = list(range(k))
        I = list(range(k, n))
        C_set = set(C)
        
        internal_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        used_edges = set()
        partition = []
        
        I_sorted = sorted(I, key=lambda v: len(adj[v] & C_set))
        
        for v in I_sorted:
            N_v = adj[v] & C_set
            if not N_v:
                continue
            
            n_v = len(N_v)
            if n_v == 1:
                u = next(iter(N_v))
                partition.append(sorted([v, u]))
                continue
            
            h_vertices = list(N_v)
            h_adj = {u: set() for u in h_vertices}
            
            for i in range(len(h_vertices)):
                u = h_vertices[i]
                for j in range(i + 1, len(h_vertices)):
                    w = h_vertices[j]
                    e = (min(u, w), max(u, w))
                    if e not in used_edges:
                        h_adj[u].add(w)
                        h_adj[w].add(u)
            
            colors = dsatur_coloring(h_vertices, h_adj)
            
            color_classes = {}
            for u, c in colors.items():
                if c not in color_classes:
                    color_classes[c] = []
                color_classes[c].append(u)
            
            for c in sorted(color_classes.keys()):
                S = color_classes[c]
                if len(S) < 1:
                    continue
                clique = sorted(S + [v])
                partition.append(clique)
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        a, b = S[i], S[j]
                        used_edges.add((min(a, b), max(a, b)))
        
        remaining_edges = internal_edges - used_edges
        if remaining_edges:
            rem_cliques = greedy_clique_cover(remaining_edges, C, adj)
            for fc in rem_cliques:
                cl = sorted(list(fc))
                if len(cl) >= 2:
                    partition.append(cl)
        
        return partition

    def solve_instance_v2(n, k, p, seed):
        """Process I vertices, color their C-neighborhoods (descending degree)."""
        adj, edge_set, edges = build_adj(n, k, p, seed)
        C = list(range(k))
        I = list(range(k, n))
        C_set = set(C)
        
        internal_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        used_edges = set()
        partition = []
        
        I_sorted = sorted(I, key=lambda v: len(adj[v] & C_set), reverse=True)
        
        for v in I_sorted:
            N_v = adj[v] & C_set
            if not N_v:
                continue
            
            n_v = len(N_v)
            if n_v == 1:
                u = next(iter(N_v))
                partition.append(sorted([v, u]))
                continue
            
            h_vertices = list(N_v)
            h_adj = {u: set() for u in h_vertices}
            
            for i in range(len(h_vertices)):
                u = h_vertices[i]
                for j in range(i + 1, len(h_vertices)):
                    w = h_vertices[j]
                    e = (min(u, w), max(u, w))
                    if e not in used_edges:
                        h_adj[u].add(w)
                        h_adj[w].add(u)
            
            colors = dsatur_coloring(h_vertices, h_adj)
            
            color_classes = {}
            for u, c in colors.items():
                if c not in color_classes:
                    color_classes[c] = []
                color_classes[c].append(u)
            
            for c in sorted(color_classes.keys()):
                S = color_classes[c]
                if len(S) < 1:
                    continue
                clique = sorted(S + [v])
                partition.append(clique)
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        a, b = S[i], S[j]
                        used_edges.add((min(a, b), max(a, b)))
        
        remaining_edges = internal_edges - used_edges
        if remaining_edges:
            rem_cliques = greedy_clique_cover(remaining_edges, C, adj)
            for fc in rem_cliques:
                cl = sorted(list(fc))
                if len(cl) >= 2:
                    partition.append(cl)
        
        return partition

    def solve_instance_v3(n, k, p, seed):
        """Process I vertices with random restarts."""
        best_partition = None
        best_count = float('inf')
        
        for trial in range(20):
            adj, edge_set, edges = build_adj(n, k, p, seed)
            C = list(range(k))
            I = list(range(k, n))
            C_set = set(C)
            
            internal_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
            used_edges = set()
            partition = []
            
            rng_trial = Random(seed + trial)
            I_shuffled = list(I)
            rng_trial.shuffle(I_shuffled)
            
            for v in I_shuffled:
                N_v = adj[v] & C_set
                if not N_v:
                    continue
                
                n_v = len(N_v)
                if n_v == 1:
                    u = next(iter(N_v))
                    partition.append(sorted([v, u]))
                    continue
                
                h_vertices = list(N_v)
                h_adj = {u: set() for u in h_vertices}
                
                for i in range(len(h_vertices)):
                    u = h_vertices[i]
                    for j in range(i + 1, len(h_vertices)):
                        w = h_vertices[j]
                        e = (min(u, w), max(u, w))
                        if e not in used_edges:
                            h_adj[u].add(w)
                            h_adj[w].add(u)
                
                colors = dsatur_coloring(h_vertices, h_adj)
                
                color_classes = {}
                for u, c in colors.items():
                    if c not in color_classes:
                        color_classes[c] = []
                    color_classes[c].append(u)
                
                for c in sorted(color_classes.keys()):
                    S = color_classes[c]
                    if len(S) < 1:
                        continue
                    clique = sorted(S + [v])
                    partition.append(clique)
                    for i in range(len(S)):
                        for j in range(i + 1, len(S)):
                            a, b = S[i], S[j]
                            used_edges.add((min(a, b), max(a, b)))
            
            remaining_edges = internal_edges - used_edges
            if remaining_edges:
                rem_cliques = greedy_clique_cover(remaining_edges, C, adj)
                for fc in rem_cliques:
                    cl = sorted(list(fc))
                    if len(cl) >= 2:
                        partition.append(cl)
            
            count = len(partition)
            if count < best_count:
                best_count = count
                best_partition = [list(cl) for cl in partition]
            
            if time.time() > deadline - 5:
                break
        
        return best_partition if best_partition is not None else []

    def solve_grouping(n, k, p, seed):
        """Structural grouping approach from inspiration."""
        adj, edge_set, edges = build_adj(n, k, p, seed)
        C = list(range(k))
        I = list(range(k, n))
        C_set = set(C)
        
        # Group I vertices by their neighborhood in C
        groups = {}
        for v in I:
            nbrs = frozenset(adj[v] & C_set)
            if nbrs not in groups:
                groups[nbrs] = []
            groups[nbrs].append(v)
        
        cliques = []
        covered_edges_in_C = set()
        
        for nbrs, verts in groups.items():
            if not verts:
                continue
            clique = list(nbrs) + verts
            if len(clique) >= 2:
                cliques.append(clique)
            
            nbrs_list = list(nbrs)
            for i in range(len(nbrs_list)):
                u = nbrs_list[i]
                for j in range(i + 1, len(nbrs_list)):
                    v = nbrs_list[j]
                    covered_edges_in_C.add((u, v) if u < v else (v, u))
        
        internal_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        remaining_edges = internal_edges - covered_edges_in_C
        
        if remaining_edges:
            rem_cliques = greedy_clique_cover(remaining_edges, C, adj)
            for fc in rem_cliques:
                cl = sorted(list(fc))
                if len(cl) >= 2:
                    cliques.append(cl)
        
        return cliques

    def validate_partition(partition, n, k, p, seed):
        """Check that the partition is valid."""
        adj, edge_set, edges = build_adj(n, k, p, seed)
        
        seen_edges = set()
        for clique in partition:
            if len(clique) < 2:
                return False
            if len(set(clique)) != len(clique):
                return False
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    if b not in adj[a]:
                        return False
                    e = (min(a, b), max(a, b))
                    if e in seen_edges:
                        return False
                    seen_edges.add(e)
        
        if seen_edges != edge_set:
            return False
        
        return True

    def try_all_solvers(n, k, p, seed):
        """Try multiple solvers and return the best valid result."""
        candidates = []
        
        solvers = [
            ("v1_asc", solve_instance_v1),
            ("v2_desc", solve_instance_v2),
            ("grouping", solve_grouping),
        ]
        
        for name, solver in solvers:
            try:
                result = solver(n, k, p, seed)
                if validate_partition(result, n, k, p, seed):
                    candidates.append((len(result), result))
            except Exception:
                pass
        
        if time.time() < deadline - 10:
            try:
                result = solve_instance_v3(n, k, p, seed)
                if validate_partition(result, n, k, p, seed):
                    candidates.append((len(result), result))
            except Exception:
                pass
        
        if not candidates:
            return []
        
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    
    for spec in specs:
        if time.time() > deadline - 5:
            while len(results) < 3:
                results.append([])
            return results
        
        n, k, p, seed = spec
        result = try_all_solvers(n, k, p, seed)
        
        cleaned = []
        for clique in result:
            cleaned.append([int(x) for x in clique])
        results.append(cleaned)
    
    return results
