"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random as _random_mod
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # leave 5s margin before 150s kill

    def random_graph(n, p, seed):
        rng = _random_mod.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        R = set()
        P = set(range(n))
        X = set()

        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            # Choose pivot
            u = max(P | X, key=lambda v: len(adj[v] & P)) if (P | X) else None
            candidates = P - (adj[u] if u is not None else set())
            for v in list(candidates):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)

        bk(R, P, X)
        return cliques

    def solve_hitting_set(n, cliques, time_limit):
        """Find a small set of vertices hitting all cliques."""
        if not cliques:
            return set()
        
        # Map each vertex to the set of clique indices it belongs to
        vertex_to_cliques = defaultdict(set)
        for idx, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].add(idx)
        
        # Greedy: repeatedly pick vertex covering most uncovered cliques
        uncovered = set(range(len(cliques)))
        selected = set()
        
        # Build list of (vertex, frozenset of clique indices it covers)
        v_cover = {v: frozenset(cs) for v, cs in vertex_to_cliques.items()}
        
        while uncovered:
            if time.time() > time_limit:
                break
            # Find vertex covering most uncovered
            best_v = None
            best_count = -1
            for v in range(n):
                if v in selected:
                    continue
                cov = v_cover.get(v, frozenset())
                cnt = len(cov & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_v = v
            if best_v is None or best_count <= 0:
                break
            selected.add(best_v)
            uncovered -= v_cover.get(best_v, frozenset())
        
        # Local search: try to remove vertices
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for v in list(selected):
                # Check if removing v still hits all cliques
                temp_sel = selected - {v}
                all_hit = True
                for c in cliques:
                    if not (c & temp_sel):
                        all_hit = False
                        break
                if all_hit:
                    selected = temp_sel
                    improved = True
                    break  # restart scan
        
        # Local search: try swapping one vertex for another
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for v in list(selected):
                # For each candidate w not in selected
                for w in range(n):
                    if w in selected:
                        continue
                    new_sel = selected - {v} | {w}
                    if len(new_sel) >= len(selected):
                        continue
                    all_hit = True
                    for c in cliques:
                        if not (c & new_sel):
                            all_hit = False
                            break
                    if all_hit:
                        selected = new_sel
                        improved = True
                        break
                if improved:
                    break
        
        return selected

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE:
            # Fallback: return all vertices in any edge
            edges = random_graph(n, p, seed)
            touched = set()
            for u, v in edges:
                touched.add(u)
                touched.add(v)
            results.append(sorted(touched))
            continue
        
        edges = random_graph(n, p, seed)
        adj = build_adj(n, edges)
        cliques = bron_kerbosch(adj, n)
        
        # Solve hitting set with remaining time
        remaining = DEADLINE - time.time()
        if remaining < 2.0:
            # Fallback
            touched = set()
            for u, v in edges:
                touched.add(u)
                touched.add(v)
            results.append(sorted(touched))
            continue
        
        selected = solve_hitting_set(n, cliques, DEADLINE - 1.0)
        
        # Verify validity
        valid = True
        for c in cliques:
            if not (c & selected):
                valid = False
                break
        
        if not valid:
            # Fallback to all touched vertices
            touched = set()
            for u, v in edges:
                touched.add(u)
                touched.add(v)
            results.append(sorted(touched))
        else:
            results.append(sorted(selected))
    
    return results
# EVOLVE-BLOCK-END
