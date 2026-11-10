"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

DEADLINE = 150.0  # wall-clock limit

def _elapsed():
    return time.time() - _START_TIME

def _build_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch(adj):
    """Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivot."""
    n = len(adj)
    maximal_cliques = []
    
    def expand(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                maximal_cliques.append(frozenset(R))
            return
        
        # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
        if P:
            pivot_pool = P | X
        else:
            pivot_pool = X
        
        if not pivot_pool:
            # No pivot, iterate over all P
            candidates = list(P)
        else:
            # Pick pivot with max degree in P
            best_pivot = max(pivot_pool, key=lambda u: len(P & adj[u]))
            candidates = [v for v in P if best_pivot not in adj[v]]
        
        for v in candidates:
            if _elapsed() > DEADLINE - 5.0:
                return
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            expand(new_R, new_P, new_X)
            P.remove(v)
            X.add(v)
    
    expand(set(), set(range(n)), set())
    return maximal_cliques

def _greedy_set_cover(cliques, all_vertices):
    """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
    # Map each clique to index
    uncovered = list(range(len(cliques)))
    # For each vertex, compute which cliques it covers
    vertex_covers = {v: set() for v in all_vertices}
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_covers[v].add(i)
    
    covered = [False] * len(cliques)
    selected = []
    
    while any(covered[i] == False for i in uncovered) and _elapsed() < DEADLINE - 5.0:
        # Find vertex covering most uncovered cliques
        best_v = None
        best_count = -1
        for v in all_vertices:
            count = sum(1 for i in vertex_covers[v] if not covered[i])
            if count > best_count:
                best_count = count
                best_v = v
            elif count == best_count and best_v is not None:
                # Tie-break: prefer vertex with smaller total coverage (more specialized)
                if len(vertex_covers[v]) < len(vertex_covers[best_v]):
                    best_v = v
        
        if best_v is None or best_count <= 0:
            break
        
        selected.append(best_v)
        for i in vertex_covers[best_v]:
            covered[i] = True
    
    return selected

def _local_search(selected, cliques, all_vertices, max_rounds=100):
    """Try to remove vertices from selected while maintaining coverage."""
    selected_set = set(selected)
    
    # Precompute: for each clique, which selected vertices cover it
    def is_valid(sel_set):
        for clique in cliques:
            if not (clique & sel_set):
                return False
        return True
    
    for _ in range(max_rounds):
        if _elapsed() > DEADLINE - 5.0:
            break
        improved = False
        for v in list(selected_set):
            # Try removing v
            if not is_valid(selected_set - {v}):
                continue
            # Try adding a different vertex to see if we can swap
            # First just try removal
            selected_set.remove(v)
            improved = True
            break
        if not improved:
            # Try swaps: remove v, add w
            for v in list(selected_set):
                if _elapsed() > DEADLINE - 5.0:
                    break
                remaining = selected_set - {v}
                # Find uncovered cliques if we remove v
                uncovered_cliques = [c for c in cliques if not (c & remaining)]
                if not uncovered_cliques:
                    continue
                # Find a vertex w not in remaining that covers all uncovered_cliques
                for w in all_vertices:
                    if w in remaining:
                        continue
                    if all(w in c for c in uncovered_cliques):
                        selected_set = remaining | {w}
                        # Check if this is actually an improvement (same size, but maybe better structure)
                        # Only accept if size doesn't increase
                        if len(selected_set) <= len(selected_set | {v}):
                            improved = True
                        break
                if improved:
                    break
        if not improved:
            break
    
    return list(selected_set)

def _solve_instance(n, p, seed):
    """Solve one instance: find minimum vertex set meeting all maximal cliques of size >= 2."""
    adj = _build_graph(n, p, seed)
    cliques = _bron_kerbosch(adj)
    
    if not cliques:
        return []
    
    all_vertices = set(range(n))
    
    # Greedy set cover
    selected = _greedy_set_cover(cliques, all_vertices)
    
    # Local search to improve
    selected = _local_search(selected, cliques, all_vertices)
    
    # Try multiple random restarts with different greedy orderings
    best = selected
    best_size = len(selected)
    
    rng = random.Random(42)
    for restart in range(50):
        if _elapsed() > DEADLINE - 10.0:
            break
        
        # Randomize vertex ordering for greedy
        shuffled_vertices = list(all_vertices)
        rng.shuffle(shuffled_vertices)
        
        # Modified greedy with random tie-breaking
        covered = [False] * len(cliques)
        vertex_covers = {v: set() for v in shuffled_vertices}
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_covers[v].add(i)
        
        sel = []
        while any(not c for c in covered) and _elapsed() < DEADLINE - 10.0:
            best_v = None
            best_count = -1
            for v in shuffled_vertices:
                count = sum(1 for i in vertex_covers[v] if not covered[i])
                if count > best_count:
                    best_count = count
                    best_v = v
                elif count == best_count and best_v is not None:
                    # Random tie-break
                    if rng.random() < 0.5:
                        best_v = v
            
            if best_v is None or best_count <= 0:
                break
            
            sel.append(best_v)
            for i in vertex_covers[best_v]:
                covered[i] = True
        
        # Local search
        sel = _local_search(sel, cliques, all_vertices, max_rounds=50)
        
        if len(sel) < best_size:
            best_size = len(sel)
            best = sel
    
    return best

_START_TIME = time.time()

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if _elapsed() > DEADLINE - 20.0:
            # Not enough time left; return empty lists as fallback (invalid but avoids timeout)
            results.append([])
            continue
        sol = _solve_instance(n, p, seed)
        results.append(sol)
    
    return results
# EVOLVE-BLOCK-END
