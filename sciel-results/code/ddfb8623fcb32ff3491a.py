# EVOLVE-BLOCK-START
import time
from random import Random

def _random_graph_edges(n, p, seed):
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bron_kerbosch(adj, n):
    """Return all maximal cliques as frozensets of vertices."""
    cliques = []
    # Use pivot-based Bron-Kerbosch
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # Pivot selection: pick vertex in P∪X with max |P ∩ N(u)|
        union = P | X
        if union:
            pivot = max(union, key=lambda u: len(P & adj[u]))
        else:
            return
        candidates = P - adj[pivot]
        for v in list(candidates):
            bk(R | {v}, P & adj[v], X & adj[v])
            P = P - {v}
            X = X | {v}
    bk(frozenset(), frozenset(range(n)), frozenset())
    return cliques

def _find_min_hitting_set(vertices, cliques, timeout=40):
    """Find minimum hitting set for cliques using greedy + local search."""
    if not cliques:
        return set()
    
    # Greedy initialization
    covered = [False] * len(cliques)
    remaining = [set(c) for c in cliques]
    chosen = set()
    
    # Count how many cliques each vertex covers
    def greedy_init():
        chosen = set()
        remaining_cliques = list(range(len(cliques)))
        
        while remaining_cliques:
            # Find vertex that covers most remaining cliques
            best_v = None
            best_count = -1
            for v in vertices:
                count = 0
                for ci in remaining_cliques:
                    if v in cliques[ci]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count == 0:
                break
            chosen.add(best_v)
            # Remove covered cliques
            new_remaining = []
            for ci in remaining_cliques:
                if best_v not in cliques[ci]:
                    new_remaining.append(ci)
            remaining_cliques = new_remaining
        
        return chosen
    
    chosen = greedy_init()
    
    # Verify and fix if needed
    def is_valid(chosen_set):
        for c in cliques:
            if not chosen_set & c:
                return False
        return True
    
    if not is_valid(chosen):
        # Add vertices to cover remaining cliques
        for c in cliques:
            if not chosen & c:
                # Pick vertex in c with highest frequency in uncovered cliques
                best_v = max(c, key=lambda v: sum(1 for cc in cliques if v in cc and not chosen & cc))
                chosen.add(best_v)
    
    # Local search: try removals
    def try_remove(chosen_set):
        improved = True
        while improved:
            improved = False
            for v in list(chosen_set):
                if v in chosen_set:
                    test = chosen_set - {v}
                    if is_valid(test):
                        chosen_set = test
                        improved = True
        return chosen_set
    
    chosen = try_remove(chosen)
    
    # Local search: try swaps (remove one, add one)
    deadline = time.time() + timeout
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v_remove in list(chosen):
            if time.time() > deadline:
                break
            test = chosen - {v_remove}
            # Find cliques not covered by test
            uncovered = [c for c in cliques if not test & c]
            if not uncovered:
                chosen = test
                improved = True
                break
            # Try adding a single vertex to cover all uncovered
            # Find vertex that covers most uncovered cliques
            best_add = None
            best_cov = -1
            # Candidate vertices: those in uncovered cliques
            candidates = set()
            for c in uncovered:
                candidates |= c
            for v_add in candidates:
                cov = sum(1 for c in uncovered if v_add in c)
                if cov > best_cov:
                    best_cov = cov
                    best_add = v_add
            if best_add is not None and best_add not in chosen:
                test2 = test | {best_add}
                if is_valid(test2) and len(test2) < len(chosen):
                    chosen = test2
                    improved = True
                    break
    
    # Final removal pass
    chosen = try_remove(chosen)
    
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s
    
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            out.append(list(range(n)))
            continue
        
        edges = _random_graph_edges(n, p, seed)
        adj = _build_adjacency(n, edges)
        cliques = _bron_kerbosch(adj, n)
        
        vertices = list(range(n))
        hitting = _find_min_hitting_set(vertices, cliques, timeout=30)
        
        # Ensure validity
        if not all(hitting & c for c in cliques):
            # Fallback: add missing
            for c in cliques:
                if not hitting & c:
                    hitting.add(min(c))
        
        out.append(sorted(hitting))
    
    return out
# EVOLVE-BLOCK-END
