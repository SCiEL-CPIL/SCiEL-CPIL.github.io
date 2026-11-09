import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    def random_graph_edges(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_maximal_cliques(n, edges):
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        maximal_cliques = []
        
        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P union X to minimize recursion
            # Heuristic: pick u with max degree in P
            U = P | X
            if U:
                # Find pivot with max connections in P
                max_deg = -1
                pivot = None
                for u in U:
                    deg = len(P & adj[u])
                    if deg > max_deg:
                        max_deg = deg
                        pivot = u
                # If pivot exists, only recurse on P \ adj[pivot]
                candidates = P - adj[pivot] if pivot is not None else P
            else:
                candidates = P
            
            for v in candidates:
                expand(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
                
        expand(frozenset(), set(range(n)), set())
        return maximal_cliques

    def greedy_hitting_set(cliques, n):
        if not cliques:
            return []
        
        # cliques is a list of frozensets
        # We want to pick vertices to cover all cliques
        # Greedy: pick vertex that covers most uncovered cliques
        
        # Map vertex -> list of indices of cliques it covers
        # Or simpler: maintain list of uncovered cliques
        
        uncovered = list(cliques)
        if not uncovered:
            return []
        
        # Precompute incidence: vertex -> set of clique indices?
        # Since N is small (<=40), we can just iterate.
        
        selected = []
        
        # Optimization: If a clique is a subset of another, the larger one is "harder"?
        # No, hitting the smaller one doesn't guarantee hitting the larger one.
        # But hitting the larger one doesn't guarantee hitting the smaller one.
        # Actually, if C1 subset C2, and we hit C2 with v, it's possible v is not in C1.
        # So we must treat them as independent constraints.
        
        # However, we can prune: if we select a vertex, all cliques containing it are satisfied.
        
        # To speed up, we can sort vertices by initial frequency?
        # Or just run the standard greedy loop.
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Count how many uncovered cliques each vertex covers
            counts = [0] * n
            for c in uncovered:
                for v in c:
                    counts[v] += 1
            
            # Pick vertex with max count
            best_v = -1
            max_count = -1
            for v in range(n):
                if counts[v] > max_count:
                    max_count = counts[v]
                    best_v = v
            
            if best_v == -1:
                break # Should not happen if uncovered is not empty and cliques are valid
                
            selected.append(best_v)
            
            # Remove all cliques containing best_v
            new_uncovered = []
            for c in uncovered:
                if best_v not in c:
                    new_uncovered.append(c)
            uncovered = new_uncovered
            
        return selected

    def solve_instance(n, p, seed):
        edges = random_graph_edges(n, p, seed)
        if not edges:
            return []
        
        cliques = bron_kerbosch_maximal_cliques(n, edges)
        
        if not cliques:
            return []
            
        # Greedy Hitting Set
        solution = greedy_hitting_set(cliques, n)
        
        # Local Search Improvement:
        # Try to remove vertices if possible
        # Try to swap vertices to reduce size?
        # For now, greedy is a strong baseline.
        
        return sorted(solution)

    instances = [
        (30, 0.3, 201),
        (35, 0.25, 202),
        (40, 0.2, 203)
    ]
    
    results = []
    for n, p, seed in instances:
        if time.time() > deadline:
            # Fallback: return empty lists or previous best?
            # If we run out of time, we should return what we have or a valid trivial solution.
            # A trivial valid solution is all vertices? No, that's large.
            # If we haven't computed, we can't return a valid solution easily.
            # But the deadline is 110s, and these graphs are small. Bron-Kerbosch should be fast.
            # If we hit deadline, we return the current partial results.
            results.append([])
            continue
            
        res = solve_instance(n, p, seed)
        results.append(res)
        
    return results

# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from collections import defaultdict

    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit

    def random_graph_edges(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_maximal_cliques(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        maximal_cliques = []
        
        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            
            U = P | X
            if U:
                max_deg = -1
                pivot = None
                for u in U:
                    deg = len(P & adj[u])
                    if deg > max_deg:
                        max_deg = deg
                        pivot = u
                candidates = P - adj[pivot] if pivot is not None else P
            else:
                candidates = P
            
            for v in candidates:
                expand(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
                
        expand(frozenset(), set(range(n)), set())
        return maximal_cliques

    def greedy_hitting_set(cliques, n):
        if not cliques:
            return []
        
        uncovered = list(cliques)
        if not uncovered:
            return []
        
        selected = []
        
        while uncovered:
            if time.time() > deadline:
                break
            
            counts = [0] * n
            for c in uncovered:
                for v in c:
                    counts[v] += 1
            
            best_v = -1
            max_count = -1
            for v in range(n):
                if counts[v] > max_count:
                    max_count = counts[v]
                    best_v = v
            
            if best_v == -1:
                break
                
            selected.append(best_v)
            
            new_uncovered = []
            for c in uncovered:
                if best_v not in c:
                    new_uncovered.append(c)
            uncovered = new_uncovered
            
        return selected

    def solve_instance(n, p, seed):
        edges = random_graph_edges(n, p, seed)
        if not edges:
            return []
        
        cliques = bron_kerbosch_maximal_cliques(n, edges)
        
        if not cliques:
            return []
            
        solution = greedy_hitting_set(cliques, n)
        
        return sorted(solution)

    instances = [
        (30, 0.3, 201),
        (35, 0.25, 202),
        (40, 0.2, 203)
    ]
    
    results = []
    for n, p, seed in instances:
        if time.time() > deadline:
            results.append([])
            continue
            
        res = solve_instance(n, p, seed)
        results.append(res)
        
    return results
# EVOLVE-BLOCK-END
