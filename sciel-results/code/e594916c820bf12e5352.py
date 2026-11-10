"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from random import Random
from collections import defaultdict

def _random_graph(n, p, seed):
    """Generate a random graph using the same RNG sequence as the seed program."""
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _bron_kerbosch_maximal_cliques(n, adj):
    """
    Enumerate all maximal cliques using Bron-Kerbosch with pivot.
    adj: list of sets, adj[i] is the set of neighbors of i.
    Returns list of sets (each a clique).
    """
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(set(R))
            return
        
        if P:
            u = max(P, key=lambda v: len(adj[v] & P))
            candidates = P - adj[u]
        elif X:
            u = max(X, key=lambda v: len(adj[v] & X))
            candidates = P
            return
        else:
            return
        
        for v in list(candidates):
            R.add(v)
            P_new = P & adj[v]
            X_new = X & adj[v]
            bk(R, P_new, X_new)
            R.remove(v)
            P.remove(v)
            X.add(v)
    
    initial_P = set(range(n))
    bk(set(), initial_P, set())
    return cliques

def _solve_set_cover_subgradient(vertices, cliques, n, max_iters=500, restarts=5):
    """
    Solve the set cover problem using subgradient optimization.
    """
    if not cliques:
        return set()
    
    num_cliques = len(cliques)
    
    vertex_to_cliques = defaultdict(list)
    for ci, clique in enumerate(cliques):
        for v in clique:
            vertex_to_cliques[v].append(ci)
    
    coverage = {v: vertex_to_cliques.get(v, []) for v in range(n)}
    
    def subgradient_solve(coverage, num_cliques, n, rng, max_iters=500):
        y = np.ones(num_cliques) * 0.1
        best_solution = None
        best_size = float('inf')
        step_size = 1.0
        
        for it in range(max_iters):
            profit = np.zeros(n)
            for v in range(n):
                if coverage[v]:
                    profit[v] = sum(y[ci] for ci in coverage[v])
            
            selected = set()
            uncovered = set(range(num_cliques))
            
            order = sorted(range(n), key=lambda v: (-profit[v], rng.random()))
            
            for v in order:
                if not uncovered:
                    break
                covers_uncovered = any(ci in uncovered for ci in coverage[v])
                if covers_uncovered and profit[v] > 0:
                    selected.add(v)
                    for ci in coverage[v]:
                        if ci in uncovered:
                            uncovered.discard(ci)
            
            if uncovered:
                for ci in list(uncovered):
                    clique = cliques[ci]
                    v = rng.choice(list(clique))
                    selected.add(v)
                    for ci2 in coverage[v]:
                        uncovered.discard(ci2)
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_solution = set(selected)
            
            uncovered_set = set()
            for v in selected:
                for ci in coverage[v]:
                    uncovered_set.add(ci)
            
            truly_uncovered = set(range(num_cliques)) - uncovered_set
            
            g = np.zeros(num_cliques)
            for ci in truly_uncovered:
                g[ci] = 1.0
            
            cover_count = np.zeros(num_cliques)
            for v in selected:
                for ci in coverage[v]:
                    cover_count[ci] += 1
            
            for ci in range(num_cliques):
                if cover_count[ci] > 1:
                    g[ci] -= 0.1 * (cover_count[ci] - 1)
            
            step_size = 1.0 / (1.0 + 0.01 * it)
            
            y = np.maximum(0, y + step_size * g)
            
            y_max = np.max(y)
            if y_max > 0:
                y = y / y_max * 0.5
        
        return best_solution if best_solution else set()
    
    best_solution = None
    best_size = float('inf')
    
    for trial in range(restarts):
        rng = Random(1000 + trial * 7)
        sol = subgradient_solve(coverage, num_cliques, n, rng, max_iters=300)
        
        if sol is not None and len(sol) < best_size:
            best_size = len(sol)
            best_solution = set(sol)
    
    if best_solution is None or best_size > 15:
        selected = set()
        uncovered = set(range(num_cliques))
        
        while uncovered:
            best_v = None
            best_count = -1
            for v in range(n):
                if v in selected:
                    continue
                count = sum(1 for ci in coverage[v] if ci in uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count == 0:
                break
            selected.add(best_v)
            for ci in coverage[best_v]:
                uncovered.discard(ci)
        
        improved = True
        while improved:
            improved = False
            for v in list(selected):
                temp_selected = selected - {v}
                covered = set()
                for sv in temp_selected:
                    for ci in coverage[sv]:
                        covered.add(ci)
                if covered == set(range(num_cliques)):
                    selected = temp_selected
                    improved = True
                    break
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = selected
    
    if best_solution is None:
        best_solution = set()
    
    return best_solution

def construct():
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = _random_graph(n, p, seed)
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = _bron_kerbosch_maximal_cliques(n, adj)
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            out.append([])
        else:
            selected = _solve_set_cover_subgradient(range(n), cliques, n)
            out.append(sorted(selected))
    
    return out
# EVOLVE-BLOCK-END
