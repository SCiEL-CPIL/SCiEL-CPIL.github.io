"""Bootstrap direct approach: meet every maximal clique (size>=2) with a minimal-ish transversal."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

_DEADLINE = 150.0
_START_TIME = time.time()

def _remaining():
    return _DEADLINE - time.time()

def construct():
    start = time.time()
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() - start > _DEADLINE - 5:
            # Fallback: return all touched vertices for remaining instances
            rng = random.Random(seed)
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
            touched = sorted({x for e in edges for x in e})
            out.append(touched)
            continue
        sol = _solve_instance(n, p, seed)
        out.append(sol)
    return out

def _solve_instance(n, p, seed):
    # Build adjacency
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    
    # Enumerate all maximal cliques via Bron-Kerbosch with pivot
    cliques = []
    def bronk(R, P, X):
        if time.time() - _START_TIME > _DEADLINE - 10:
            return
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # Choose pivot
        best_pivot = None
        max_count = -1
        for v in P:
            c = len(adj[v] & (P | X))
            if c > max_count:
                max_count = c
                best_pivot = v
        if best_pivot is not None:
            candidates = P - adj[best_pivot]
        else:
            candidates = set(P)
        for v in list(candidates):
            bronk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
    
    bronk(set(), set(range(n)), set())
    
    # Filter: only cliques of size >= 2 matter
    cliques = [c for c in cliques if len(c) >= 2]
    
    if not cliques:
        return []
    
    # Try MILP approach first for better solutions
    milp_sol = _solve_milp(n, cliques)
    if milp_sol is not None and time.time() - _START_TIME < _DEADLINE - 15:
        # Verify and potentially improve with local search
        verified = _verify_and_improve(n, adj, cliques, milp_sol)
        return verified
    
    # Fallback: Greedy set cover
    return _greedy_set_cover(n, cliques)

def _solve_milp(n, cliques):
    """Solve the minimum clique transversal as a 0-1 ILP using scipy's MILP solver."""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
    except ImportError:
        return None
    
    if not cliques:
        return set()
    
    # Variables: x[v] = 1 if vertex v is selected
    # Minimize: sum(x[v])
    # Subject to: for each clique C, sum_{v in C} x[v] >= 1
    #             x[v] in {0, 1}
    
    m = len(cliques)
    
    # Objective: minimize sum of x
    c = np.ones(n)
    
    # Constraints: A @ x >= 1 for each clique
    # Build constraint matrix
    A_rows = []
    for clique in cliques:
        row = np.zeros(n)
        for v in clique:
            row[v] = 1.0
        A_rows.append(row)
    
    if not A_rows:
        return set()
    
    A = np.array(A_rows)
    lb = np.ones(m)  # Each clique must be hit at least once
    ub = np.full(m, np.inf)
    
    # Variable bounds: 0 <= x[v] <= 1, integer
    var_lb = np.zeros(n)
    var_ub = np.ones(n)
    integrality = np.ones(n)  # All variables are binary
    
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, lb, ub),
            integrality=integrality,
            bounds=Bounds(var_lb, var_ub)
        )
        
        if res.status == 0:  # Success
            selected = set()
            for v in range(n):
                if res.x[v] > 0.5:
                    selected.add(v)
            return selected
        else:
            return None
    except Exception:
        return None

def _verify_and_improve(n, adj, cliques, initial_sol):
    """Verify solution and improve with local search (remove redundant, try swaps)."""
    selected = set(initial_sol)
    
    # Remove redundant vertices
    changed = True
    while changed and time.time() - _START_TIME < _DEADLINE - 10:
        changed = False
        for v in list(selected):
            if time.time() - _START_TIME > _DEADLINE - 10:
                break
            # Check if removing v still covers all cliques
            test_set = selected - {v}
            if all(any(x in test_set for x in clique) for clique in cliques):
                selected = test_set
                changed = True
                break
    
    # Try 1-swap improvements: remove one, add another
    improved = True
    while improved and time.time() - _START_TIME < _DEADLINE - 10:
        improved = False
        for v_out in list(selected):
            if time.time() - _START_TIME > _DEADLINE - 10:
                break
            for v_in in range(n):
                if v_in in selected:
                    continue
                test_set = (selected - {v_out}) | {v_in}
                if len(test_set) < len(selected):
                    # Same size, try to improve
                    pass
                # Check if this swap maintains coverage and doesn't increase size
                if all(any(x in test_set for x in clique) for clique in cliques):
                    if len(test_set) < len(selected):
                        selected = test_set
                        improved = True
                        break
        # Also try pure additions if we can remove more later
        # For now, focus on removals and swaps that reduce size
    
    return sorted(selected)

def _greedy_set_cover(n, cliques):
    """Greedy set cover: select vertices to hit all cliques."""
    remaining = list(cliques)
    selected = set()
    
    # Precompute vertex-clique incidence for efficiency
    vertex_to_cliques = defaultdict(list)
    for i, c in enumerate(cliques):
        for v in c:
            vertex_to_cliques[v].append(i)
    
    uncovered = set(range(len(cliques)))
    
    while uncovered and time.time() - _START_TIME < _DEADLINE - 15:
        # Find vertex that hits the most uncovered cliques
        best_v = None
        best_count = 0
        for v in range(n):
            count = 0
            for ci in vertex_to_cliques[v]:
                if ci in uncovered:
                    count += 1
                    if count <= best_count:
                        continue
            if count > best_count:
                best_count = count
                best_v = v
                if count == len(uncovered):
                    break
        if best_v is None or best_count == 0:
            break
        selected.add(best_v)
        for ci in vertex_to_cliques[best_v]:
            uncovered.discard(ci)
    
    # If we still have uncovered cliques, add one vertex from each
    for ci in uncovered:
        # Pick a vertex from this clique
        v = next(iter(cliques[ci]))
        selected.add(v)
    
    # Local search to improve
    return _verify_and_improve(n, None, cliques, selected)

# EVOLVE-BLOCK-END
