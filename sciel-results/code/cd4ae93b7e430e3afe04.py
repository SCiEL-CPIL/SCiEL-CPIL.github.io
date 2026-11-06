"""Solve minimum hitting set of maximal cliques via MILP with cutting planes."""
# EVOLVE-BLOCK-START
import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def _generate_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch(adj, n):
    """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Choose pivot u from P ∪ X to minimize branching
        P_union_X = P | X
        if not P_union_X:
            return
        # Pick the vertex with maximum degree in the induced subgraph on P
        best_u = max(P_union_X, key=lambda u: len(adj[u] & P))
        
        candidates = P - adj[best_u]
        for v in list(candidates):
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            bk(new_R, new_P, new_X)
            P.remove(v)
            X.add(v)
    
    bk(frozenset(), set(range(n)), set())
    return cliques

def _solve_instance(n, p, seed, time_limit=10):
    """Solve the hitting set problem for one graph instance using cutting-plane approach."""
    adj = _generate_graph(n, p, seed)
    cliques = _bron_kerbosch(adj, n)
    
    if not cliques:
        return []
    
    num_vars = n
    
    # Start with a subset of constraints (use all initially since graphs are small)
    # But implement iterative refinement: start with fewer constraints, add violated ones
    
    # Initial constraint set: all maximal cliques
    active_cliques = list(cliques)
    
    def solve_milp(constraint_cliques, tl):
        """Solve MILP with given constraints."""
        if not constraint_cliques:
            return list(range(n))  # empty set satisfies no constraints
        
        m = len(constraint_cliques)
        A = np.zeros((m, num_vars), dtype=float)
        for i, clique in enumerate(constraint_cliques):
            for v in clique:
                A[i, v] = 1.0
        
        c = np.ones(num_vars, dtype=float)
        constraints = LinearConstraint(A, 1.0, np.inf)
        integrality = np.ones(num_vars, dtype=int)
        bounds = Bounds(0, 1)
        
        try:
            res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds,
                       options={"time_limit": tl})
            
            if res.status in (0, 1, 2, 3):  # optimal or feasible
                x = res.x
                selected = [v for v in range(num_vars) if x[v] > 0.5]
                return selected, res.status
            else:
                return None, res.status
        except Exception as e:
            return None, -1
    
    # First, try solving with all constraints
    start_time = time.time()
    elapsed = time.time() - start_time
    remaining_time = time_limit - elapsed
    
    if remaining_time < 2:
        # Not enough time, use greedy
        return _greedy_hitting_set(cliques, n)
    
    # Solve with all constraints
    selected, status = solve_milp(active_cliques, remaining_time)
    
    if selected is not None:
        # Verify solution and try to improve with local search
        selected_set = set(selected)
        
        # Check if all cliques are hit
        all_hit = all(any(v in selected_set for v in clique) for clique in cliques)
        
        if all_hit:
            # Try local search to improve
            improved = _local_search_improve(selected, cliques, adj, n, 
                                           time_limit - (time.time() - start_time))
            return sorted(improved)
        else:
            # Solution doesn't hit all cliques, use greedy
            return _greedy_hitting_set(cliques, n)
    else:
        # MILP failed, use greedy
        return _greedy_hitting_set(cliques, n)

def _local_search_improve(selected, cliques, adj, n, time_limit=5):
    """Try to improve the solution using local search."""
    selected = list(selected)
    selected_set = set(selected)
    
    # Verify all cliques are hit
    def is_valid(sel_set):
        return all(any(v in sel_set for v in clique) for clique in cliques)
    
    if not is_valid(selected_set):
        return selected
    
    best_solution = list(selected)
    best_size = len(selected)
    
    start_time = time.time()
    
    # Try removing vertices
    changed = True
    while changed and (time.time() - start_time) < time_limit * 0.5:
        changed = False
        for v in list(selected):
            new_set = selected_set - {v}
            if is_valid(new_set):
                selected_set = new_set
                selected = list(new_set)
                best_solution = list(selected)
                best_size = len(selected)
                changed = True
    
    # Try swapping: remove one, add another
    if (time.time() - start_time) < time_limit:
        for v_remove in list(selected):
            if time.time() - start_time > time_limit:
                break
            for v_add in range(n):
                if v_add in selected_set:
                    continue
                new_set = (selected_set - {v_remove}) | {v_add}
                if is_valid(new_set) and len(new_set) < best_size:
                    selected_set = new_set
                    selected = list(new_set)
                    best_solution = list(selected)
                    best_size = len(selected)
                    break
    
    return best_solution

def _greedy_hitting_set(cliques, n):
    """Greedy set cover approximation for hitting set."""
    remaining_cliques = [set(c) for c in cliques]
    selected = []
    
    while remaining_cliques:
        # Find vertex that hits the most remaining cliques
        best_v = -1
        best_count = -1
        for v in range(n):
            count = sum(1 for rc in remaining_cliques if v in rc)
            if count > best_count:
                best_count = count
                best_v = v
        
        if best_v == -1 or best_count <= 0:
            break
        
        selected.append(best_v)
        # Remove all cliques containing best_v
        remaining_cliques = [rc for rc in remaining_cliques if best_v not in rc]
    
    return sorted(selected)

def construct():
    results = []
    total_deadline = 140.0  # safety margin before 150s kill
    start = time.time()
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for idx, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - start
        remaining = total_deadline - elapsed
        if remaining < 5:
            # Not enough time, use fast greedy for all remaining
            adj = _generate_graph(n, p, seed)
            cliques = _bron_kerbosch(adj, n)
            results.append(_greedy_hitting_set(cliques, n))
        else:
            # Allocate time per instance, give more time to earlier instances
            # to allow for local search improvement
            time_limit = min(remaining / (3 - idx), 30)
            results.append(_solve_instance(n, p, seed, time_limit=time_limit))
    
    return results
# EVOLVE-BLOCK-END
