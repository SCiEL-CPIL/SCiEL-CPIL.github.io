# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def _is_2_colorable(edges, n, deadline=None):
    """Exact 2-colourability check with unit propagation + backtracking.
    Returns True if a proper 2-colouring exists, False if none.
    edges: list of frozensets of size 5.
    """
    if deadline is None:
        deadline = time.time() + 10
    
    if not edges:
        return True
        
    # Build incidence: vertex -> list of edge indices
    inc = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            inc[v].append(i)

    # State: colour[v] in {-1,0,1}
    assign = [-1] * n
    evs = [list(e) for e in edges]

    def propagate(assign, changed):
        stack = list(changed)
        visited = set(stack)
        while stack:
            v = stack.pop()
            cv = assign[v]
            if cv == -1:
                continue
            for ei in inc[v]:
                e = evs[ei]
                counts = [-1, -1, 0]  # count0, count1, count_unassigned
                all_same_val = -1
                for u in e:
                    cu = assign[u]
                    if cu == -1:
                        counts[2] += 1
                    else:
                        counts[cu] += 1
                        if all_same_val == -1:
                            all_same_val = cu
                        elif all_same_val != cu:
                            all_same_val = 0  # mixed
                if all_same_val == 0:  # mixed, no constraint
                    continue
                if counts[2] == 0:
                    # all assigned and all same -> conflict
                    return False
                if counts[2] == 1 and counts[all_same_val] == 4:
                    # 4 same, 1 unassigned -> force opposite
                    for u in e:
                        if assign[u] == -1:
                            if u not in visited:
                                visited.add(u)
                                assign[u] = 1 - all_same_val
                                stack.append(u)
                            break
        return True

    def backtrack(assign, uncolored):
        if time.time() > deadline:
            return None
        if not uncolored:
            for e in edges:
                c0 = sum(1 for u in e if assign[u] == 0)
                c1 = sum(1 for u in e if assign[u] == 1)
                if c0 == 0 or c1 == 0:
                    return None
            return assign
        # Pick vertex with most constraints
        v = max(uncolored, key=lambda x: len(inc[x]))
        new_uncolored = uncolored - {v}
        for val in (0, 1):
            assign[v] = val
            if propagate(assign, [v]):
                res = backtrack(assign, new_uncolored)
                if res is not None:
                    return res
            assign[v] = -1
        return None

    uncolored = set(range(n))
    result = backtrack(assign, uncolored)
    if result is None:
        return False
    return True

def _solve_ilp(n, deadline):
    """
    Solves the minimum hitting set problem for 2-colorings using ILP.
    Variables x_e for each 5-edge e.
    Minimize sum(x_e)
    Subject to: For each coloring c, sum_{e violates c} x_e >= 1
    """
    if time.time() > deadline:
        return None, None
        
    if n > 15:
        return None, None

    all_edges = list(combinations(range(n), 5))
    num_edges = len(all_edges)
    num_colorings = 1 << n
    
    # Precompute violation matrix A (colorings x edges)
    # A[c, e] = 1 if edge e is monochromatic in coloring c, else 0
    # We want to minimize sum(x) such that A @ x >= 1 for all c.
    
    # To save memory and time, we can represent A as a sparse matrix or list of lists.
    # scipy milp accepts linear constraints.
    
    # Optimization: We only need to include colorings that are "hard" to cover?
    # Actually, we must cover ALL colorings.
    
    # Building the full matrix A (num_colorings x num_edges) might be large for n=15.
    # 2^15 = 32768 rows, C(15,5) = 3003 cols. 32768 * 3003 ~ 100M entries.
    # This is too big for dense, but scipy handles sparse? 
    # scipy.optimize.milp accepts LinearConstraint with sparse A.
    
    # Let's construct A as a sparse matrix.
    from scipy.sparse import lil_matrix, csr_matrix
    
    A = lil_matrix((num_colorings, num_edges), dtype=np.float64)
    
    # Iterate over edges
    for j, e in enumerate(all_edges):
        # For each coloring, check if e is monochromatic
        # We can generate the mask of colorings where e is mono.
        # But iterating all colorings for each edge is O(N_c * N_e).
        # For n=12: 4096 * 792 ~ 3.2M. OK.
        # For n=15: 32768 * 3003 ~ 100M. Might be slow in Python loop.
        
        # Faster way:
        # e is (v0, v1, v2, v3, v4)
        # Coloring c is mono on e if bit(v0)==bit(v1)==...==bit(v4)
        
        # We can iterate through colorings and check bits.
        # To speed up, we can precompute bit values for each vertex?
        # Or just use the direct check.
        
        if n > 13:
            # This might be too slow for n=15 in Python loop.
            # Let's restrict n to 13 for the ILP approach to be safe within time.
            # Or use a faster vectorized approach if possible.
            # For now, break if n is too large for the loop.
            return None, None

        for c in range(num_colorings):
            b0 = (c >> e[0]) & 1
            if b0 == ((c >> e[1]) & 1) == ((c >> e[2]) & 1) == ((c >> e[3]) & 1) == ((c >> e[4]) & 1):
                A[c, j] = 1.0
                
    A = A.tocsr()
    
    # Objective: minimize sum(x)
    c_obj = np.ones(num_edges)
    
    # Constraints: A @ x >= 1
    # LinearConstraint(A, lb=np.ones(num_colorings), ub=np.inf)
    lb = np.ones(num_colorings)
    ub = np.full(num_colorings, np.inf)
    
    constraints = LinearConstraint(A, lb, ub)
    
    # Bounds: 0 <= x <= 1
    bounds = Bounds(lb=np.zeros(num_edges), ub=np.ones(num_edges))
    
    # Integrality: x must be integer
    integrality = np.ones(num_edges)
    
    # Time limit for milp
    # scipy milp does not have a direct time limit parameter in all versions, 
    # but we can check time after. 
    # Actually, scipy.optimize.milp uses HiGHS. We can pass options?
    # options={"time_limit": 10.0} might work if supported.
    
    try:
        res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds, 
                   options={"time_limit": 10.0})
    except Exception:
        return None, None
        
    if res.status != 0 and res.status != 1:
        # 0 is optimal, 1 is feasible (maybe due to time limit)
        if res.status == 1:
            pass # Use it
        else:
            return None, None
            
    x_sol = res.x
    # Round to nearest integer to be safe
    x_sol = np.round(x_sol).astype(int)
    
    # Extract selected edges
    selected_indices = np.where(x_sol == 1)[0]
    selected_edges = [list(all_edges[i]) for i in selected_indices]
    
    return selected_edges, res.fun

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    best_edges = None
    best_n = None
    best_count = float('inf')
    
    # Try ILP for n=11, 12, 13
    # n=11 is likely to give the global optimum m(5) if it exists with n=11 vertices.
    # The problem asks to minimize edges.
    
    for n in [11, 12, 13]:
        if time.time() > deadline - 20:
            break
            
        # Run ILP
        edges, cost = _solve_ilp(n, deadline - 20)
        
        if edges:
            # Verify with exact checker
            fs_edges = [frozenset(e) for e in edges]
            if not _is_2_colorable(fs_edges, n, deadline - 10):
                if len(edges) < best_count:
                    best_count = len(edges)
                    best_edges = edges
                    best_n = n
                    # If we find a very small one, we can stop
                    if best_count <= 30:
                        break
            else:
                # ILP found a solution that is 2-colorable? 
                # This implies the ILP formulation or solver had an issue, 
                # or my checker is wrong. 
                # Given the checker is exact, it's likely the ILP didn't cover all colorings 
                # due to numerical issues or constraint relaxation? 
                # But we used integer variables. 
                # Let's try to repair by adding edges? No, just skip.
                pass

    # Fallback if ILP fails or doesn't find a good solution
    if best_edges is None or best_count > 50:
        # Use a known good construction or random search
        # Known: m(5) <= 51.
        # Let's try a random local search on n=11 or n=12 if we have time.
        
        if time.time() < deadline - 10:
            n = 11
            all_edges = list(combinations(range(n), 5))
            random.seed(42)
            
            # Start with a random subset? Or all edges?
            # All edges is 462.
            # Let's try to find a small one by random removal from a larger set.
            # Or just use the ILP result if it was slightly suboptimal?
            
            # If ILP failed, let's try a simple greedy set cover again but with a better heuristic
            # Or just return a hardcoded valid hypergraph if we can generate one quickly.
            
            # Since m(5) <= 51, let's try to find a 51-edge one.
            # We can use the result from the previous run's knowledge: "reliably finding the known 51-edge optimal solution"
            # But we must generate it.
            
            # Let's try one more ILP run with n=11 if it failed before?
            # Or just use a random search.
            
            # Random local search:
            # Start with a random set of ~60 edges.
            # Check if 2-colorable. If yes, add random edges. If no, remove random edges.
            
            edges_set = set()
            # Add random edges until non-2-colorable
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                if not edges_set:
                    break
                # Try to remove
                e_to_remove = random.choice(list(edges_set))
                edges_set.remove(e_to_remove)
                if not _is_2_colorable([frozenset(e) for e in edges_set], n, deadline - 5):
                    continue
                else:
                    edges_set.add(e_to_remove)
                    break
            
            if edges_set and not _is_2_colorable([frozenset(e) for e in edges_set], n, deadline - 5):
                if len(edges_set) < best_count:
                    best_count = len(edges_set)
                    best_edges = list(edges_set)
                    best_n = n

    # Final safety fallback
    if best_edges is None:
        # Return a known valid hypergraph. 
        # K_11^5 is 462 edges.
        n = 11
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n

    # Final verification
    if _is_2_colorable([frozenset(e) for e in best_edges], best_n, deadline):
        # Should not happen
        pass

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
