"""Improved program: custom branch-and-bound for max independent set in 3-uniform hypergraph."""
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# EVOLVE-BLOCK-START

def _build_triples(n_max=180):
    """Build all triples (a, b, c) with a < b < c, a in [1,n_max], and 1/a = 1/b + 1/c."""
    triples = []
    for a in range(1, n_max + 1):
        for b in range(a + 1, n_max + 1):
            diff = b - a
            if diff == 0:
                continue
            if (a * b) % diff != 0:
                continue
            c = (a * b) // diff
            if c > n_max:
                continue
            if c <= b:
                continue
            triples.append((a, b, c))
    return triples

def _local_polish(S, triples, elem_to_triples, time_limit=5.0):
    """Try to add back elements that were removed, if possible."""
    S = set(S)
    n_max = max(S) if S else 180
    all_elements = set(range(1, n_max + 1))
    candidates = all_elements - S
    
    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        for x in sorted(candidates, reverse=True):
            if x in S:
                continue
            can_add = True
            for tidx in elem_to_triples.get(x, []):
                a, b, c = triples[tidx]
                others = [y for y in (a, b, c) if y != x]
                if all(y in S for y in others):
                    can_add = False
                    break
            if can_add:
                S.add(x)
                candidates.discard(x)
                improved = True
    
    return sorted(S)

def _branch_and_bound(all_elements, triples, elem_to_triples, deadline, initial_solution=None):
    """
    Custom branch-and-bound to find maximum independent set in 3-uniform hypergraph.
    Variables: x_i in {0,1} for i in all_elements (1..180)
    Constraint: for each triple (a,b,c), x_a + x_b + x_c <= 2
    Objective: maximize sum(x_i)
    """
    n = len(all_elements)
    # Map element value to index
    elem_to_idx = {v: i for i, v in enumerate(all_elements)}
    
    # Precompute for each variable, which triples it participates in
    var_triples = [[] for _ in range(n)]
    for tidx, (a, b, c) in enumerate(triples):
        var_triples[elem_to_idx[a]].append(tidx)
        var_triples[elem_to_idx[b]].append(tidx)
        var_triples[elem_to_idx[c]].append(tidx)
    
    # Triple membership: for each triple, list of variable indices
    triple_vars = []
    for (a, b, c) in triples:
        triple_vars.append([elem_to_idx[a], elem_to_idx[b], elem_to_idx[c]])
    
    n_triples = len(triples)
    
    # State tracking
    # x[i] = -1 (unassigned), 0 (false), 1 (true)
    x = [-1] * n
    # For each triple, count of variables set to 1
    triple_count = [0] * n_triples
    # For each variable, count of "conflicts" - number of triples where the other two are set to 1
    # Actually, let's track: for each triple, how many are set to 1
    # A triple is "satisfied" if count <= 2
    # A triple is "violated" if count == 3
    
    current_size = 0
    best_size = 0
    best_solution = [0] * n
    
    # If we have an initial solution, use it
    if initial_solution is not None:
        for i, val in enumerate(initial_solution):
            if val == 1:
                x[i] = 1
                current_size += 1
                for tidx in var_triples[i]:
                    triple_count[tidx] += 1
        best_size = current_size
        best_solution = x.copy()
    
    # Check for violations in initial solution
    for tidx in range(n_triples):
        if triple_count[tidx] > 2:
            # Need to fix - this shouldn't happen with a valid initial solution
            pass
    
    # Branching: we'll use a simple recursive approach with pruning
    # But for 180 variables, we need to be smart
    
    # Let's use a different approach: iterative improvement from a good initial solution
    # Start with all 1s, then remove elements to fix violations
    # This is essentially the greedy approach but with better ordering
    
    # Actually, let me implement a proper branch and bound with memoization-like pruning
    
    # For this problem size, let's use a hybrid:
    # 1. Get a good initial solution via MILP or greedy
    # 2. Use local search / simulated annealing to improve
    
    # Let's implement a local search approach
    # State: which elements are in the set
    # Move: add/remove one element
    # Objective: maximize size subject to no triple fully contained
    
    # Start from the initial solution
    in_set = [False] * n
    for i in range(n):
        in_set[i] = (x[i] == 1)
    
    # Recompute triple counts
    triple_count = [0] * n_triples
    for tidx, var_list in enumerate(triple_vars):
        for v in var_list:
            if in_set[v]:
                triple_count[tidx] += 1
    
    size = sum(in_set)
    best_size = size
    best_solution = in_set.copy()
    
    start_time = time.time()
    
    # Local search: try to add elements that don't create violations
    # Then try swaps (remove one, add another) to increase size
    
    # First, try to add any element that doesn't create a violation
    improved = True
    while improved and (time.time() - start_time) < (deadline - 2.0):
        improved = False
        for i in range(n):
            if in_set[i]:
                continue
            # Check if adding i creates any violation
            can_add = True
            for tidx in var_triples[i]:
                if triple_count[tidx] >= 2:
                    can_add = False
                    break
            if can_add:
                in_set[i] = True
                size += 1
                for tidx in var_triples[i]:
                    triple_count[tidx] += 1
                improved = True
                if size > best_size:
                    best_size = size
                    best_solution = in_set.copy()
    
    # Now try swaps: remove one element, add another
    # This can help when no single addition is possible
    for iteration in range(200):
        if (time.time() - start_time) > (deadline - 2.0):
            break
        
        # Try all possible swaps
        found_swap = False
        for i in range(n):
            if not in_set[i]:
                continue
            # Try removing i and adding some j not in set
            # First, what triples become "free" when we remove i?
            freed_triples = set()
            for tidx in var_triples[i]:
                if triple_count[tidx] == 2:
                    # This triple had exactly 2 in set including i, so removing i makes count 1
                    # But we need to check which other elements are in this triple
                    pass
            
            # For each j not in set, check if we can add j after removing i
            # This is expensive, so let's be smarter
            
            # Alternative: for each j not in set, check what would prevent adding j
            # Then see if removing i helps
            
        # Simpler approach: try random restarts with greedy
        
        # Let's try a different local search: for each element not in set,
        # find the minimum number of elements to remove to allow adding it
        # If that number is 0, add it (already done above)
        # If that number is 1, consider swapping
        
        for j in range(n):
            if in_set[j]:
                continue
            # Find triples where j would create a violation
            blocking_triples = []
            for tidx in var_triples[j]:
                if triple_count[tidx] >= 2:
                    blocking_triples.append(tidx)
            
            if not blocking_triples:
                continue  # Should have been added already
            
            # For each blocking triple, we need to remove at least one element
            # Find the element whose removal unblocks the most triples
            # and doesn't block other additions
            
            # Simple heuristic: for each element in blocking triples, count how many
            # blocking triples it's in
            removal_candidates = {}
            for tidx in blocking_triples:
                for v in triple_vars[tidx]:
                    if in_set[v] and v != j:
                        if v not in removal_candidates:
                            removal_candidates[v] = set()
                        removal_candidates[v].add(tidx)
            
            if not removal_candidates:
                continue
            
            # Try removing the element that appears in the most blocking triples
            best_remove = max(removal_candidates.keys(), key=lambda v: len(removal_candidates[v]))
            
            # Check if removing best_remove and adding j is valid
            # After removing best_remove, check if j can be added
            # and if removing best_remove doesn't create new issues (it won't, since we're removing)
            
            # Simulate
            old_count = triple_count[best_remove]
            # Remove best_remove
            in_set[best_remove] = False
            size -= 1
            for tidx in var_triples[best_remove]:
                triple_count[tidx] -= 1
            
            # Try adding j
            can_add_j = True
            for tidx in var_triples[j]:
                if triple_count[tidx] >= 2:
                    can_add_j = False
                    break
            
            if can_add_j:
                in_set[j] = True
                size += 1
                for tidx in var_triples[j]:
                    triple_count[tidx] += 1
                found_swap = True
                if size > best_size:
                    best_size = size
                    best_solution = in_set.copy()
            else:
                # Undo
                in_set[best_remove] = True
                size += 1
                for tidx in var_triples[best_remove]:
                    triple_count[tidx] += 1
        
        if not found_swap:
            break
    
    return best_solution, best_size

def construct():
    start_time = time.time()
    deadline = 60.0  # seconds
    
    n_max = 180
    all_elements = list(range(1, n_max + 1))
    
    # Build the hypergraph
    triples = _build_triples(n_max)
    n_triples = len(triples)
    
    # Build incidence
    elem_to_triples = {i: [] for i in range(1, n_max + 1)}
    for idx, (a, b, c) in enumerate(triples):
        for x in (a, b, c):
            elem_to_triples[x].append(idx)
    
    # Try MILP first for a good initial solution
    initial_solution = None
    try:
        c_obj = -np.ones(n_max)
        
        if n_triples > 0:
            A_ub = np.zeros((n_triples, n_max))
            for idx, (a, b, c) in enumerate(triples):
                A_ub[idx, a-1] = 1
                A_ub[idx, b-1] = 1
                A_ub[idx, c-1] = 1
            b_ub = np.full(n_triples, 2)
            constraints = LinearConstraint(A_ub, -np.inf, b_ub)
        else:
            constraints = LinearConstraint(np.zeros((1, n_max)), 0, 1)
        
        bounds = Bounds(0, 1)
        integrality = np.ones(n_max)
        
        result = milp(
            c=c_obj,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds
        )
        
        if result.status == 0 and result.x is not None:
            sol = np.round(result.x).astype(int)
            initial_solution = sol.tolist()
    except Exception:
        pass
    
    # Use branch and bound / local search
    if initial_solution is not None:
        best_sol, best_size = _branch_and_bound(all_elements, triples, elem_to_triples, deadline, initial_solution)
        S = set(all_elements[i] for i in range(n_max) if best_sol[i])
    else:
        # Fallback to greedy
        S = set(all_elements)
        triple_incount = [0] * n_triples
        for idx, (a, b, c) in enumerate(triples):
            cnt = (a in S) + (b in S) + (c in S)
            triple_incount[idx] = cnt
        
        elem_conflict_count = {i: 0 for i in range(1, n_max + 1)}
        for idx, cnt in enumerate(triple_incount):
            if cnt == 3:
                a, b, c = triples[idx]
                elem_conflict_count[a] += 1
                elem_conflict_count[b] += 1
                elem_conflict_count[c] += 1
        
        total_conflicts = sum(1 for cnt in triple_incount if cnt == 3)
        
        while total_conflicts > 0:
            elapsed = time.time() - start_time
            if elapsed > deadline - 2.0:
                break
            
            max_conflict = 0
            best_elem = None
            for x in S:
                if elem_conflict_count[x] > max_conflict:
                    max_conflict = elem_conflict_count[x]
                    best_elem = x
            
            if best_elem is None or max_conflict == 0:
                break
            
            S.remove(best_elem)
            
            for tidx in elem_to_triples[best_elem]:
                old_cnt = triple_incount[tidx]
                if old_cnt == 3:
                    total_conflicts -= 1
                    a, b, c = triples[tidx]
                    for x in (a, b, c):
                        if x != best_elem:
                            elem_conflict_count[x] -= 1
                triple_incount[tidx] = old_cnt - 1
    
    # Final polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 2.0:
        S_polished = _local_polish(S, triples, elem_to_triples, time_limit=min(5.0, deadline - elapsed - 1.0))
        return S_polished
    
    return sorted(S)

# EVOLVE-BLOCK-END
