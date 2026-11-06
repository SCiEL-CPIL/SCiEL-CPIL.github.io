import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    N = 180
    elements = list(range(1, N + 1))
    
    # 1. Precompute all forbidden triples
    triples = []
    elem_to_triples = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    t = (a, b, c)
                    idx = len(triples)
                    triples.append(t)
                    for x in t:
                        elem_to_triples[x].append(idx)
                        
    # 2. Try ILP approach
    best_set = set()
    best_size = 0
    
    try:
        # Variables x[0] ... x[179] correspond to elements 1 ... 180
        n_vars = N
        c_obj = -np.ones(n_vars)  # Minimize -sum(x) == Maximize sum(x)
        
        # Constraints: For each triple, sum of x in triple <= 2
        # A_ub is sparse matrix
        import scipy.sparse as sp
        
        rows = []
        cols = []
        data = []
        
        for idx, (a, b, c) in enumerate(triples):
            # Convert 1-based to 0-based
            a0, b0, c0 = a-1, b-1, c-1
            rows.extend([idx, idx, idx])
            cols.extend([a0, b0, c0])
            data.extend([1, 1, 1])
            
        if rows:
            A_ub = sp.csr_matrix((data, (rows, cols)), shape=(len(triples), n_vars))
            b_ub = np.full(len(triples), 2)
        else:
            # No triples, take all
            return list(elements)
            
        # Bounds
        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)
        
        # Integrality
        integrality = np.ones(n_vars)  # All integer
        
        # Solver options
        # 'highs' is the default and robust solver
        res = milp(
            c=c_obj,
            constraints=LinearConstraint(A_ub, ub=b_ub),
            integrality=integrality,
            bounds=Bounds(lb, ub),
            options={"time_limit": 40, "mip_rel_gap": 0.01} # 1% gap, 40s limit
        )
        
        if res.success and res.x is not None:
            # res.x is the solution vector
            # Rounding might be needed if not strictly integer, but integrality=1 ensures it
            sol = np.round(res.x).astype(int)
            # Filter for 1s
            selected = [i+1 for i in range(n_vars) if sol[i] == 1]
            best_set = set(selected)
            best_size = len(best_set)
            
    except Exception as e:
        pass # Fallback to heuristic if ILP fails or times out badly

    # 3. Heuristic Fallback / Polish
    # If ILP didn't find a good solution or failed, use the previous greedy + local search
    # We can also use the ILP solution as a seed for local search if it exists
    
    def can_add(S, x):
        for idx in elem_to_triples[x]:
            a, b, c = triples[idx]
            others = [y for y in (a, b, c) if y != x]
            if others[0] in S and others[1] in S:
                return False
        return True

    def local_search(S, time_budget=5.0):
        S = set(S)
        removed = set(elements) - S
        removed_list = list(removed)
        random.shuffle(removed_list)
        
        # First pass: add any element that can be added directly
        for x in removed_list:
            if time.time() > start_time + time_budget:
                break
            if x in S:
                continue
            if can_add(S, x):
                S.add(x)
        
        # Second pass: try 1-for-1 swaps
        improved = True
        while improved and time.time() < start_time + time_budget:
            improved = False
            removed = list(set(elements) - S)
            random.shuffle(removed)
            for x in removed:
                if time.time() > start_time + time_budget:
                    break
                conflicting_ys = set()
                for idx in elem_to_triples[x]:
                    a, b, c = triples[idx]
                    others = [y for y in (a, b, c) if y != x]
                    if others[0] in S:
                        conflicting_ys.add(others[0])
                    if others[1] in S:
                        conflicting_ys.add(others[1])
                
                if not conflicting_ys:
                    continue
                
                for y in list(conflicting_ys):
                    if time.time() > start_time + time_budget:
                        break
                    S.remove(y)
                    if can_add(S, x):
                        S.add(x)
                        improved = True
                        # Try to add back y or other elements
                        for z in list(set(elements) - S):
                            if time.time() > start_time + time_budget:
                                break
                            if z in S:
                                continue
                            if can_add(S, z):
                                S.add(z)
                    else:
                        S.add(y)
                        break
        
        return S

    # If ILP gave a result, polish it
    if best_size > 0:
        remaining_time = deadline - time.time() - 1
        if remaining_time > 2:
            S = local_search(best_set, time_budget=min(remaining_time, 10.0))
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
    else:
        # Fallback to pure heuristic if ILP failed
        def run_greedy(randomize_ties=False):
            active_elements = set(elements)
            active_triples = set(range(len(triples)))
            conflict_count = {i: 0 for i in range(1, N + 1)}
            
            for idx in active_triples:
                for x in triples[idx]:
                    conflict_count[x] += 1
            
            while time.time() < deadline:
                max_conflict = -1
                candidates = []
                for x in active_elements:
                    if conflict_count[x] > max_conflict:
                        max_conflict = conflict_count[x]
                        candidates = [x]
                    elif conflict_count[x] == max_conflict:
                        candidates.append(x)
                
                if max_conflict == 0:
                    break
                
                if randomize_ties and len(candidates) > 1:
                    victim = random.choice(candidates)
                else:
                    victim = candidates[0]

                active_elements.remove(victim)
                
                for idx in elem_to_triples[victim]:
                    if idx in active_triples:
                        active_triples.remove(idx)
                        for x in triples[idx]:
                            if x != victim and x in active_elements:
                                conflict_count[x] -= 1
            
            return active_elements

        # Run a few greedy passes
        for i in range(50):
            if time.time() > deadline - 5:
                break
            s = run_greedy(randomize_ties=True)
            if len(s) > best_size:
                best_size = len(s)
                best_set = s
        
        # Local search on best
        remaining_time = deadline - time.time() - 1
        if remaining_time > 2:
            S = local_search(best_set, time_budget=min(remaining_time, 10.0))
            if len(S) > best_size:
                best_size = len(S)
                best_set = S

    return list(best_set)
