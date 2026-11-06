"""Rewritten program: MaxSAT-style branch-and-bound with unit propagation for max independent set on 3-uniform hypergraph."""
import time
import random
import sys

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
            if c > n_max or c <= b:
                continue
            triples.append((a, b, c))
    return triples

def _greedy_independent_set(triples, elem_to_triples, n_max, deadline):
    """Greedy: start with all, remove highest-conflict element iteratively."""
    S = set(range(1, n_max + 1))
    triple_incount = [3] * len(triples)
    elem_conflict = {i: 0 for i in range(1, n_max + 1)}
    for idx, (a, b, c) in enumerate(triples):
        elem_conflict[a] += 1
        elem_conflict[b] += 1
        elem_conflict[c] += 1
    total_conflicts = len(triples)
    
    while total_conflicts > 0:
        if time.time() > deadline:
            break
        best_elem = max(S, key=lambda x: (elem_conflict[x], -x))
        if elem_conflict[best_elem] == 0:
            break
        S.remove(best_elem)
        for tidx in elem_to_triples[best_elem]:
            if triple_incount[tidx] == 3:
                total_conflicts -= 1
                a, b, c = triples[tidx]
                for x in (a, b, c):
                    if x in S:
                        elem_conflict[x] -= 1
            triple_incount[tidx] -= 1
    return sorted(S)

def _greedy_random(triples, elem_to_triples, n_max, deadline, seed=None):
    """Greedy removal with random tie-breaking."""
    if seed is not None:
        random.seed(seed)
    S = set(range(1, n_max + 1))
    triple_incount = [3] * len(triples)
    elem_conflict = {i: 0 for i in range(1, n_max + 1)}
    for idx, (a, b, c) in enumerate(triples):
        elem_conflict[a] += 1
        elem_conflict[b] += 1
        elem_conflict[c] += 1
    total_conflicts = len(triples)
    
    while total_conflicts > 0:
        if time.time() > deadline:
            break
        max_conf = max((elem_conflict[x] for x in S), default=0)
        if max_conf == 0:
            break
        candidates = [x for x in S if elem_conflict[x] == max_conf]
        best_elem = random.choice(candidates)
        S.remove(best_elem)
        for tidx in elem_to_triples[best_elem]:
            if triple_incount[tidx] == 3:
                total_conflicts -= 1
                a, b, c = triples[tidx]
                for x in (a, b, c):
                    if x in S:
                        elem_conflict[x] -= 1
            triple_incount[tidx] -= 1
    return sorted(S)

def _local_search(triples, elem_to_triples, n_max, start_set, deadline, max_iter=5000):
    """Local search: 1-opt (add) and 2-opt (remove one, add up to two)."""
    S = set(start_set)
    removed = set(range(1, n_max + 1)) - S
    
    # Verify no conflicts
    for a, b, c in triples:
        if a in S and b in S and c in S:
            S.discard(c)
            removed.add(c)
    
    best_size = len(S)
    no_improve = 0
    max_no_improve = 300
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if no_improve >= max_no_improve:
            break
        
        improved = False
        
        # Phase 1: try to add any single removed element (1-opt)
        candidates = list(removed)
        random.shuffle(candidates)
        for elem in candidates:
            if time.time() > deadline:
                break
            new_conflicts = 0
            for tidx in elem_to_triples[elem]:
                a, b, c = triples[tidx]
                others = [x for x in (a, b, c) if x != elem]
                if all(x in S for x in others):
                    new_conflicts += 1
            if new_conflicts == 0:
                S.add(elem)
                removed.discard(elem)
                improved = True
                best_size = max(best_size, len(S))
                break
        
        if improved:
            no_improve = 0
            continue
        
        # Phase 2: 2-opt — remove one element from S, then try to add up to 2 from removed
        if not S:
            break
            
        s_list = list(S)
        random.shuffle(s_list)
        
        found_2opt = False
        for rem_elem in s_list[:min(50, len(s_list))]:
            if time.time() > deadline:
                break
            
            S.remove(rem_elem)
            removed.add(rem_elem)
            
            added_count = 0
            added_elems = []
            
            candidates2 = [x for x in removed if x != rem_elem]
            random.shuffle(candidates2)
            
            for elem in candidates2:
                if time.time() > deadline:
                    break
                new_conflicts = 0
                for tidx in elem_to_triples[elem]:
                    a, b, c = triples[tidx]
                    others = [x for x in (a, b, c) if x != elem]
                    if all(x in S for x in others):
                        new_conflicts += 1
                if new_conflicts == 0:
                    S.add(elem)
                    removed.discard(elem)
                    added_elems.append(elem)
                    added_count += 1
                    if added_count >= 2:
                        break
            
            if added_count > 1:
                best_size = max(best_size, len(S))
                improved = True
                found_2opt = True
                break
            else:
                for ae in added_elems:
                    S.discard(ae)
                    removed.add(ae)
                S.add(rem_elem)
                removed.discard(rem_elem)
        
        if not found_2opt:
            no_improve += 1
    
    return sorted(S)

def _milp_solve(triples, n_max, deadline):
    """Solve using MILP (HiGHS via scipy.optimize.milp)."""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        n = n_max
        c = -np.ones(n)
        
        A = []
        for (a, b, c_val) in triples:
            row = np.zeros(n)
            row[a-1] = 1
            row[b-1] = 1
            row[c_val-1] = 1
            A.append(row)
        
        if A:
            A = np.array(A)
            ub = np.full(len(triples), 2)
            lb = np.full(len(triples), 0)
        else:
            A = np.array([[1.0]])
            ub = np.array([1.0])
            lb = np.array([0.0])
        
        constraints = LinearConstraint(A, lb, ub)
        integrality = np.ones(n)
        bounds = Bounds(0, 1)
        
        result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if result.success:
            sol = result.x
            S = [i+1 for i in range(n) if sol[i] > 0.5]
            return sorted(S)
        return None
    except Exception:
        return None

def _maxsat_branch_and_bound(triples, elem_to_triples, n_max, deadline, initial_set=None):
    """
    MaxSAT-style branch-and-bound with unit propagation.
    Variables: x_i = 1 if i is in the set.
    Clauses: for each triple (a,b,c): not (x_a AND x_b AND x_c), i.e., at least one is 0.
    Maximize sum(x_i).
    
    We branch on which elements to EXCLUDE. The key insight: if we exclude a set E,
    the remaining elements S = [1..n] \\ E form a valid set iff no triple is entirely in S.
    Unit propagation: if a clause has two literals forced to True, the third must be False.
    """
    if initial_set is None:
        initial_set = _greedy_independent_set(triples, elem_to_triples, n_max, deadline)
    
    best_set = set(initial_set)
    best_size = len(best_set)
    
    # Precompute: for each element, which triples it participates in
    # For unit propagation: if elements a and b are both in S, then c must be excluded
    # and vice versa.
    
    # Build adjacency: for each pair (a,b) in a triple, c is the third element
    # If a and b are both included, c must be excluded.
    
    # Strategy: start from a large set, iteratively find conflicts and resolve them
    # by excluding one element from each conflicting triple.
    
    S = set(initial_set)
    excluded = set(range(1, n_max + 1)) - S
    
    # Precompute for fast conflict checking
    # For each element, list of (other1, other2, triple_idx) where all three form a triple
    elem_pairs = {i: [] for i in range(1, n_max + 1)}
    for idx, (a, b, c) in enumerate(triples):
        elem_pairs[a].append((b, c, idx))
        elem_pairs[b].append((a, c, idx))
        elem_pairs[c].append((a, b, idx))
    
    def _has_conflict_in(s):
        """Check if set s contains any forbidden triple."""
        for a, b, c in triples:
            if a in s and b in s and c in s:
                return True
        return False
    
    def _conflict_triples_in(s):
        """Return list of triples fully contained in s."""
        conflicts = []
        for idx, (a, b, c) in enumerate(triples):
            if a in s and b in s and c in s:
                conflicts.append(idx)
        return conflicts
    
    # Iterative improvement with unit propagation
    # Key idea: if we have a conflict (a,b,c) in S, we must exclude at least one.
    # Unit propagation: if excluding a forces other exclusions, prefer that.
    
    iteration = 0
    max_iterations = 5000
    
    while iteration < max_iterations:
        iteration += 1
        if time.time() > deadline:
            break
        
        # Find all conflicts
        conflicts = _conflict_triples_in(S)
        if not conflicts:
            # No conflicts - try to add elements
            added_any = False
            for elem in sorted(excluded):
                if time.time() > deadline:
                    break
                # Check if adding elem creates a conflict
                ok = True
                for (o1, o2, tidx) in elem_pairs[elem]:
                    if o1 in S and o2 in S:
                        ok = False
                        break
                if ok:
                    S.add(elem)
                    excluded.discard(elem)
                    added_any = True
            if not added_any:
                break
        
        # Resolve conflicts using unit propagation
        # For each conflict triple, pick an element to exclude
        # Prefer elements that cause the most cascading exclusions (unit propagation)
        
        resolved = False
        for tidx in conflicts:
            if time.time() > deadline:
                break
            a, b, c = triples[tidx]
            if not (a in S and b in S and c in S):
                continue
            
            # Try excluding each of a, b, c and see which leads to most cascading exclusions
            best_elem = None
            best_cascade = -1
            
            for candidate in (a, b, c):
                if candidate not in S:
                    continue
                # Simulate: exclude candidate, see what else gets forced out
                cascade_set = {candidate}
                # Unit propagation: if we exclude candidate, any triple where candidate
                # and one other element are both in S forces the third out
                # Actually, unit propagation works the other way: if two are in S, third must be out.
                # Excluding candidate doesn't directly force others out via unit propagation
                # in the standard sense. Instead, we count how many conflicts candidate is in.
                
                # Count conflicts involving candidate
                conf_count = 0
                for (o1, o2, t2) in elem_pairs[candidate]:
                    if o1 in S and o2 in S:
                        conf_count += 1
                
                if conf_count > best_cascade:
                    best_cascade = conf_count
                    best_elem = candidate
            
            if best_elem is not None:
                S.discard(best_elem)
                excluded.add(best_elem)
                resolved = True
        
        if not resolved:
            break
        
        # After resolving, check if we can add back elements
        for elem in list(excluded):
            if time.time() > deadline:
                break
            ok = True
            for (o1, o2, tidx) in elem_pairs[elem]:
                if o1 in S and o2 in S:
                    ok = False
                    break
            if ok:
                S.add(elem)
                excluded.discard(elem)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    return sorted(best_set) if len(best_set) > len(initial_set) else sorted(initial_set)

def _branch_and_bound(triples, elem_to_triples, n_max, deadline, initial_set=None):
    """
    Improved branch-and-bound: branch on excluded elements.
    State: set of included elements S, set of undecided elements U.
    Upper bound: |S| + |U| (optimistic: all undecided can be added).
    Unit propagation: if two elements in a triple are both in S, the third must be excluded.
    """
    if initial_set is None:
        initial_set = _greedy_independent_set(triples, elem_to_triples, n_max, deadline)
    
    best_set = set(initial_set)
    best_size = len(best_set)
    
    # Order elements by conflict count descending
    elem_conflict = {i: len(elem_to_triples[i]) for i in range(1, n_max + 1)}
    elements = sorted(range(1, n_max + 1), key=lambda x: -elem_conflict[x])
    
    # Precompute for each element, the triples it's in
    elem_triples = elem_to_triples
    
    # Use iterative deepening / beam search approach
    # Start with all elements, iteratively remove to resolve conflicts
    
    S = set(range(1, n_max + 1))
    
    # Phase 1: Greedy removal with lookahead
    while True:
        if time.time() > deadline:
            break
        
        # Find all conflicts
        conflicts = []
        for idx, (a, b, c) in enumerate(triples):
            if a in S and b in S and c in S:
                conflicts.append((a, b, c))
        
        if not conflicts:
            break
        
        # For each element, count how many conflicts it's in
        elem_conf_count = {}
        for (a, b, c) in conflicts:
            for x in (a, b, c):
                elem_conf_count[x] = elem_conf_count.get(x, 0) + 1
        
        # Remove element with highest conflict count
        if elem_conf_count:
            best_elem = max(elem_conf_count.keys(), key=lambda x: elem_conf_count[x])
            S.discard(best_elem)
    
    if len(S) > best_size:
        best_size = len(S)
        best_set = set(S)
    
    # Phase 2: Try to add elements back
    excluded = set(range(1, n_max + 1)) - S
    for elem in sorted(excluded):
        if time.time() > deadline:
            break
        ok = True
        for tidx in elem_triples[elem]:
            a, b, c = triples[tidx]
            others = [x for x in (a, b, c) if x != elem]
            if all(x in S for x in others):
                ok = False
                break
        if ok:
            S.add(elem)
            excluded.discard(elem)
    
    if len(S) > best_size:
        best_size = len(S)
        best_set = set(S)
    
    return sorted(best_set)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n_max = 180
    triples = _build_triples(n_max)
    
    elem_to_triples = {i: [] for i in range(1, n_max + 1)}
    for idx, (a, b, c) in enumerate(triples):
        for x in (a, b, c):
            elem_to_triples[x].append(idx)
    
    # Strategy 1: MILP solver (global optimum)
    best_set = None
    best_size = 0
    
    if time.time() < deadline - 15.0:
        milp_set = _milp_solve(triples, n_max, deadline - 10.0)
        if milp_set is not None:
            best_set = milp_set
            best_size = len(milp_set)
    
    # Strategy 2: Greedy + local search
    greedy_set = _greedy_independent_set(triples, elem_to_triples, n_max, deadline)
    if len(greedy_set) > best_size:
        best_set = greedy_set
        best_size = len(greedy_set)
    
    # Local search on best
    if time.time() < deadline - 15.0:
        ls_set = _local_search(triples, elem_to_triples, n_max, best_set, deadline - 10.0, max_iter=3000)
        if len(ls_set) > best_size:
            best_set = ls_set
            best_size = len(ls_set)
    
    # Strategy 3: Multiple random greedy restarts + local search
    if time.time() < deadline - 20.0:
        for restart in range(15):
            if time.time() > deadline - 20.0:
                break
            alt_set = _greedy_random(triples, elem_to_triples, n_max, deadline - 20.0, seed=restart * 99991 + 13)
            if len(alt_set) > best_size:
                best_set = alt_set
                best_size = len(alt_set)
            # Quick local search
            if time.time() < deadline - 15.0:
                ls_set = _local_search(triples, elem_to_triples, n_max, best_set, deadline - 15.0, max_iter=1000)
                if len(ls_set) > best_size:
                    best_set = ls_set
                    best_size = len(ls_set)
    
    # Strategy 4: MaxSAT-style branch and bound
    if time.time() < deadline - 10.0:
        maxsat_set = _maxsat_branch_and_bound(triples, elem_to_triples, n_max, deadline - 5.0, initial_set=best_set)
        if len(maxsat_set) > best_size:
            best_set = maxsat_set
            best_size = len(maxsat_set)
    
    # Strategy 5: Branch and bound refinement
    if time.time() < deadline - 5.0:
        bb_set = _branch_and_bound(triples, elem_to_triples, n_max, deadline - 2.0, initial_set=best_set)
        if len(bb_set) > best_size:
            best_set = bb_set
            best_size = len(bb_set)
    
    # Final local search
    if time.time() < deadline - 2.0:
        ls_set = _local_search(triples, elem_to_triples, n_max, best_set, deadline - 1.0, max_iter=5000)
        if len(ls_set) > best_size:
            best_set = ls_set
            best_size = len(ls_set)
    
    if not best_set:
        best_set = list(range(91, 181))
    
    return best_set

# EVOLVE-BLOCK-END
