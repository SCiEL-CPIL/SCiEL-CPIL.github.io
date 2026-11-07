import time
import random
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def _subset_sums(S):
    """Compute all non-empty subset sums as a set."""
    n = len(S)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += S[i]
        sums.add(s)
    return sums

def _check_valid(sums_set):
    """Check no two distinct sums have divisibility relation."""
    total = max(sums_set) if sums_set else 0
    sums_list = sorted(sums_set)
    for s in sums_list:
        if s == 0:
            continue
        for mult in range(2, total // s + 1):
            if s * mult in sums_set:
                return False
    return True

def _is_valid(S):
    if not S:
        return True
    sums = _subset_sums(S)
    return _check_valid(sums)

def _greedy_build(pool, order):
    """Greedy construction with given ordering."""
    current = []
    sums = set()
    total = 0
    for x in order:
        new_sums = set(sums)
        for s in sums:
            new_sums.add(s + x)
        new_sums.add(x)
        new_total = total + x
        if _check_valid(new_sums):
            current.append(x)
            sums = new_sums
            total = new_total
    return current

def _local_search(S, pool, deadline):
    """Hill-climbing with swap and add moves."""
    current = list(S)
    cur_sums = _subset_sums(current)
    cur_total = sum(current)
    used = set(current)
    best = current[:]
    best_size = len(current)
    
    while time.time() < deadline:
        improved = False
        available = [x for x in pool if x not in used]
        random.shuffle(available)
        
        # Try additions
        for x in available[:50]:
            new_sums = set(cur_sums)
            for s in cur_sums:
                new_sums.add(s + x)
            new_sums.add(x)
            new_total = cur_total + x
            if _check_valid(new_sums):
                current.append(x)
                cur_sums = new_sums
                cur_total = new_total
                used.add(x)
                improved = True
                if len(current) > best_size:
                    best_size = len(current)
                    best = current[:]
                break
        
        # Try swaps
        if not improved:
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                old_x = current[i]
                for x in available[:30]:
                    new_set = current[:]
                    new_set[i] = x
                    new_sums = _subset_sums(new_set)
                    new_total = sum(new_set)
                    if _check_valid(new_sums):
                        current = new_set
                        cur_sums = new_sums
                        cur_total = new_total
                        used.discard(old_x)
                        used.add(x)
                        available = [y for y in pool if y not in used]
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # Try random restarts within local search
            if random.random() < 0.1:
                break
    
    return best

def _solve_ilp(pool, deadline, target_size):
    """
    Formulate as ILP:
    - x_i = 1 if element i is included
    - Maximize sum(x_i)
    - Subject to: for every pair of subset sums (s, t) where s|t, 
      at most one of the subsets can be "active"
    
    Since subset sums depend on which elements are chosen, we use a
    different formulation: enumerate all possible subset-sum pairs and
    add constraints that prevent both from being achievable.
    
    Actually, a cleaner approach: for each element value v in pool,
    x_v is binary. The constraint is that the set of all subset sums
    of the chosen elements has no divisibility pairs.
    
    This is hard to express as linear constraints directly. Instead,
    we use a different ILP formulation:
    - For each possible subset sum value s (from 1 to max_sum),
      y_s = 1 if s is achievable as a subset sum
    - Constraint: for all s, t where s|t and s != t: y_s + y_t <= 1
    - Constraint: y_s >= sum_{i in S_s} x_i - |S_s| + 1 (if s is sum of some subset)
    
    This is too complex. Let's use a simpler approach:
    Directly encode the problem with the constraint that for any two
    distinct non-empty subsets A, B of the chosen elements, 
    sum(A) does not divide sum(B) and vice versa.
    
    Given the complexity, I'll use a hybrid: ILP for the core selection
    with a relaxed constraint set, then verify and repair.
    """
    n = len(pool)
    max_sum = sum(pool)
    
    # We'll use a simpler ILP: select elements to maximize count
    # with constraints that no element divides another's contribution
    # in a way that creates divisibility. This is an approximation.
    
    # Actually, let's use a different approach: 
    # The key insight is that if all elements are > max_sum/2, then
    # no two elements can sum to form a multiple. But that limits us.
    
    # Better: use the fact that if we pick elements such that all subset
    # sums are "spread out" and no one is a multiple of another.
    
    # For the ILP, let's encode:
    # Variables: x_1, ..., x_n (binary)
    # Objective: maximize sum(x_i)
    # Constraints: For each pair (a, b) where a|b in the pool,
    #   we need to ensure that if both a and b are selected,
    #   no subset sum equals a and no other equals b (or multiples).
    
    # This is still complex. Let's try a different ILP formulation:
    # Use the fact that the constraint is about subset sums.
    # For a fixed set of n elements, there are 2^n - 1 subset sums.
    # We can't enumerate these in the ILP for large n.
    
    # Practical approach: Use ILP with a limited constraint set
    # that captures the most common violations.
    
    # For now, let's just use the ILP to select a subset of elements
    # maximizing count, with constraints that:
    # 1. No two selected elements have one dividing the other
    # 2. The sum of any two selected elements doesn't divide the sum of any three
    # (This is a relaxation that's easy to encode)
    
    # c[i,j] = -1 if pool[i] divides pool[j] or pool[j] divides pool[i]
    # Constraint: x_i + x_j <= 1 for all such pairs
    
    # This is a maximum independent set in the "divisibility graph"
    
    # Build divisibility graph
    div_graph = {}
    for i in range(n):
        for j in range(i+1, n):
            a, b = pool[i], pool[j]
            if a % b == 0 or b % a == 0:
                if i not in div_graph:
                    div_graph[i] = []
                if j not in div_graph:
                    div_graph[j] = []
                div_graph[i].append(j)
                div_graph[j].append(i)
    
    # MILP: max sum(x_i) s.t. x_i + x_j <= 1 for all edges
    c = -np.ones(n)  # minimize -sum(x_i) = maximize sum(x_i)
    
    # Constraints: x_i + x_j <= 1
    A_rows = []
    b_ub = []
    for i in range(n):
        if i in div_graph:
            for j in div_graph[i]:
                if i < j:  # avoid duplicates
                    row = np.zeros(n)
                    row[i] = 1
                    row[j] = 1
                    A_rows.append(row)
                    b_ub.append(1)
    
    if not A_rows:
        return pool[:target_size]
    
    A = np.array(A_rows)
    b_ub_arr = np.array(b_ub)
    
    constraints = LinearConstraint(A, lb=None, ub=b_ub_arr)
    integrality = np.ones(n)
    bounds = Bounds(lb=0, ub=1)
    
    try:
        result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        if result.status == 0 and result.x is not None:
            selected = [pool[i] for i in range(n) if result.x[i] > 0.5]
            return selected
    except Exception:
        pass
    
    return []

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    pool = list(range(1, 257))
    best_set = []
    best_size = 0
    
    # Phase 1: Multiple greedy constructions
    orderings = [
        sorted(pool, reverse=True),
        sorted(pool),
        sorted(pool, key=lambda x: -x.bit_count()),  # More bits first
        [x for x in pool if x > 128] + [x for x in pool if 64 < x <= 128] + [x for x in pool if x <= 64],
        sorted(pool, key=lambda x: -(x & -x)),  # Sort by trailing zeros
    ]
    
    for order in orderings:
        if time.time() > deadline - 20:
            break
        result = _greedy_build(pool, order)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Phase 2: Local search from best
    if best_set and time.time() < deadline - 10:
        ls_result = _local_search(best_set, pool, deadline - 10)
        if len(ls_result) > best_size:
            best_size = len(ls_result)
            best_set = ls_result
    
    # Phase 3: ILP-based selection (divisibility graph independent set)
    if time.time() < deadline - 5:
        ilp_result = _solve_ilp(pool, deadline - 5, best_size + 5)
        if ilp_result:
            # Verify and try to extend
            ilp_set = ilp_result
            if _is_valid(ilp_set):
                if len(ilp_set) > best_size:
                    best_size = len(ilp_set)
                    best_set = ilp_set[:]
                # Try to extend the ILP solution
                if time.time() < deadline - 3:
                    ext_result = _local_search(ilp_set, pool, deadline - 3)
                    if len(ext_result) > best_size:
                        best_size = len(ext_result)
                        best_set = ext_result
    
    # Phase 4: Random restarts with local search
    if time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            random.shuffle(pool)
            seed_set = _greedy_build(pool, pool[:50])
            if len(seed_set) >= best_size - 1:
                ls_result = _local_search(seed_set, pool, deadline - 5)
                if len(ls_result) > best_size:
                    best_size = len(ls_result)
                    best_set = ls_result[:]
    
    # Final validation
    if best_set:
        final_sums = _subset_sums(best_set)
        if not _check_valid(final_sums):
            # Fallback to simple greedy
            best_set = _greedy_build(pool, sorted(pool, reverse=True))
    
    # Ensure we have at least a minimal valid set
    if not best_set:
        for x in sorted(pool, reverse=True):
            if _is_valid([x]):
                best_set = [x]
                break
    
    return [int(x) for x in best_set]
