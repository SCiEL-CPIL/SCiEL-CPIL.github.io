# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    n = 128
    numbers = list(range(1, n + 1))

    # 1. Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    # 2. Identify all bad triples: (a,b,c) with a<b<c where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))

    # 3. Build conflict structure for fast validity checks
    # For each element x, store the set of pairs (u,v) such that {x,u,v} is a bad triple
    conflict = {i: set() for i in range(1, n + 1)}
    for a, b, c in bad_triples:
        conflict[a].add((b, c))
        conflict[b].add((a, c))
        conflict[c].add((a, b))

    def is_valid(S, x):
        """Check if x can be added to set S without forming a bad triple."""
        for u, v in conflict[x]:
            if u in S and v in S:
                return False
        return True

    # 4. QUBO / Quadratic Programming formulation:
    # Maximize sum(x_i) subject to: for each bad triple (a,b,c), x_a*x_b + x_a*x_c + x_b*x_c <= 2
    # (i.e., at most 2 of the 3 can be selected)
    # Equivalently, minimize: -sum(x_i) + M * sum_over_triples [max(0, x_a+x_b+x_c - 2)^2]
    # For smooth optimization, use: minimize -sum(x_i) + M * sum_over_triples [(x_a+x_b+x_c-2)^2 * indicator]
    # Better: minimize -sum(x_i) + M * sum_over_triples [x_a*x_b + x_a*x_c + x_b*x_c - 2*x_a*x_b*x_c + ...]
    # Simplest smooth surrogate: minimize -sum(x_i) + M * sum_over_triples [(x_a+x_b+x_c)^2 - 4*(x_a+x_b+x_c) + 4]_+^2
    # Even simpler: minimize -sum(x_i) + M * sum_over_triples [(x_a+x_b+x_c - 2)^2]
    # This penalizes when sum > 2, but also when sum < 2. We want to penalize sum > 2 only.
    # Use: minimize -sum(x_i) + M * sum_over_triples [max(0, x_a+x_b+x_c - 2)^2]
    # For differentiability, approximate max(0,t)^2 with t^2 * sigmoid(k*t) or just t^2 with softplus.
    # Actually, let's use a simpler approach: the constraint is that NOT all three are 1.
    # The quadratic form x_a*x_b + x_a*x_c + x_b*x_c >= 1 means at least two are selected.
    # For a bad triple, we need: NOT (x_a=1 AND x_b=1 AND x_c=1)
    # A smooth penalty: M * (x_a * x_b * x_c)^2 penalizes all-three-selected.
    # But this is cubic. Let's use: M * (x_a*x_b + x_a*x_c + x_b*x_c - 2*x_a*x_b*x_c)
    # Hmm, let me think differently.
    
    # The cleanest QUBO: for each bad triple (a,b,c), add penalty M*(x_a*x_b + x_a*x_c + x_b*x_c - 2*x_a*x_b*x_c)
    # When all three are 1: penalty = M*(1+1+1-2) = M. Good.
    # When two are 1: penalty = M*(1+0+0-0) = M. Bad! We don't want to penalize two.
    # 
    # Better: the constraint is x_a + x_b + x_c <= 2.
    # Penalty for violation: M * max(0, x_a + x_b + x_c - 2)^2
    # For differentiability, use: M * softplus(x_a + x_b + x_c - 2)^2
    # Or simply: M * (x_a + x_b + x_c - 2)^2 and accept the symmetric penalty.
    # Since we're maximizing sum(x_i), the symmetric penalty will still push toward valid solutions
    # because the objective wants high sum, and the penalty only matters when sum > 2.
    # When sum < 2, the penalty (x_a+x_b+x_c-2)^2 is positive but the main objective -sum(x_i) 
    # is also negative (we want to maximize), so there's tension. Let's use one-sided penalty.
    
    # I'll use: minimize f(x) = -sum(x_i) + M * sum_triples [relu(x_a+x_b+x_c-2)]^2
    # Approximate relu(t)^2 with t^2 * sigmoid(20*t) for smoothness, or just t^2 for simplicity
    # since L-BFGS-B will handle the boundary.
    
    # Actually, the simplest effective approach: just use (x_a+x_b+x_c-2)^2 as penalty.
    # The optimizer will find that setting x_i=1 for valid elements and 0 for others minimizes this.
    
    M = 50.0  # Penalty weight
    
    # Build the quadratic form for scipy
    # f(x) = -sum(x_i) + M * sum_{(a,b,c) in bad_triples} (x_a + x_b + x_c - 2)^2
    # = -sum(x_i) + M * sum_{(a,b,c)} (x_a^2 + x_b^2 + x_c^2 + 2*x_a*x_b + 2*x_a*x_c + 2*x_b*x_c - 4*x_a - 4*x_b - 4*x_c + 4)
    
    # Collect quadratic coefficients: Q[i][j] for i,j in 1..128
    # f(x) = x^T Q x + c^T x + const
    # Linear part: c[i] = -1 + M * sum_{triples containing i} (-4 + 2*count_in_other_positions)
    # This is getting complex. Let me just use scipy.optimize.minimize with explicit gradient.
    
    import numpy as np
    
    # Map element i (1-indexed) to index i-1 (0-indexed)
    idx = lambda i: i - 1
    
    # Precompute for each element, which triples it's in
    triple_membership = {i: [] for i in range(1, n + 1)}
    for a, b, c in bad_triples:
        triple_membership[a].append((a, b, c))
        triple_membership[b].append((a, b, c))
        triple_membership[c].append((a, b, c))
    
    def objective(x):
        """x is a 128-dim vector in [0,1]. Returns scalar."""
        # x[i-1] corresponds to element i
        s = np.sum(x)
        penalty = 0.0
        for a, b, c in bad_triples:
            t = x[idx(a)] + x[idx(b)] + x[idx(c)] - 2.0
            penalty += t * t
        return -s + M * penalty
    
    def gradient(x):
        """Gradient of objective."""
        g = np.ones(n)  # d(-sum(x))/dx_i = -1, but we return negative for minimize
        # Actually: d(-sum(x))/dx_i = -1
        # d(M * sum (t)^2)/dx_i where t = x_a+x_b+x_c-2
        # = M * 2 * t * 1 for each triple containing i
        g = -np.ones(n)
        for a, b, c in bad_triples:
            t = x[idx(a)] + x[idx(b)] + x[idx(c)] - 2.0
            g[idx(a)] += M * 2.0 * t
            g[idx(b)] += M * 2.0 * t
            g[idx(c)] += M * 2.0 * t
        return g
    
    # 5. Solve the continuous relaxation using L-BFGS-B
    # Multiple restarts with different initializations
    best_set = set()
    best_size = 0
    
    def refine_from_seed(seed_set):
        """Take a seed set and try to improve via local search."""
        S = set(seed_set)
        # Phase 1: Add any valid elements
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for x in numbers:
                if x not in S and is_valid(S, x):
                    S.add(x)
                    changed = True
        # Phase 2: Randomized 1-swap local search
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            trial = set(S)
            elems = list(trial)
            if not elems:
                break
            out_elem = random.choice(elems)
            trial.remove(out_elem)
            # Try to add best replacement
            best_in = None
            for in_elem in numbers:
                if in_elem in trial:
                    continue
                if is_valid(trial, in_elem):
                    # Check if this improves (it always does: remove 1, add 1, size same)
                    # But after adding, we might be able to add more
                    trial2 = set(trial)
                    trial2.add(in_elem)
                    # Try to add more
                    local_changed = True
                    while local_changed and time.time() < deadline - 5:
                        local_changed = False
                        for y in numbers:
                            if y not in trial2 and is_valid(trial2, y):
                                trial2.add(y)
                                local_changed = True
                    if len(trial2) > len(S):
                        best_in = in_elem
                        S = trial2
                        break
        return S
    
    # Run QP relaxation with multiple restarts
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        if restart == 0:
            # Start from all-ones (maximally inclusive)
            x0 = np.ones(n)
        elif restart == 1:
            # Start from random
            x0 = np.random.uniform(0.3, 0.7, n)
        elif restart == 2:
            # Start from zeros
            x0 = np.zeros(n)
        else:
            # Random
            x0 = np.random.uniform(0.0, 1.0, n)
        
        try:
            result = scipy_optimize.minimize(
                objective, x0, jac=gradient, method='L-BFGS-B',
                bounds=[(0, 1)] * n,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
            )
        except Exception:
            continue
        
        # Round the solution: threshold at 0.5
        x_sol = result.x
        seed_set = set()
        for i in range(1, n + 1):
            if x_sol[idx(i)] > 0.5:
                seed_set.add(i)
        
        # Refine the seed
        refined = refine_from_seed(seed_set)
        if len(refined) > best_size:
            best_size = len(refined)
            best_set = refined
    
    # 6. Also try the greedy-remove-then-add-back strategy (proven to help)
    if time.time() < deadline - 10:
        # Start with all elements, remove those that cause conflicts
        S_all = set(numbers)
        # Remove elements in order of highest conflict count
        conflict_count = {i: len(conflict[i]) for i in range(1, n + 1)}
        order = sorted(numbers, key=lambda x: -conflict_count[x])
        
        S_remove = set(S_all)
        for x in order:
            if time.time() > deadline - 5:
                break
            # Check if x is part of any bad triple in current set
            needs_removal = False
            for u, v in conflict[x]:
                if u in S_remove and v in S_remove:
                    needs_removal = True
                    break
            if needs_removal:
                S_remove.discard(x)
        
        # Now try to add back
        S_back = set(S_remove)
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for x in numbers:
                if x not in S_back and is_valid(S_back, x):
                    S_back.add(x)
                    changed = True
        
        if len(S_back) > best_size:
            best_size = len(S_back)
            best_set = S_back
    
    # 7. Final local search polish on best set
    if time.time() < deadline - 5 and best_set:
        current = set(best_set)
        # Add phase
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            for x in numbers:
                if x not in current and is_valid(current, x):
                    current.add(x)
                    changed = True
        # Swap phase
        for _ in range(300):
            if time.time() > deadline - 3:
                break
            elems = list(current)
            if not elems:
                break
            out_elem = random.choice(elems)
            trial = set(current)
            trial.discard(out_elem)
            for in_elem in numbers:
                if in_elem in trial:
                    continue
                if is_valid(trial, in_elem):
                    trial.add(in_elem)
                    # Add more
                    local_changed = True
                    while local_changed and time.time() < deadline - 3:
                        local_changed = False
                        for y in numbers:
                            if y not in trial and is_valid(trial, y):
                                trial.add(y)
                                local_changed = True
                    if len(trial) > len(current):
                        current = trial
                    break
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current
    
    # 8. Fallback: ensure non-empty
    if not best_set:
        best_set = {1}
    
    return sorted(list(best_set))

# EVOLVE-BLOCK-END
