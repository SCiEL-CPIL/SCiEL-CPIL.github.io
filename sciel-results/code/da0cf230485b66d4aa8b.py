"""Rewritten program: exact reciprocal-subsum-free maximization via
Integer Linear Programming (MILP) using SciPy."""

def construct():
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    from itertools import combinations

    N = 24
    # Scale factor to convert reciprocals to integers
    # lcm(1..24)
    import math
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    scale = 1
    for i in range(1, N + 1):
        scale = lcm(scale, i)
        
    # Weights W_i = scale // i
    W = [0] * (N + 1) # 1-indexed
    for i in range(1, N + 1):
        W[i] = scale // i

    # Identify all "bad" configurations (x, S) where sum_{y in S} 1/y == 1/x
    # S must be a subset of {x+1, ..., N} because any y < x has 1/y > 1/x.
    bad_constraints = []
    
    for x in range(1, N + 1):
        target_w = W[x]
        # Candidates for S are elements y > x
        candidates = list(range(x + 1, N + 1))
        n_cand = len(candidates)
        
        # We need to find subsets S of candidates such that sum(W[y] for y in S) == target_w
        # Since N=24, max candidates for x=1 is 23. 2^23 is too large for brute force.
        # However, we can use meet-in-the-middle or just iterate subsets if small.
        # Actually, for x=1, target=scale. Sum of all others?
        # Let's use a dictionary of sums for the second half.
        
        # Optimization: Only consider subsets S.
        # If n_cand is large, MITM is better.
        # But note: W[y] are large integers.
        
        # Simple MITM for subset sum
        if n_cand == 0:
            continue
            
        mid = n_cand // 2
        left = candidates[:mid]
        right = candidates[mid:]
        
        left_sums = {0: []}
        for i, y in enumerate(left):
            new_sums = {}
            for s, idxs in left_sums.items():
                # existing sum
                if s not in new_sums:
                    new_sums[s] = []
                new_sums[s].append(idxs)
                # sum + W[y]
                ns = s + W[y]
                if ns not in new_sums:
                    new_sums[ns] = []
                new_sums[ns].append(idxs + [i])
            left_sums = new_sums
            
        right_sums = {0: []}
        for i, y in enumerate(right):
            new_sums = {}
            for s, idxs in right_sums.items():
                if s not in new_sums:
                    new_sums[s] = []
                new_sums[s].append(idxs)
                ns = s + W[y]
                if ns not in new_sums:
                    new_sums[ns] = []
                new_sums[ns].append(idxs + [i])
            right_sums = new_sums

        # Find pairs (sl, sr) such that sl + sr == target_w
        found_subsets = set()
        for sl, idxs_l in left_sums.items():
            needed = target_w - sl
            if needed in right_sums:
                for idxs_r in right_sums[needed]:
                    # Combine indices
                    # idxs_l are indices in 'left', idxs_r in 'right'
                    # Map back to original candidate indices
                    orig_indices = []
                    for idx in idxs_l:
                        orig_indices.append(left[idx])
                    for idx in idxs_r:
                        orig_indices.append(right[idx])
                    # The subset S consists of these candidate values
                    S_tuple = tuple(sorted(orig_indices))
                    found_subsets.add(S_tuple)
                    
        # Add constraints for each unique bad subset S
        for S in found_subsets:
            # Constraint: x_x + sum_{y in S} x_y <= |S|
            # Variables are x_1 ... x_24 (0-indexed in scipy: x[0] is x_1)
            # x_x corresponds to index x-1
            # y in S corresponds to index y-1
            bad_constraints.append((x - 1, S))

    # Build MILP
    # Variables: x[0] ... x[23]
    c = np.ones(N) # Maximize sum(x) -> Minimize -sum(x)
    
    # Constraints:
    # 1. x_i >= 0 (handled by bounds)
    # 2. x_i <= 1 (handled by bounds)
    # 3. Bad constraints: x_x + sum_{y in S} x_y <= |S|
    
    A_rows = []
    b_rows = []
    
    for x_idx, S in bad_constraints:
        row = np.zeros(N)
        row[x_idx] = 1.0
        for y in S:
            row[y - 1] = 1.0
        A_rows.append(row)
        b_rows.append(len(S))
        
    if not A_rows:
        # No bad constraints? Then we can pick all.
        # But 1/1 = 1. Is there a subset of others summing to 1?
        # 1/2 + 1/3 + 1/6 = 1. So {2,3,6} is a bad subset for 1.
        # So A_rows should not be empty.
        pass

    A = np.array(A_rows)
    b = np.array(b_rows)
    
    # LinearConstraint: A @ x <= b
    constraints = LinearConstraint(A, None, b)
    integrality = np.ones(N)
    bounds = Bounds(0, 1)
    
    res = milp(c=-c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.success:
        selected = [i + 1 for i in range(N) if res.x[i] > 0.5]
        if selected:
            return sorted(selected)
            
    # Fallback: Greedy (from previous logic, simplified)
    # Just return a known valid small set or run a quick greedy
    def greedy():
        sums = {0}
        chosen = []
        for i in range(N, 0, -1):
            w = scale // i
            if w not in sums:
                chosen.append(i)
                sums = sums | {s + w for s in sums}
        return sorted(chosen)
        
    return greedy()

# EVOLVE-BLOCK-END
