import time
import random
import math
import numpy as np
from random import Random
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    deadline = time.time() + 145.0  # Safety margin

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if S is sum-free."""
        s = set(S)
        for a in s:
            for b in s:
                if a + b in s:
                    return False
        return True

    def solve_ilp(elements):
        """Solve the sum-free subset problem using MILP."""
        n = len(elements)
        if n == 0:
            return []
        
        val_to_idx = {v: i for i, v in enumerate(elements)}
        
        # Collect constraints: for each pair (i, j), if elements[i] + elements[j] is in elements,
        # add constraint x_i + x_j + x_k <= 2
        # We need to handle i == j case correctly.
        # If i == j, constraint is 2*x_i + x_k <= 2.
        # If i != j and k == i or k == j, constraint is x_i + x_j + x_i <= 2 => 2*x_i + x_j <= 2.
        # Generally, we just sum the coefficients.
        
        constraints = []
        # To avoid duplicate constraints, we can iterate i <= j
        # But the constraint structure depends on the specific values.
        # Let's just generate all unique triples of indices (i, j, k) such that v_i + v_j = v_k.
        # Note: (i, j, k) and (j, i, k) represent the same constraint.
        # We should canonicalize or just let the solver handle it, but unique is better for speed.
        
        seen = set()
        A_rows = []
        
        for i in range(n):
            vi = elements[i]
            for j in range(i, n): # i <= j to avoid duplicates for (i,j) and (j,i)
                vj = elements[j]
                s = vi + vj
                if s in val_to_idx:
                    k = val_to_idx[s]
                    # The constraint involves indices i, j, k.
                    # We need to build a row for A.
                    # The row is a vector of length n, mostly zeros.
                    # Coefficients:
                    # count of i in {i, j, k}
                    # count of j in {i, j, k}
                    # count of k in {i, j, k}
                    
                    # To avoid storing huge sparse matrices in dense form, we might use sparse, 
                    # but scipy milp accepts sparse matrices.
                    # However, for n=240, dense is 240*240 floats = 46k floats, which is fine.
                    # But M could be up to n^2/2 ~ 30k. 30k * 240 = 7.2M floats. ~50MB. Acceptable.
                    
                    # Let's collect indices and counts
                    counts = {}
                    counts[i] = counts.get(i, 0) + 1
                    counts[j] = counts.get(j, 0) + 1
                    counts[k] = counts.get(k, 0) + 1
                    
                    # Create a key to ensure we don't add the same constraint twice 
                    # (though i<=j and k is determined, could different (i,j) lead to same multiset of indices?)
                    # Example: 1+2=3. i=1, j=2, k=3. Indices {1,2,3}.
                    # Is there another pair? No, sum is unique.
                    # What about 1+1=2? i=1, j=1, k=2. Indices {1,1,2}.
                    # So the multiset of indices uniquely identifies the constraint.
                    
                    key = tuple(sorted(counts.keys())) # Not quite, need counts.
                    # Better key: sorted tuple of (index, count)
                    key = tuple(sorted((idx, cnt) for idx, cnt in counts.items()))
                    
                    if key in seen:
                        continue
                    seen.add(key)
                    
                    row = np.zeros(n)
                    for idx, cnt in counts.items():
                        row[idx] = cnt
                    A_rows.append(row)
        
        if not A_rows:
            return elements[:]
            
        A = np.array(A_rows)
        
        # Objective: maximize sum(x) -> minimize -sum(x)
        c = -np.ones(n)
        
        # Constraints: A @ x <= 2
        # LinearConstraint(A, lb=-np.inf, ub=2)
        cons = LinearConstraint(A, lb=-np.inf, ub=2)
        
        # Bounds: 0 <= x <= 1
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: all integer
        integrality = np.ones(n)
        
        try:
            res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
            if res.success:
                selected_indices = [i for i in range(n) if res.x[i] > 0.5]
                return [elements[i] for i in selected_indices]
            else:
                # Fallback if ILP fails
                return None
        except Exception:
            return None

    def heuristic_fallback(elements):
        """Simple greedy fallback if ILP fails."""
        included_vals = set()
        # Sort by absolute value descending? Or just iterate.
        # A common heuristic: elements with largest absolute value are less likely to form sums.
        for v in sorted(elements, key=abs, reverse=True):
            ok = True
            for a in included_vals:
                if (v - a) in included_vals or (v + a) in included_vals:
                    ok = False
                    break
                if 2 * v in included_vals:
                    ok = False
                    break
            if ok:
                included_vals.add(v)
        return list(included_vals)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elems = instance(size, seed)
        
        # Try ILP first
        ilp_result = solve_ilp(elems)
        
        if ilp_result is not None:
            # Verify
            if is_sum_free(ilp_result):
                results.append(ilp_result)
            else:
                # Should not happen, but fallback just in case
                results.append(heuristic_fallback(elems))
        else:
            # Fallback
            results.append(heuristic_fallback(elems))
            
        # Check time
        if time.time() > deadline - 5:
             # If we are running out of time, ensure we have 3 results
             break

    # Ensure we have 3 results
    while len(results) < 3:
        size, seed = ((60, 801), (120, 802), (240, 803))[len(results)]
        elems = instance(size, seed)
        results.append(heuristic_fallback(elems))
        
    return results
