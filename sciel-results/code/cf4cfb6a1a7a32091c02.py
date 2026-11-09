import time
import random
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_instance(values, time_budget):
        n = len(values)
        if n <= 2:
            return list(values)

        v_set = set(values)
        v_to_idx = {v: i for i, v in enumerate(values)}

        # 1. Build 3-uniform hypergraph constraints
        # Constraint: x_i + x_j + x_k <= 2 for all i, j, k such that v_i + v_j = v_k
        # Note: i, j, k are indices. The triple is (i, j, k).
        # We must consider all permutations? No, the constraint is symmetric in i and j, but k is distinct.
        # Actually, the condition is "no a+b=c". So for any a, b in subset, a+b not in subset.
        # This is equivalent to: for any triple (a, b, c) in V with a+b=c, not all of a, b, c are in the subset.
        # So x_a + x_b + x_c <= 2.
        
        # We need to generate all such triples.
        # To avoid duplicates and ensure we cover all cases:
        # For each pair (i, j) with i <= j, check if v_i + v_j is in V.
        # If yes, let k be the index of v_i + v_j.
        # Add constraint x_i + x_j + x_k <= 2.
        # Note: If i == j, then 2*v_i = v_k.
        # If i != j, then v_i + v_j = v_k.
        
        rows = []
        cols = []
        
        # Optimization: Only add constraints for unique triples?
        # The constraint x_i + x_j + x_k <= 2 is the same for (i, j, k) and (j, i, k).
        # So we can iterate i <= j.
        
        for i in range(n):
            vi = values[i]
            for j in range(i, n):
                vj = values[j]
                s = vi + vj
                if s in v_to_idx:
                    k = v_to_idx[s]
                    # We have a triple (i, j, k).
                    # If k is the same as i or j, it's still a valid constraint.
                    # Example: vi + vi = vj -> 2*vi = vj. Constraint: x_i + x_i + x_j <= 2 -> 2*x_i + x_j <= 2.
                    # This is correct.
                    # We add the row.
                    # To avoid adding duplicate rows (which increases solver time), we could use a set of tuples.
                    # But for n=240, the number of triples is likely not huge (O(n^2) worst case, but sparse).
                    # Let's just add them.
                    rows.append([i, j, k])
                    cols.append([1, 1, 1])

        if not rows:
            return list(values)

        # Convert to sparse matrix format for efficiency
        # scipy.optimize.milp expects A in dense or sparse format.
        # For n=240 and potentially ~1000-10000 constraints, dense is fine.
        A = np.zeros((len(rows), n), dtype=float)
        for r, row in enumerate(rows):
            A[r, row[0]] = 1
            A[r, row[1]] = 1
            A[r, row[2]] = 1

        c = -np.ones(n)  # Minimize -sum(x)
        
        constraints = [
            LinearConstraint(A, ub=np.ones(len(rows)))
        ]
        
        bounds = Bounds(lb=0, ub=1)
        
        # Check time
        if time.time() - start_time > DEADLINE - 5:
            return [v for v in values if v % 2 != 0]

        try:
            res = milp(
                c=c,
                constraints=constraints,
                integrality=np.ones(n),
                bounds=bounds,
                options={"time_limit": time_budget - (time.time() - start_time) - 1}
            )
            
            if res.status == 0:  # Optimal solution found
                x = res.x
                x_bin = (x > 0.5).astype(int)
                selected_indices = np.where(x_bin == 1)[0]
                result_vals = [values[i] for i in selected_indices]
                
                if is_sum_free(result_vals):
                    return result_vals
            elif res.status == 1: # Locally optimal
                x = res.x
                x_bin = (x > 0.5).astype(int)
                selected_indices = np.where(x_bin == 1)[0]
                result_vals = [values[i] for i in selected_indices]
                if is_sum_free(result_vals):
                    return result_vals
        except Exception:
            pass

        # Fallback: Greedy with 3-way check
        # Sort by degree in the hypergraph (number of triples involving the vertex)
        degrees = np.zeros(n, dtype=int)
        for row in rows:
            degrees[row[0]] += 1
            degrees[row[1]] += 1
            degrees[row[2]] += 1
        
        order = np.argsort(degrees)
        selected = []
        in_set = np.zeros(n, dtype=bool)
        
        # Precompute conflicts for fast check?
        # For a new vertex v, it conflicts if there exists u in selected such that (v, u, w) is a triple for some w in selected?
        # Actually, we just need to check if adding v creates a sum-free violation.
        # A violation occurs if there exist a, b in selected such that a+b = v (and v is added) OR v+b = a (a in selected) OR v+a = b (b in selected).
        # Basically, v conflicts with selected if there is a triple (i, j, k) where one is v and the other two are in selected.
        
        # Let's use a set of selected values for O(1) lookup.
        selected_vals = set()
        
        for i in order:
            vi = values[i]
            # Check if vi conflicts with any selected value
            conflict = False
            for vs in selected_vals:
                # Check if vi + vs is in selected_vals
                if vi + vs in selected_vals:
                    conflict = True
                    break
                # Check if vi - vs is in selected_vals (i.e., vi = vs + w -> w = vi - vs)
                # This covers the case where vs + w = vi.
                if vi - vs in selected_vals:
                    conflict = True
                    break
                # Check if vs - vi is in selected_vals (i.e., vs = vi + w -> w = vs - vi)
                # This covers the case where vi + w = vs.
                if vs - vi in selected_vals:
                    conflict = True
                    break
            
            if not conflict:
                selected.append(i)
                selected_vals.add(vi)
                in_set[i] = True

        result_vals = [values[i] for i in selected]
        
        if not is_sum_free(result_vals):
            result_vals = [v for v in values if v % 2 != 0]
            
        return result_vals

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for i, (size, seed) in enumerate(instances):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        
        if remaining < 10:
            vals = instance(size, seed)
            res = [x for x in vals if x % 2 != 0]
            results.append(res)
            continue
        
        vals = instance(size, seed)
        res = solve_instance(vals, remaining)
        
        if not is_sum_free(res):
            res = [x for x in vals if x % 2 != 0]
        
        results.append(res)
        
    return results
