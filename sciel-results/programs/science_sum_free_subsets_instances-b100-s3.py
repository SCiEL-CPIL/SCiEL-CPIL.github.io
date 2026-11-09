"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    from random import Random

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def greedy_sum_free(elements):
        """Fallback greedy method if ILP fails or for small sets."""
        if not elements:
            return []
        sorted_elems = sorted(elements, key=lambda x: -abs(x))
        selected = []
        selected_set = set()
        for elem in sorted_elems:
            can_add = True
            for a in selected:
                if (elem - a) in selected_set:
                    can_add = False
                    break
            if can_add:
                for b in selected:
                    if (b - elem) in selected_set:
                        can_add = False
                        break
            if can_add:
                if (2 * elem) in selected_set:
                    can_add = False
            if can_add:
                selected.append(elem)
                selected_set.add(elem)
        return selected

    def solve_sum_free_ilp(elements):
        """
        Solve for the maximum sum-free subset using Integer Linear Programming.
        Variables: x_i in {0,1} for each element i.
        Objective: Maximize sum(x_i).
        Constraints: For all i,j,k such that e_i + e_j == e_k, x_i + x_j + x_k <= 2.
        """
        n = len(elements)
        if n == 0:
            return []
        
        # For very small n, greedy is fast enough and avoids solver overhead
        if n < 10:
            return greedy_sum_free(elements)

        val_to_idx = {v: i for i, v in enumerate(elements)}
        
        # Collect constraints
        # We iterate over unique pairs (i, j) with i <= j to cover a=b cases and avoid duplicates
        # If e_i + e_j == e_k, we add constraint x_i + x_j + x_k <= 2
        constraint_rows = []
        for i in range(n):
            for j in range(i, n):
                s = elements[i] + elements[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    # Store the indices involved in this constraint
                    constraint_rows.append((i, j, k))
        
        M = len(constraint_rows)
        
        if M == 0:
            # No additive triples, any subset is sum-free
            return elements[:]

        # Build constraint matrix A
        A = np.zeros((M, n), dtype=np.float64)
        for idx, (i, j, k) in enumerate(constraint_rows):
            A[idx, i] += 1.0
            A[idx, j] += 1.0
            A[idx, k] += 1.0

        # Objective: Maximize sum(x) -> Minimize -sum(x)
        c = -np.ones(n)
        
        # Bounds: 0 <= x_i <= 1
        bounds = Bounds(lb=0, ub=1)
        
        # Constraints: 0 <= A x <= 2
        # Actually we just need A x <= 2. The lower bound 0 is implied by x>=0 and A>=0.
        # But LinearConstraint requires lb and ub.
        cons = LinearConstraint(A, lb=0, ub=2)
        
        # Integrality: All variables are integer (binary)
        integrality = np.ones(n)
        
        try:
            res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
            
            if res.success:
                # Extract solution
                # res.x might have small floating point errors
                selected_indices = [i for i in range(n) if res.x[i] > 0.5]
                return [elements[i] for i in selected_indices]
            else:
                # If solver fails, fallback to greedy
                return greedy_sum_free(elements)
                
        except Exception:
            # Fallback on any exception
            return greedy_sum_free(elements)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elems = instance(size, seed)
        subset = solve_sum_free_ilp(elems)
        results.append(subset)

    return results
# EVOLVE-BLOCK-END
