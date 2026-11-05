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

    def greedy_fallback(elements):
        """Fallback greedy if ILP fails or is too slow."""
        S = set()
        # Sort by absolute value descending as a heuristic
        for x in sorted(elements, key=abs, reverse=True):
            ok = True
            if 2 * x in S:
                ok = False
            else:
                for a in S:
                    if x + a in S or a - x in S:
                        ok = False
                        break
            if ok:
                S.add(x)
        return sorted(S)

    def solve_sum_free_ilp(elements):
        n = len(elements)
        if n == 0:
            return []
        
        # Map value to index
        val_to_idx = {v: i for i, v in enumerate(elements)}
        
        # Find all bad triples
        constraints_rows = []
        
        # Iterate over all pairs (i, j) with i <= j
        for i in range(n):
            vi = elements[i]
            for j in range(i, n):
                vj = elements[j]
                s = vi + vj
                if s in val_to_idx:
                    k = val_to_idx[s]
                    # Constraint: xi + xj + xk <= 2
                    # If i == j, 2xi + xk <= 2
                    row = [0.0] * n
                    if i == j:
                        row[i] = 2.0
                        row[k] = 1.0
                    else:
                        row[i] = 1.0
                        row[j] = 1.0
                        row[k] = 1.0
                    constraints_rows.append(row)
        
        if not constraints_rows:
            # No constraints, take all
            return elements[:]
        
        A = np.array(constraints_rows)
        
        # Objective: maximize sum(xi) -> minimize -sum(xi)
        c = -np.ones(n)
        
        # Bounds: 0 <= xi <= 1
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: 1 (integer)
        integrality = np.ones(n)
        
        # Constraint: 0 <= A @ x <= 2
        constraint = LinearConstraint(A, lb=0, ub=2)
        
        try:
            res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
            
            if not res.success:
                return greedy_fallback(elements)
                
            # res.x contains the solution
            # Due to numerical precision, threshold at 0.5
            solution_mask = res.x > 0.5
            return [elements[i] for i in range(n) if solution_mask[i]]
        except Exception:
            return greedy_fallback(elements)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elems = instance(size, seed)
        res = solve_sum_free_ilp(elems)
        results.append(sorted(res))
        
    return results
# EVOLVE-BLOCK-END
