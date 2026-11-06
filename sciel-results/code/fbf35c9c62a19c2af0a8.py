"""
Rewrite using SciPy's MILP solver (HiGHS) to find the maximum cardinality subset
avoiding reciprocal Schur triples. This models the problem as a 0-1 Integer
Linear Program, which is solved via branch-and-bound/cutting planes, providing
a global or near-global optimum significantly better than greedy heuristics.
"""
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    N = 180
    
    # Precompute all forbidden triples (a, b, c) with a < b < c
    # such that 1/a = 1/b + 1/c <=> b*c = a*(b+c)
    forbidden = []
    for b in range(2, N + 1):
        for c in range(b + 1, N + 1):
            numerator = b * c
            denominator = b + c
            if numerator % denominator == 0:
                a = numerator // denominator
                if 1 <= a < b:
                    forbidden.append((a, b, c))
                    
    num_constraints = len(forbidden)
    
    # Objective: Maximize sum(x_i)
    # min -sum(x_i)
    c_obj = -np.ones(N)
    
    # Constraints:
    # 1. x_i >= 0 (handled by bounds)
    # 2. x_a + x_b + x_c <= 2 for each forbidden triple
    
    if num_constraints == 0:
        return list(range(1, N + 1))
        
    # A matrix for constraints: num_constraints x N
    A = np.zeros((num_constraints, N), dtype=float)
    for i, (a, b, c) in enumerate(forbidden):
        A[i, a-1] = 1
        A[i, b-1] = 1
        A[i, c-1] = 1
        
    # Bounds for variables: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Linear Constraint: A @ x <= 2
    # A @ x >= -inf
    lin_cons = LinearConstraint(A, ub=np.full(num_constraints, 2))
    
    # Integrality: all variables are integers
    integrality = np.ones(N)
    
    # Solve
    res = milp(c=c_obj, constraints=lin_cons, integrality=integrality, bounds=bounds)
    
    if res.success:
        # res.x is the optimal solution
        # Convert to set of indices where x_i > 0.5
        solution_mask = res.x > 0.5
        result = [i+1 for i in range(N) if solution_mask[i]]
        if len(result) > 0:
            return result
    else:
        # Fallback to greedy if solver fails
        # (Unlikely for this size, but safe)
        pass
        
    # Fallback: Simple Greedy
    # Re-use the logic from the previous version if needed, 
    # but for this specific task, the solver should work.
    # Let's implement a quick greedy fallback just in case.
    S = set(range(1, N + 1))
    # ... (omitted for brevity in thought, will include in code) ...
    
    # Since we must return a valid set, and the solver is robust, 
    # we return the solver result. If it fails, we return a valid subset 
    # found by a quick greedy pass.
    
    # Quick Greedy Fallback
    S = set(range(1, N + 1))
    triples = [[] for _ in range(N + 1)]
    for i, (a, b, c) in enumerate(forbidden):
        triples[a].append(i)
        triples[b].append(i)
        triples[c].append(i)
        
    triple_count = [3] * len(forbidden)
    weights = [len(triples[x]) for x in range(N + 1)]
    
    while True:
        max_w = -1
        best_x = -1
        for x in range(1, N + 1):
            if x in S and weights[x] > max_w:
                max_w = weights[x]
                best_x = x
        if max_w <= 0:
            break
        S.remove(best_x)
        for t_idx in triples[best_x]:
            triple_count[t_idx] -= 1
            if triple_count[t_idx] == 2:
                a, b, c = forbidden[t_idx]
                for y in (a, b, c):
                    if y != best_x and y in S:
                        weights[y] -= 1
    return list(S)
