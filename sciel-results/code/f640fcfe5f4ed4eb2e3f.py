"""Combined program: exact MILP with heuristic fallback and local-search refinement."""
import time
import math
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill

    N = 256
    max_sum = 2 * N

    # 1. Precompute squares up to 2*N
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(i * i)

    # 2. Filter vertices: 2*v must not be a square (diagonal constraint a=a => 2a not square)
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)

    n_vars = len(valid_vertices)
    if n_vars == 0:
        return [1]  # Fallback, though 1 is valid (2 != square)

    # Map vertex to index
    v_to_idx = {v: i for i, v in enumerate(valid_vertices)}

    # 3. Build constraint matrix for MILP
    # Constraints: x_i + x_j <= 1 for all pairs (i, j) such that valid_vertices[i] + valid_vertices[j] is a square.
    constraint_pairs = []
    for i in range(n_vars):
        vi = valid_vertices[i]
        for j in range(i + 1, n_vars):
            vj = valid_vertices[j]
            if vi + vj in squares:
                constraint_pairs.append((i, j))

    m_constraints = len(constraint_pairs)

    if m_constraints == 0:
        # No conflicts between distinct vertices. All valid vertices can be selected.
        return valid_vertices

    # Initialize A_ub
    A_ub = np.zeros((m_constraints, n_vars), dtype=np.float64)
    b_ub = np.ones(m_constraints, dtype=np.float64)

    for k, (i, j) in enumerate(constraint_pairs):
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0

    # Objective: Maximize sum(x) => Minimize -sum(x)
    c = -np.ones(n_vars, dtype=np.float64)

    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)

    # Integrality: x_i must be integer (0 or 1)
    integrality = np.ones(n_vars, dtype=np.int8)

    best_solution = None

    # Try MILP first
    try:
        res = milp(
            c=c,
            constraints=[LinearConstraint(A_ub, ub=b_ub)],
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 40}  # Leave 15s for overhead and return
        )

        if res.status in (0, 1):  # Optimal or best found within limit
            solution = []
            for i in range(n_vars):
                if res.x[i] > 0.5:
                    solution.append(valid_vertices[i])
            if solution:
                best_solution = solution
        else:
            # Solver failed or timed out without a solution
            pass

    except Exception:
        pass

    # If MILP didn't give a good solution, use greedy + local search
    if best_solution is None or len(best_solution) < n_vars // 2:
        # Greedy initialization
        S = set()
        for v in valid_vertices:
            can_add = True
            for u in S:
                if u + v in squares:
                    can_add = False
                    break
            if can_add:
                S.add(v)

        # Local search refinement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = list(S)
            for v in S_list:
                # Try removing v
                S_without = S - {v}
                # Try adding any vertex not in S that doesn't conflict
                for u in valid_vertices:
                    if u not in S_without:
                        can_add = True
                        for w in S_without:
                            if u + w in squares:
                                can_add = False
                                break
                        if can_add:
                            S_new = S_without | {u}
                            if len(S_new) > len(S):
                                S = S_new
                                improved = True
                                break
                if improved:
                    break

        if len(S) > (len(best_solution) if best_solution else 0):
            best_solution = list(S)

    if best_solution is None:
        # Final fallback: simple greedy
        S = set()
        for v in valid_vertices:
            can_add = True
            for u in S:
                if u + v in squares:
                    can_add = False
                    break
            if can_add:
                S.add(v)
        best_solution = list(S)

    if not best_solution:
        best_solution = [valid_vertices[0]]

    return best_solution
