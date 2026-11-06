import math
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start = time.time()
    deadline = 85.0  # safety margin before the 90s wall-clock kill

    n = 256
    max_sum = 2 * n
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1

    # Diagonal constraint: 2*a must never be a square.
    candidates = [a for a in range(1, n + 1) if 2 * a not in squares]
    m = len(candidates)
    idx_map = {a: i for i, a in enumerate(candidates)}

    # Build conflict adjacency matrix (symmetric, no self-loops needed since diagonal handled)
    # A[i,j] = 1 if candidates[i] + candidates[j] is a perfect square
    A = np.zeros((m, m), dtype=np.int8)
    for i in range(m):
        ai = candidates[i]
        for j in range(i + 1, m):
            aj = candidates[j]
            if ai + aj in squares:
                A[i, j] = 1
                A[j, i] = 1

    # MILP formulation:
    # Maximize sum(x_i)
    # Subject to:
    #   x_i + x_j <= 1 for all (i,j) with A[i,j] == 1
    #   x_i in {0, 1}
    
    # c: objective coefficients (all 1)
    c = np.ones(m, dtype=np.float64)
    
    # Constraints: for each edge (i,j), x_i + x_j <= 1
    # Number of constraints = number of edges
    edges = np.argwhere(A == 1)
    # Only take upper triangle to avoid duplicates
    edges = edges[edges[:, 0] < edges[:, 1]]
    num_edges = len(edges)
    
    if num_edges == 0:
        # No conflicts, take all
        return sorted(candidates)
    
    # Build constraint matrix: each row is one constraint x_i + x_j <= 1
    # LinearConstraint(A_ub, b_ub) where A_ub @ x <= b_ub
    A_ub = np.zeros((num_edges, m), dtype=np.float64)
    for k, (i, j) in enumerate(edges):
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
    b_ub = np.ones(num_edges, dtype=np.float64)
    
    # Bounds: 0 <= x_i <= 1
    lb = np.zeros(m, dtype=np.float64)
    ub = np.ones(m, dtype=np.float64)
    
    # Integrality: x_i in {0, 1}
    integrality = np.ones(m, dtype=np.int8)
    
    # Time limit for HiGHS: leave 10s for overhead and verification
    time_limit = max(1.0, deadline - time.time() - 10.0)
    
    try:
        result = milp(
            c=c,
            constraints=LinearConstraint(A_ub, -np.inf, b_ub),
            integrality=integrality,
            bounds=Bounds(lb, ub),
            options={"time_limit": time_limit, "mip_rel_gap": 0.0}
        )
    except Exception:
        result = None
    
    if result is not None and result.status == 0:
        # Extract solution
        x = result.x
        solution = set()
        for i in range(m):
            if x[i] > 0.5:
                solution.add(candidates[i])
    else:
        # Fallback: greedy min-degree
        rem = set(range(m))
        greedy = []
        while rem:
            best = None
            bd = None
            for node in rem:
                d = sum(1 for v in rem if A[node, v] == 1)
                if bd is None or d < bd:
                    bd, best = d, node
                    if d == 0:
                        break
            greedy.append(best)
            rem.discard(best)
            rem -= set(np.where(A[best] == 1)[0])
        solution = set(candidates[i] for i in greedy)
    
    # Verification
    ok = True
    lst = sorted(solution)
    for i in range(len(lst)):
        a = lst[i]
        if 2 * a in squares:
            ok = False
            break
        for j in range(i + 1, len(lst)):
            if a + lst[j] in squares:
                ok = False
                break
        if not ok:
            break
    
    if not ok:
        # Fallback: deterministic min-degree greedy
        rem = set(range(m))
        fb = []
        while rem:
            mn = None
            md = None
            for node in rem:
                d = sum(1 for v in rem if A[node, v] == 1)
                if md is None or d < md:
                    md, mn = d, node
            fb.append(mn)
            rem.discard(mn)
            rem -= set(np.where(A[mn] == 1)[0])
        solution = set(candidates[i] for i in fb)
    
    return sorted(solution)
