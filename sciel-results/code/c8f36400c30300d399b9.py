"""Improved program: use MILP/ILP for maximum weight independent set on coprime graph."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd

def _build_conflict_and_weights():
    """
    Build the conflict graph and weights.
    Nodes are integers 1..511.
    Weight of node a is 1/(512 - a).
    Edge between a and b if gcd(a,b) > 1.
    We want to select a subset with no edges (independent set) maximizing sum of weights.
    """
    n = 511
    # weights[i] for node i (1-indexed, but we'll use 0-indexed internally: node i represents i+1)
    weights = [1.0 / (512 - (i + 1)) for i in range(n)]
    
    # Conflict: for each node i (value i+1), which nodes j (value j+1) have gcd > 1?
    # We'll build adjacency as a list of lists for the ILP formulation.
    # Actually, for ILP we need constraints: x_i + x_j <= 1 for each conflicting pair.
    # This can be many constraints. Let's think of a smarter way.
    
    # Alternative: Use the fact that if a number has prime factors, it conflicts with all multiples of those primes.
    # But the ILP formulation is standard.
    
    # Let's build the list of conflicting pairs.
    conflicts = []
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            if gcd(a, b) > 1:
                conflicts.append((i, j))
    
    return weights, conflicts, n

def _solve_ilp(weights, conflicts, n, deadline):
    """
    Solve the maximum weight independent set problem using ILP.
    x_i in {0,1}, x_i + x_j <= 1 for each conflict (i,j), maximize sum w_i x_i.
    """
    from scipy.optimize import linprog
    
    # Variables: x_0, ..., x_{n-1}
    # Maximize: sum w_i x_i
    # Constraints: x_i + x_j <= 1 for each conflict
    #              0 <= x_i <= 1
    #              x_i integer
    
    # linprog minimizes, so we minimize -sum w_i x_i
    c = [-w for w in weights]
    
    # Inequality constraints: A_ub @ x <= b_ub
    # For each conflict (i, j): x_i + x_j <= 1
    # This is a large matrix. Let's build it as a sparse matrix for efficiency.
    from scipy.sparse import lil_matrix, csr_matrix
    
    m = len(conflicts)
    A_ub = lil_matrix((m, n), dtype=float)
    b_ub = [1.0] * m
    
    for k, (i, j) in enumerate(conflicts):
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
    
    A_ub = A_ub.tocsr()
    
    # Bounds: 0 <= x_i <= 1
    bounds = [(0, 1)] * n
    
    # Solve LP relaxation first to get an upper bound and initial solution
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        # Fallback to greedy
        return _greedy_solve(weights, conflicts, n)
    
    # The LP solution may not be integer. We need to round or use integer programming.
    # Since SciPy's linprog doesn't support integer constraints directly, 
    # we'll use a branch-and-bound approach or try rounding.
    
    # For this problem, the LP relaxation might be close to integer.
    # Let's try a simple rounding strategy: sort by LP value, pick greedily.
    
    # Actually, let's implement a simple branch and bound or use the LP solution as a guide.
    
    # Better approach: Use the LP solution to guide a greedy selection.
    # Sort nodes by LP value (descending), pick if no conflict with already picked.
    
    lp_sol = res.x
    # Sort indices by LP value descending
    order = sorted(range(n), key=lambda i: -lp_sol[i])
    
    chosen = []
    chosen_set = set()
    for i in order:
        if lp_sol[i] < 1e-6:  # Skip nodes with near-zero LP value
            continue
        # Check if i conflicts with any chosen node
        conflict = False
        for j in chosen:
            if gcd(i + 1, j + 1) > 1:
                conflict = True
                break
        if not conflict:
            chosen.append(i)
            chosen_set.add(i)
    
    if not chosen:
        return _greedy_solve(weights, conflicts, n)
    
    return [i + 1 for i in chosen]

def _greedy_solve(weights, conflicts, n):
    """
    Greedy fallback: sort by weight descending, pick if no conflict.
    """
    order = sorted(range(n), key=lambda i: -weights[i])
    chosen = []
    chosen_set = set()
    
    # For efficiency, precompute conflicts for each node
    # conflicts is a list of (i, j) pairs
    # Let's build an adjacency list
    adj = [set() for _ in range(n)]
    for i, j in conflicts:
        adj[i].add(j)
        adj[j].add(i)
    
    for i in order:
        # Check if i conflicts with any chosen node
        conflict = False
        for j in chosen:
            if j in adj[i]:
                conflict = True
                break
        if not conflict:
            chosen.append(i)
            chosen_set.add(i)
    
    return [i + 1 for i in chosen]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    weights, conflicts, n = _build_conflict_and_weights()
    
    # Check time
    if time.time() > deadline - 10.0:
        # Not enough time, use simple greedy
        return _greedy_solve(weights, conflicts, n)
    
    # Try ILP
    try:
        result = _solve_ilp(weights, conflicts, n, deadline)
        return result
    except Exception:
        # Fallback to greedy
        return _greedy_solve(weights, conflicts, n)

# EVOLVE-BLOCK-END
