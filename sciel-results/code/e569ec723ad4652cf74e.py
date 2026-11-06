import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    deadline = time.time() + 85  # Safety margin for 90s limit
    
    N = 180
    universe = list(range(1, N + 1))
    
    # 1. Precompute all "bad" triples {a, b, c} such that 1/a = 1/b + 1/c
    # This implies bc = a(b+c)
    # We assume a < b < c for uniqueness, but the equation is symmetric in b and c.
    # a must be the smallest? 
    # 1/a = 1/b + 1/c > 1/c => a < c.
    # 1/a = 1/b + 1/c > 1/b => a < b.
    # So a is always the smallest element in the triple.
    # We iterate a from 1 to N.
    # For a given a, we need b, c such that 1/b + 1/c = 1/a.
    # (b+c)/(bc) = 1/a => a(b+c) = bc => bc - ab - ac = 0 => (b-a)(c-a) = a^2.
    # So we need to find divisors of a^2.
    # Let x = b-a, y = c-a. Then xy = a^2.
    # b = a+x, c = a+y.
    # We need b, c <= N.
    # Also b != c is required for "distinct elements"? 
    # The task says "no THREE DISTINCT elements". So a, b, c must be distinct.
    # If x=y, then b=c, which is not allowed for distinct elements.
    # So we skip x=y.
    
    bad_triples = set()
    
    for a in range(1, N + 1):
        a2 = a * a
        # Find divisors x of a^2 such that x < a^2/x (to ensure x < y => b < c) 
        # and b, c <= N.
        # x can range from 1 to sqrt(a^2) = a.
        for x in range(1, int(np.sqrt(a2)) + 1):
            if a2 % x == 0:
                y = a2 // x
                if x == y:
                    continue # b == c, not distinct
                b = a + x
                c = a + y
                if b <= N and c <= N:
                    # Ensure a, b, c are distinct (a < b < c guaranteed by x,y > 0)
                    bad_triples.add((a, b, c))
                # Note: we don't need to add (a, c, b) because the set is unordered
                # and the constraint is symmetric on b,c.
    
    # 2. Formulate MILP
    # Variables: x_1, ..., x_N (index 0 to N-1)
    # Maximize sum(x_i)
    # Constraints: for each (a, b, c) in bad_triples: x_{a-1} + x_{b-1} + x_{c-1} <= 2
    # Bounds: 0 <= x_i <= 1
    # Integrality: x_i is integer
    
    n_vars = N
    c = np.ones(n_vars) # Maximize sum, so minimize -sum
    
    # If no bad triples, return all
    if not bad_triples:
        return universe
        
    # Build constraint matrix
    # We can use sparse matrix for efficiency, but N=180 is small.
    # Number of constraints is len(bad_triples).
    n_constraints = len(bad_triples)
    
    if n_constraints == 0:
        return universe

    # A is n_constraints x n_vars
    # Using list of lists for scipy
    A_list = []
    for (a, b, c) in bad_triples:
        row = [0] * N
        row[a-1] = 1
        row[b-1] = 1
        row[c-1] = 1
        A_list.append(row)
        
    A = np.array(A_list)
    b_ub = np.full(n_constraints, 2)
    
    constraints = LinearConstraint(A, ub=b_ub)
    integrality = np.ones(n_vars)
    bounds = Bounds(0, 1)
    
    best_set = []
    best_size = 0
    
    # Try to solve MILP
    try:
        # Use HiGHS solver (default in scipy)
        # time_limit in seconds
        # Allocate most of the time to MILP, leaving some for fallback if needed
        # But if MILP returns early with optimal, we are done.
        # If it times out, we use incumbent.
        res = milp(c=-c, constraints=constraints, integrality=integrality, bounds=bounds, options={"time_limit": 75})
        
        if res.status == 0: # Optimal
            # res.x is the optimal solution
            x_opt = res.x
            # Due to floating point, round to nearest integer
            x_int = np.round(x_opt).astype(int)
            # Verify validity and extract set
            # Check if any constraint is violated significantly
            # A @ x_int <= 2
            if np.all(A @ x_int <= 2 + 1e-5):
                candidate_set = [i+1 for i in range(N) if x_int[i] == 1]
                if len(candidate_set) > 0:
                    return candidate_set
            else:
                # Fallback to rounding heuristic if strict check fails (shouldn't happen)
                candidate_set = [i+1 for i in range(N) if x_int[i] == 1]
                # Greedy repair?
                # Just use the LP solution rounded
                pass
        elif res.status == 2: # Time limit
             # Use the current best incumbent if available
             if hasattr(res, 'x') and res.x is not None:
                 x_int = np.round(res.x).astype(int)
                 if np.all(A @ x_int <= 2 + 1e-5):
                     candidate_set = [i+1 for i in range(N) if x_int[i] == 1]
                     if len(candidate_set) > 0:
                         return candidate_set
    except Exception as e:
        pass # Fallback to greedy
        
    # Fallback: Fast Greedy with precomputed conflicts
    # We want to maximize size.
    # Heuristic: Process elements in an order that minimizes future conflicts.
    # Elements with fewer conflicts are "safer" to include.
    # Or, elements that are in many triples should be excluded?
    # Let's try: Sort by number of triples they are in (ascending).
    # Add them if no conflict is created.
    
    # Precompute for each element i, the list of pairs (j, k) such that {i, j, k} is a bad triple.
    # Actually, we just need to check if adding x creates a triple.
    # A triple is formed if we add x, and there exist u, v in S such that {x, u, v} is bad.
    # We can precompute for each x, the set of "forbidden pairs" (u, v) in S.
    # Forbidden pairs for x: pairs (u, v) such that {x, u, v} is a bad triple.
    
    # Let's build a dictionary: x -> set of (u, v)
    # Note: (u, v) is unordered.
    conflict_pairs_for = {i: set() for i in range(1, N+1)}
    
    for (a, b, c) in bad_triples:
        # a is smallest.
        # If we add a, we can't have (b, c) in S.
        conflict_pairs_for[a].add((b, c))
        # If we add b, we can't have (a, c) in S.
        conflict_pairs_for[b].add((a, c))
        # If we add c, we can't have (a, b) in S.
        conflict_pairs_for[c].add((a, b))
        
    # Sort elements by degree (number of conflict pairs) ascending?
    # Or just try a few random orders.
    
    def greedy_solve(order):
        S = set()
        for x in order:
            # Check if any forbidden pair for x is in S
            is_conflict = False
            for (u, v) in conflict_pairs_for[x]:
                if u in S and v in S:
                    is_conflict = True
                    break
            if not is_conflict:
                S.add(x)
        return S

    # Try a few deterministic orders
    # 1. Ascending
    # 2. Descending
    # 3. Random
    
    orders = [
        list(range(1, N+1)),
        list(range(N, 0, -1)),
        random.sample(range(1, N+1), N)
    ]
    
    for order in orders:
        S = greedy_solve(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            
    # Local search: try to add removed elements?
    # The greedy above doesn't explicitly track removed ones, but we can try to add elements not in best_set.
    # Actually, let's just do a few random restarts if time permits.
    
    start_time = time.time()
    while time.time() < deadline:
        order = random.sample(range(1, N+1), N)
        S = greedy_solve(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            
    return best_set
