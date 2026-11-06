"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    n = 128
    # Precompute GCD matrix
    gcd_mat = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            g = math.gcd(i + 1, j + 1)
            gcd_mat[i, j] = g
            gcd_mat[j, i] = g

    # Identify conflicting triples
    # A triple (i, j, k) is invalid if gcd(i,j) == gcd(i,k) == gcd(j,k)
    # We need to find all such triples to create constraints: x_i + x_j + x_k <= 2
    # For efficiency, we can iterate. n=128, C(128,3) is ~350k, which is fast.
    
    # However, generating 350k constraints might be heavy for MILP if done naively.
    # Let's check the number of invalid triples first.
    invalid_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            g_ij = gcd_mat[i, j]
            if g_ij == 1:
                continue # If gcd is 1, it's hard for all 3 to be 1 unless all are 1, but distinct elements in [1,128] only have gcd 1 if coprime.
                         # Actually, if gcd(i,j)=1, then for the triple to be invalid, we need gcd(i,k)=1 and gcd(j,k)=1.
                         # This means k is coprime to both i and j.
            for k in range(j + 1, n):
                if gcd_mat[i, k] == g_ij and gcd_mat[j, k] == g_ij:
                    invalid_triples.append((i, j, k))
    
    # If there are too many constraints, MILP might be slow.
    # Let's estimate: In [1, 128], how many such triples?
    # For g=1: i,j,k pairwise coprime.
    # For g>1: i,j,k share a common factor g? No, just pairwise gcds equal.
    # Example: 6, 10, 15. gcd(6,10)=2, gcd(6,15)=3, gcd(10,15)=5. Not equal.
    # Example: 2, 4, 6. gcd(2,4)=2, gcd(2,6)=2, gcd(4,6)=2. Equal!
    # So we are looking for sets where all three pairwise gcds are the same value g.
    # This implies i=g*a, j=g*b, k=g*c where gcd(a,b)=gcd(a,c)=gcd(b,c)=1?
    # Wait. If gcd(i,j)=g, then i=g*a, j=g*b with gcd(a,b)=1.
    # If gcd(i,k)=g, then k=g*c with gcd(a,c)=1.
    # If gcd(j,k)=g, then gcd(b,c)=1.
    # So a,b,c must be pairwise coprime.
    # And i,j,k are distinct multiples of g.
    
    # Let's count invalid_triples length.
    # If len(invalid_triples) is manageable (< 100k), MILP is a good bet.
    # With n=128, C(128,3) ~ 348k. The subset of invalid ones is likely much smaller.
    
    if len(invalid_triples) > 50000:
        # Fallback to a fast heuristic if constraints are too numerous for quick MILP
        # But let's try MILP first as it's a structural change.
        # If it takes too long, we might need to prune or use a different method.
        # Given the 85s deadline, if MILP is slow, we might timeout.
        # Let's add a time check inside the loop? No, MILP is a single call.
        # We can set a time limit in MILP? No direct simple time limit in scipy.milp that guarantees return.
        # But HiGHS (default) is fast.
        
        pass

    # Build MILP model
    # Variables: x_i in {0, 1}
    # Objective: maximize sum(x_i) -> minimize -sum(x_i)
    # Constraints: x_i + x_j + x_k <= 2 for all invalid triples (i,j,k)
    
    c = np.full(n, -1.0) # Minimize -sum(x)
    
    if len(invalid_triples) == 0:
        return set(range(1, 129))
        
    # Create constraint matrix
    # Rows = len(invalid_triples), Cols = n
    # This could be large. Let's use sparse matrix.
    row_idx = []
    col_idx = []
    data = []
    
    for i, (a, b, c_idx) in enumerate(invalid_triples):
        row_idx.extend([i, i, i])
        col_idx.extend([a, b, c_idx])
        data.extend([1, 1, 1])
        
    A = coo_matrix((data, (row_idx, col_idx)), shape=(len(invalid_triples), n)).tocsr()
    
    # Constraint: A @ x <= 2
    cons = LinearConstraint(A, ub=2)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: x_i are integers
    integrality = np.ones(n)
    
    try:
        # Use milp
        # If this takes too long, we can't easily interrupt.
        # But for n=128 and moderate constraints, HiGHS should be fast (< 1s).
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
        
        if res.success:
            # Extract solution
            selected = set()
            for i in range(n):
                if res.x[i] > 0.5:
                    selected.add(i + 1)
            
            # Verify validity just in case
            if _is_valid(selected):
                return selected
            else:
                # Fallback if solver gave weird result
                return _greedy_fallback()
        else:
            return _greedy_fallback()
            
    except Exception:
        return _greedy_fallback()

def _is_valid(s):
    s_list = list(s)
    for i in range(len(s_list)):
        for j in range(i + 1, len(s_list)):
            for k in range(j + 1, len(s_list)):
                a, b, c = s_list[i], s_list[j], s_list[k]
                if math.gcd(a, b) == math.gcd(a, c) == math.gcd(b, c):
                    return False
    return True

def _greedy_fallback():
    selected = []
    for x in range(1, 129):
        if all(len({math.gcd(a,b), math.gcd(a,x), math.gcd(b,x)}) > 1 for a,b in combinations(selected,2)):
            selected.append(x)
    return set(selected)

from itertools import combinations
# EVOLVE-BLOCK-END
