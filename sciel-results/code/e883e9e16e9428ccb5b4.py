"""Program for rank-minimal tensor decomposition of 2x3x7 matrix multiplication."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 by 3x7 matrix.
    
    The tensor T[i,j,k] has T[i*m+j, j*p+k, k*n+i] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    We seek U (n*m x r), V (m*p x r), W (p*n x r) such that
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    The schoolbook decomposition uses rank = n*m*p = 42.
    We attempt to find a lower-rank decomposition through structured search.
    
    Key insight: For matrix multiplication M[n,m,p], the tensor rank is at most n*m*p,
    and known lower bounds exist. For 2x3x7, we search for decompositions with
    half-integer or integer entries.
    
    Strategy: Use the fact that the tensor has a special structure. We can try to
    factor through smaller representations. One approach: use the fact that for
    small dimensions, we can try to find a decomposition where the rank terms
    share structure.
    
    We'll try a systematic search for rank r from a reasonable lower bound up to 42,
    using the constraint that entries should be in {0, ±1/2, ±1, ±3/2, ±2}.
    """
    import time
    import random
    from itertools import product
    
    start_time = time.time()
    deadline = 110  # seconds, leaving margin before 120s limit
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook baseline
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Known lower bound: for M_{n,m,p}, rank >= max(n*m, m*p, p*n) * some factor
    # Trivial lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # But this is a very weak bound. The actual minimum rank for 2x3x7 is unknown
    # to me, so we search from a reasonable point.
    
    # Let's try to find a decomposition with rank less than 42.
    # We'll use a heuristic: try random assignments of entries from a small set
    # and check if the tensor equation is satisfied.
    
    # For efficiency, we'll work with the fact that the tensor is sparse.
    # The tensor has exactly n*m*p = 42 non-zero entries (all 1s) in the 
    # multi-index (i*m+j, j*p+k, k*n+i) space.
    
    # Let's define the target tensor as a set of (row, col, k_idx) triples
    # where row = i*m+j, col = j*p+k, k_idx = k*n+i
    target_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = i * m + j
                col = j * p + k
                kidx = k * n + i
                target_triples.append((row, col, kidx))
    
    # For a decomposition of rank r, we need:
    # sum_{c=0}^{r-1} U[row, c] * V[col, c] * W[kidx, c] = 1 for each (row, col, kidx) in target
    # and = 0 for all other (row, col, kidx) triples (but there are nm*mp*pn total, most are 0)
    
    # Actually, the tensor is defined on all (row, col, kidx) where row in [0,nm), col in [0,mp), kidx in [0,pn).
    # T[row, col, kidx] = 1 if there exist i,j,k such that row=i*m+j, col=j*p+k, kidx=k*n+i, else 0.
    
    # Let's verify which triples are 1:
    # For a given (row, col, kidx):
    #   row = i*m + j => i = row // m, j = row % m
    #   col = j'*p + k => need j' = j (the same j), k = col % p, and col // p should equal j
    #   kidx = k'*n + i => need k' = k (the same k), i' = kidx % n, and kidx // n should equal k
    #   So the triple is 1 iff: (col // p == j) and (kidx // n == k) where i=row//m, j=row%m, k=col%p, and kidx%n==i and kidx//n==k
    
    def is_target_one(row, col, kidx):
        i = row // m
        j = row % m
        k_from_col = col % p
        j_from_col = col // p
        i_from_kidx = kidx % n
        k_from_kidx = kidx // n
        return (j_from_col == j) and (k_from_kidx == k_from_col) and (i_from_kidx == i)
    
    # For a given rank r, we need to find U, V, W with entries in some small set
    # such that for all (row, col, kidx):
    #   sum_c U[row,c] * V[col,c] * W[kidx,c] = 1 if is_target_one(row,col,kidx) else 0
    
    # This is a system of nm*mp*pn = 6*21*14 = 1764 equations in r*(nm+mp+pn) = r*(6+21+14) = 41r variables.
    
    # For r = 21: 41*21 = 861 variables, 1764 equations. Underdetermined? No, 1764 > 861.
    # For r = 42: 41*42 = 1722 variables, 1764 equations. Still underdetermined slightly.
    
    # The schoolbook solution has r=42 and works. Can we do better?
    
    # Let's try a different approach: exploit the structure.
    # Note that the tensor for M_{n,m,p} can sometimes be decomposed using
    # the tensor product structure.
    
    # For n=2, m=3, p=7:
    # The "i" index has 2 values, "j" has 3, "k" has 7.
    # 
    # One idea: use a decomposition that separates the k-index partially.
    # 
    # Let's try to construct a solution with rank = n*m*p - something.
    # 
    # Actually, let me try a direct search for small improvements.
    # 
    # We know the schoolbook works with rank 42. Let's see if rank 41 works, etc.
    # 
    # For a structured search, let's fix most entries and solve for a few free variables.
    # 
    # Alternative: use the fact that we can use half-integers. This expands the solution space.
    
    # Let's try a random search with a time limit.
    # We'll try to find a decomposition with rank r where r < 42.
    
    def try_rank(r, max_iters=10000):
        """Try to find a rank-r decomposition using random search."""
        # Candidate entry values: 0, ±0.5, ±1, ±1.5, ±2
        vals = [0, 0.5, 1.0, -0.5, -1.0, 1.5, -1.5, 2.0, -2.0]
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Randomly initialize U, V, W
            U = [[random.choice(vals) for _ in range(r)] for _ in range(nm)]
            V = [[random.choice(vals) for _ in range(r)] for _ in range(mp)]
            W = [[random.choice(vals) for _ in range(r)] for _ in range(pn)]
            
            # Check if this satisfies the tensor equation
            # For efficiency, only check the 42 target triples first
            valid = True
            for (row, col, kidx) in target_triples:
                s = 0.0
                for c in range(r):
                    s += U[row][c] * V[col][c] * W[kidx][c]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check a sample of non-target triples (should be 0)
            # There are 1764 total, 42 are target, so 1722 are non-target
            # Check all of them for correctness
            for row in range(nm):
                for col in range(mp):
                    for kidx in range(pn):
                        if is_target_one(row, col, kidx):
                            continue
                        s = 0.0
                        for c in range(r):
                            s += U[row][c] * V[col][c] * W[kidx][c]
                        if abs(s) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try ranks from 30 up to 41 (since 42 is known to work)
    # Start from lower ranks and go up
    for r in range(30, 42):
        result = try_rank(r, max_iters=5000)
        if result is not None:
            return result
    
    # If no better solution found, return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
