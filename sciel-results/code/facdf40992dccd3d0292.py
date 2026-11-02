"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True

def _try_random_rank(n, m, p, rank, max_time=120.0):
    """Try to find a rank-r decomposition using random initialization and verification."""
    start_time = time.time()
    deadline = start_time + max_time
    
    n_rows_U = n * m
    n_rows_V = m * p
    n_rows_W = p * n
    
    # The tensor T[a,b,c] = 1 for all (i,j,k) where a=i*m+j, b=j*p+k, c=k*n+i
    # We need: T[a,b,c] = sum_l U[a,l] V[b,l] W[c,l]
    
    # Strategy: Use the fact that for each fixed j, we have a bilinear map
    # Actually, let's use a structured approach based on the matrix multiplication structure
    
    # For matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has entries T[i,j,k] = 1 (it's the identity in the appropriate sense)
    
    # A rank-r decomposition means we can write the multiplication using r scalar multiplications
    # Each scalar multiplication involves a linear combination of A entries, a linear combination
    # of B entries, and contributes to a linear combination of C entries.
    
    # Let's try: U[i*m+j, c] = coefficient of A[i,j] in the c-th scalar product
    #           V[j*p+k, c] = coefficient of B[j,k] in the c-th scalar product  
    #           W[k*n+i, c] = coefficient of C[i,k] that receives the c-th scalar product
    
    # For the decomposition to work:
    # sum_c U[i*m+j,c] * V[j'*p+k,c] * W[k'*n+i',c] should equal
    # 1 if (i,j,k) is the correct pattern, 0 otherwise
    # Actually it's: for the multiplication to work, we need
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all valid (i,j,k)
    # And the structure ensures that cross terms cancel.
    
    # Let's use a different approach: try to find decompositions of the form
    # where we group by the middle index j
    
    # For each j, the contribution to C[i,k] from A[i,j]*B[j,k] must be captured
    # We can try: for each c, U[:,c], V[:,c], W[:,c] are such that
    # the product U[i*m+j,c]*V[j*p+k,c] is "aligned" with W[k*n+i,c]
    
    # A practical approach: use random rank-r factors and check
    # But we need to be smart about the structure.
    
    # Let's try: fix the structure so that for each c, the support of
    # U[:,c], V[:,c], W[:,c] is consistent with a valid (i,j,k) pattern
    
    # Actually, let's just try random dense factors and verify
    # For small ranks, this might work with enough random trials
    
    best_result = None
    attempts = 0
    
    while time.time() < deadline - 5.0:
        attempts += 1
        
        # Generate random factors with entries from {-1, 0, 1} or small rationals
        # Use a mix of sparse and dense
        if random.random() < 0.5:
            # Sparse: each row has 1-3 non-zero entries
            U_np = np.zeros((n_rows_U, rank))
            V_np = np.zeros((n_rows_V, rank))
            W_np = np.zeros((n_rows_W, rank))
            for c in range(rank):
                # Pick random non-zero positions
                for _ in range(random.randint(1, min(3, n_rows_U))):
                    U_np[random.randint(0, n_rows_U-1), c] = random.choice([-1, 1])
                for _ in range(random.randint(1, min(3, n_rows_V))):
                    V_np[random.randint(0, n_rows_V-1), c] = random.choice([-1, 1])
                for _ in range(random.randint(1, min(3, n_rows_W))):
                    W_np[random.randint(0, n_rows_W-1), c] = random.choice([-1, 1])
        else:
            # Dense with small integer entries
            U_np = np.random.randint(-2, 3, (n_rows_U, rank)).astype(float)
            V_np = np.random.randint(-2, 3, (n_rows_V, rank)).astype(float)
            W_np = np.random.randint(-2, 3, (n_rows_W, rank)).astype(float)
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    if abs(total - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Found a valid decomposition!
            # Try to make entries exact (round to nearest half-integer)
            U_exact = [[round(U_np[i, c] * 2) / 2 for c in range(rank)] for i in range(n_rows_U)]
            V_exact = [[round(V_np[i, c] * 2) / 2 for c in range(rank)] for i in range(n_rows_V)]
            W_exact = [[round(W_np[i, c] * 2) / 2 for c in range(rank)] for i in range(n_rows_W)]
            
            # Verify the exact version
            valid_exact = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for c in range(rank):
                            total += U_exact[i*m+j][c] * V_exact[j*p+k][c] * W_exact[k*n+i][c]
                        if abs(total - 1) > 1e-12:
                            valid_exact = False
                            break
                    if not valid_exact:
                        break
                if not valid_exact:
                    break
            
            if valid_exact:
                return U_exact, V_exact, W_exact, rank
            else:
                # Use the floating point version
                return U_np.tolist(), V_np.tolist(), W_np.tolist(), rank
    
    return None

def _try_structured_rank(n, m, p, rank, max_time=120.0):
    """Try structured decompositions based on the matrix multiplication pattern."""
    start_time = time.time()
    deadline = start_time + max_time
    
    n_rows_U = n * m
    n_rows_V = m * p
    n_rows_W = p * n
    
    # Idea: For each c, associate it with a "pattern" of which (i,j,k) it affects
    # The key constraint is that for each (i,j,k), the sum over c of
    # U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Let's try: partition the c indices into groups, each group handling a subset
    # of (i,j,k) triples. Within each group, we can have more flexibility.
    
    # Actually, let's use the known structure of matrix multiplication:
    # The minimal rank for multiplying an n x m matrix by an m x p matrix is
    # at least max(n*m, m*p, p*n) and at most n*m*p.
    # For 3x4 * 4x4: at least max(12,16,12)=16, at most 48.
    
    # Let's try ranks from the theoretical lower bound up
    # But we'll focus on trying to find ANY valid decomposition at a given rank
    
    # Use a smarter initialization: for each c, make U[:,c], V[:,c], W[:,c]
    # correspond to a "virtual" (i_c, j_c, k_c) with some mixing
    
    # Try: U[i*m+j, c] = 1 if j == j_c, else 0 (with some mixing)
    #      V[j*p+k, c] = 1 if j == j_c, else 0
    #      W[k*n+i, c] = 1 if i == i_c and k == k_c, else 0
    
    # This gives the schoolbook when each c corresponds to exactly one (i,j,k)
    # To reduce rank, we need to mix: let multiple (i,j,k) share a c
    
    # Let's try: for each c, pick a random j_c, then allow U and V to have
    # non-zero entries for that j_c, and W to have non-zero entries for 
    # multiple (i,k) pairs
    
    best_result = None
    attempts = 0
    
    while time.time() < deadline - 5.0:
        attempts += 1
        
        U_np = np.zeros((n_rows_U, rank))
        V_np = np.zeros((n_rows_V, rank))
        W_np = np.zeros((n_rows_W, rank))
        
        for c in range(rank):
            # Pick a "center" j for this term
            j_c = random.randint(0, m - 1)
            
            # U: non-zero for row i*m + j_c for random i's
            for i in range(n):
                if random.random() < 0.7:
                    U_np[i * m + j_c, c] = random.choice([-1, 1])
            
            # V: non-zero for row j_c*p + k for random k's
            for k in range(p):
                if random.random() < 0.7:
                    V_np[j_c * p + k, c] = random.choice([-1, 1])
            
            # W: non-zero for row k*n + i for random (i,k) pairs
            for i in range(n):
                for k in range(p):
                    if random.random() < 0.3:
                        W_np[k * n + i, c] = random.choice([-1, 1])
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    if abs(total - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Found it!
            return U_np.tolist(), V_np.tolist(), W_np.tolist(), rank
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find better ranks
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # Try ranks from low to high, with time budget for each
    
    # Allocate time: try lower ranks with less time, higher ranks with more
    # But we want to find the MINIMUM rank, so we should try low ranks first
    
    # Try ranks: 16, 18, 20, 24, 32, 40, 48
    # But we need to be careful about time
    
    time_per_attempt = 3.0  # seconds per rank attempt
    
    ranks_to_try = []
    for r in range(16, 49):
        ranks_to_try.append(r)
    
    # Shuffle to try various ranks, but bias toward lower
    random.shuffle(ranks_to_try)
    ranks_to_try.sort(key=lambda r: (r > 24, r))  # Lower ranks first, then by value
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10.0:
            break
        
        remaining_time = deadline - time.time() - 5.0
        if remaining_time < 2.0:
            break
        
        # Try structured first (faster to generate candidates)
        result = _try_structured_rank(n, m, p, rank, max_time=min(remaining_time, 20.0))
        if result is not None:
            U_r, V_r, W_r, r_found = result
            if r_found < best_rank:
                best_rank = r_found
                best_U, best_V, best_W = U_r, V_r, W_r
                # If we found a valid decomposition, we can stop since we're going in increasing order
                # But we shuffled, so let's continue a bit to see if we can do better
                # Actually, since we sorted, lower ranks come first, so we can break
                break
        
        # Also try random approach
        if time.time() < deadline - 10.0:
            result2 = _try_random_rank(n, m, p, rank, max_time=min(remaining_time, 15.0))
            if result2 is not None:
                U_r, V_r, W_r, r_found = result2
                if r_found < best_rank:
                    best_rank = r_found
                    best_U, best_V, best_W = U_r, V_r, W_r
                    break
    
    # Verify the best result
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Fall back to schoolbook
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = n * m * p
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
