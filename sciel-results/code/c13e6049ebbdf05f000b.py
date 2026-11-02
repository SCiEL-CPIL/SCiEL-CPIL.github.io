"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    U: shape (n*m, rank), V: shape (m*p, rank), W: shape (p*n, rank)
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    should equal 1 for all i,j,k (since A[i,j]*B[j,k] = sum_k ... wait no).
    
    Actually for matrix multiplication C = A * B where A is n x m, B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor for this operation has indices (i, j, k) where the structure is:
    T_{i,j,k} such that C[i,k] = sum_j T_{i,j,k} * A[i,j] * B[j,k]
    
    In the standard tensor decomposition formulation:
    We need: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{j is the shared index}
    
    Wait, let me reconsider. The matrix multiplication tensor M_{i,j,k} where:
    - i indexes rows of A (0..n-1)
    - j indexes columns of A / rows of B (0..m-1)  
    - k indexes columns of B (0..p-1)
    
    The tensor entry M_{i,j,k} = 1 if this corresponds to a valid multiplication term, else 0.
    Actually, the matrix multiplication tensor is defined such that:
    (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    
    In tensor form, we write this as a 3-tensor T with T[i,j,k] = 1 for all valid (i,j,k)
    because each term A[i,j]*B[j,k] contributes to C[i,k].
    
    The decomposition requires: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    Where following the convention in the task:
    U has shape (n*m, rank) indexed by (i*m + j)
    V has shape (m*p, rank) indexed by (j*p + k)
    W has shape (p*n, rank) indexed by (k*n + i)
    
    So we need: for all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _try_rank(rank, n, m, p, deadline, seed=0):
    """
    Try to find a rank-rank decomposition using random search with rational entries.
    Returns (U, V, W) if found, else None.
    """
    import random
    rng = random.Random(seed)
    
    # We'll use a simple approach: try random rational matrices and check
    # For small ranks, we can be more systematic
    
    # The known lower bound for 2x3 x 3x4 matrix multiplication is at least 
    # max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # The schoolbook rank is 2*3*4 = 24
    # Known results suggest the rank might be around 16-20
    
    # Strategy: Use a parameterized family of decompositions
    # For matrix multiplication, one common approach is to use the formula:
    # (a+b)(c+d) = ac + ad + bc + bd
    # which allows saving one multiplication.
    
    # For 2x3 x 3x4, let's think about this differently.
    # A is 2x3, B is 3x4.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3] + A[0,2]*B[2,3]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # C[1,3] = A[1,0]*B[0,3] + A[1,1]*B[1,3] + A[1,2]*B[2,3]
    
    # Let's try a direct search for small ranks using random initialization
    # and checking. For rank r, we need to find U (6xr), V (12xr), W (8xr)
    
    # Use a smarter approach: exploit the structure
    # The matrix multiplication tensor has a lot of symmetry.
    
    # Let's try rank from 12 up to 24, and for each rank try random searches
    
    # Actually, let me try a more structured approach.
    # For multiplying (n x m) by (m x p), one can sometimes achieve rank < n*m*p
    # using algorithms based on fast matrix multiplication ideas, but for these
    # small sizes, let's try random search with rational numbers.
    
    max_attempts = 10000
    attempt = 0
    while attempt < max_attempts:
        if time.time() > deadline:
            return None
        
        # Generate random rational matrices with small integer numerators/denominators
        def rand_frac():
            num = rng.randint(-4, 4)
            den = rng.randint(1, 4)
            return Fraction(num, den)
        
        U = [[rand_frac() for _ in range(rank)] for _ in range(n * m)]
        V = [[rand_frac() for _ in range(rank)] for _ in range(m * p)]
        W = [[rand_frac() for _ in range(rank)] for _ in range(p * n)]
        
        if _verify_decomposition(U, V, W, n, m, p):
            # Convert to float representation
            U_f = [[float(x) for x in row] for row in U]
            V_f = [[float(x) for x in row] for row in V]
            W_f = [[float(x) for x in row] for row in W]
            return (U_f, V_f, W_f)
        
        attempt += 1
    
    return None

def _try_structured_rank(rank, n, m, p, deadline):
    """
    Try structured decompositions based on known algebraic identities.
    """
    # For 2x3 x 3x4, let's think about whether we can use a block structure.
    # 
    # One idea: split the 3x4 matrix B into two 3x2 blocks.
    # Then C[:,0:2] = A * B[:,0:2] and C[:,2:4] = A * B[:,2:4]
    # Each is a 2x3 * 3x2 multiplication.
    # 
    # For 2x3 * 3x2, schoolbook rank is 12.
    # Can we do better? The lower bound is max(6,6,4)=6.
    # 
    # Actually, let me try a different approach. Let's use the fact that
    # we can combine terms.
    
    # Let's try rank 16 first with a structured approach.
    # 
    # Actually, let me just do a more careful random search with more attempts
    # and smaller value ranges to increase the chance of finding exact solutions.
    
    import random
    rng = random.Random(42)
    
    # Try with integer entries first (small range)
    for val_range in [2, 3, 4, 5]:
        if time.time() > deadline:
            return None
        
        for attempt in range(5000):
            U = [[rng.randint(-val_range, val_range) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.randint(-val_range, val_range) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.randint(-val_range, val_range) for _ in range(rank)] for _ in range(p * n)]
            
            # Quick check: verify a few random (i,j,k) triples first
            valid = True
            for _ in range(20):
                i = rng.randint(0, n-1)
                j = rng.randint(0, m-1)
                k = rng.randint(0, p-1)
                s = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if s != 1:
                    valid = False
                    break
            
            if valid:
                # Full verification
                if _verify_decomposition(U, V, W, n, m, p):
                    U_f = [[float(x) for x in row] for row in U]
                    V_f = [[float(x) for x in row] for row in V]
                    W_f = [[float(x) for x in row] for row in W]
                    return (U_f, V_f, W_f)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Start with the best known (schoolbook) rank = 24
    best_rank = n * m * p
    best_result = None
    
    # Create schoolbook decomposition as fallback
    rank = best_rank
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
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find better ranks
    # Lower bound is max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # Try ranks from 12 up to 23
    for r in range(12, best_rank):
        if time.time() > deadline - 5:
            break
        
        # Try structured approach
        result = _try_structured_rank(r, n, m, p, deadline - 5)
        if result is not None:
            best_rank = r
            best_result = {"U": result[0], "V": result[1], "W": result[2]}
            break
        
        # Try random search with fractions
        if time.time() > deadline - 5:
            break
        result = _try_rank(r, n, m, p, deadline - 5, seed=r)
        if result is not None:
            best_rank = r
            best_result = {"U": result[0], "V": result[1], "W": result[2]}
            break
    
    return best_result
# EVOLVE-BLOCK-END
