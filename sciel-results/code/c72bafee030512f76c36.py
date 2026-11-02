"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

import time
import itertools
import numpy as np
from fractions import Fraction

DEADLINE = 140.0  # safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta[i,j]*delta[j,k]... 
    Actually the tensor is: T[i,j,k] where the multiplication is C[i,k] = sum_j A[i,j]*B[j,k]
    The tensor decomposition: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    where U has shape (n*m, rank), V (m*p, rank), W (p*n, rank)
    and the index convention is W[k*n+i, r].
    So the condition is: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if j is valid (always) 
    Wait, the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k? No.
    
    Actually for matrix multiplication A(n x m) * B(m x p) = C(n x p):
    C[i,k] = sum_j A[i,j] * B[j,k]
    The bilinear form tensor is T[i,j,k] = 1 (it's the identity in the middle index pairing).
    So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    """
    rank = len(U[0])
    # Build check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _rank_of(U, V, W):
    return len(U[0]) if U else 0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best = (rank_sb, U_sb, V_sb, W_sb)
    
    # Try to find lower rank decompositions
    # Strategy: The tensor T[i,j,k]=1 for all i,j,k (since it's just the pairing)
    # We need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Key observation: If we can factor the "1" as a product of three functions,
    # we can reduce rank. For example, if U[i*m+j,c] = a[i,j,c], V[j*p+k,c] = b[j,k,c], W[k*n+i,c] = c[k,i,c]
    # and sum_c a[i,j,c]*b[j,k,c]*c[k,i,c] = 1
    
    # Simple improvement: Use the fact that the tensor is a "product" tensor.
    # T[i,j,k] = 1 = (1)(1)(1) so rank 1 would work if we could, but we can't because
    # the indices are coupled differently.
    
    # Let's try rank r decompositions for r from a low value up.
    # Use a randomized/local search approach with half-integer entries.
    
    def try_random_rank(r, num_trials, time_limit):
        """Try to find a rank-r decomposition with random initialization and local search."""
        deadline = time.time() + time_limit
        best_found = None
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Random initialization with small integers/half-integers
            U = [[0]*r for _ in range(n*m)]
            V = [[0]*r for _ in range(m*p)]
            W = [[0]*r for _ in range(p*n)]
            
            # Initialize with random values from {-1, -0.5, 0, 0.5, 1}
            vals = [-1, -0.5, 0, 0.5, 1]
            for c in range(r):
                for i in range(n):
                    for j in range(m):
                        U[i*m+j][c] = vals[np.random.randint(len(vals))]
                for j in range(m):
                    for k in range(p):
                        V[j*p+k][c] = vals[np.random.randint(len(vals))]
                for k in range(p):
                    for i in range(n):
                        W[k*n+i][c] = vals[np.random.randint(len(vals))]
            
            # Local search: try to improve
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Compute errors
                errors = {}
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(r):
                                s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                            errors[(i,j,k)] = s - 1
                
                # Check if solved
                if all(abs(e) < 1e-9 for e in errors.values()):
                    # Verify exactly with fractions
                    from fractions import Fraction
                    exact_ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = Fraction(0)
                                for c in range(r):
                                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                                if s != 1:
                                    exact_ok = False
                                    break
                            if not exact_ok:
                                break
                        if not exact_ok:
                            break
                    if exact_ok:
                        return (r, U, V, W)
                
                # Pick a random error to fix
                keys = list(errors.keys())
                idx = np.random.randint(len(keys))
                i, j, k = keys[idx]
                err = errors[(i,j,k)]
                
                # Try to adjust one entry
                if abs(err) < 1e-6:
                    continue
                    
                # Try adjusting U[i*m+j][c] for random c
                c = np.random.randint(r)
                old_val = U[i*m+j][c]
                
                # Try nearby values
                for delta in [-1, -0.5, 0.5, 1]:
                    new_val = old_val + delta
                    if new_val in [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]:
                        U[i*m+j][c] = new_val
                        # Quick check
                        s = 0
                        for cc in range(r):
                            s += U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc]
                        if abs(s - 1) < abs(err) or abs(s - 1) < 1e-9:
                            break
                        else:
                            U[i*m+j][c] = old_val
        
        return None
    
    # Try increasing ranks from a lower bound
    # Lower bound: max(n*m, m*p, p*n) / something... 
    # Actually the rank must be at least max(n, m, p) for non-trivial cases, but let's be more careful.
    # The tensor is T[i,j,k]=1, which has rank 1 in the "unconstrained" sense, but with the 
    # specific index structure, we need more.
    
    # Let's try ranks from 8 upward (Strassen-like gives 7 for 2x2x2, so for larger we expect more)
    # For 2x4x7, schoolbook is 56. Let's try to get well below that.
    
    # First, try a structured approach:
    # Partition j into two groups of 2: j in {0,1} and j in {2,3}
    # For each group, we have a 2x2x7 multiplication.
    # 2x2x7 can be done with rank... let's think.
    # C[i,k] = sum_{j in group} A[i,j] * B[j,k]
    # This is a 2x2 matrix times a 2x7 matrix.
    # Schoolbook: 4*7 = 28 per group, 56 total.
    # But we can use Strassen on the 2x2 part: 7 multiplications, each producing a 7-vector.
    # So 7*7 = 49 per group? No, that's not right either.
    
    # Actually, for 2x2 * 2x7:
    # Let A = [[a,b],[c,d]], B = [[b11,b12,...,b17],[b21,b22,...,b27]]
    # C = A*B, so C[i,k] = sum_j A[i,j]*B[j,k]
    # Using Strassen: 7 products of (2x2) combinations, each product is a scalar times a 7-vector.
    # So rank = 7 * 1 = 7 for the "structure", but each of the 7 products needs to be distributed
    # across the 7 output positions. So total rank = 7 * 7 = 49? No...
    
    # Let me just do the random search.
    
    for r in range(10, 50):
        if time.time() > DEADLINE - 30:
            break
        time_left = DEADLINE - time.time() - 30
        if time_left < 5:
            break
        result = try_random_rank(r, num_trials=max(1, int(time_left / 0.5)), time_limit=time_left - 1)
        if result is not None:
            best = result
            break  # Found a valid decomposition at this rank
    
    # Return best
    _, U, V, W = best
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
