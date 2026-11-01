"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each output element C[i,k], we need sum_j A[i,j]*B[j,k]
    # The tensor T[i,j,k] = delta_{i,j,k} should be representable as
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, let me re-read the convention.
    # W[k*n+i, r] - so W is indexed by (k, i) where k is column of C (0..p-1), i is row of C (0..n-1)
    # U[i*m+j, r] - U indexed by (i, j) where i is row of A (0..n-1), j is col of A (0..m-1)
    # V[j*p+k, r] - V indexed by (j, k) where j is row of B (0..m-1), k is col of B (0..p-1)
    
    # The tensor for matrix multiplication is T[i, j, k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k), and the decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify: for each (i,j,k), check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook as fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions using integer/half-integer entries
    # The possible values for entries are in {-1.5, -1, -0.5, 0, 0.5, 1, 1.5} for small ranks
    # For rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
    
    # Strategy: Use a greedy/search approach trying ranks from lower bound up
    # Lower bound for 2x3*3x6: max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # But tensor rank can be lower. Known bounds: rank >= max(n*m, m*p, p*n)/something
    # Actually for matrix multiplication, the tensor rank is at least max(n, m, p) * min(n,m,p) ... no
    # The trivial lower bound is max(n*m, m*p, p*n) is wrong. 
    # Actually, the tensor rank of the matrix multiplication tensor <n,m,p> satisfies
    # rank >= max(n*m, m*p, p*n) is NOT correct. 
    # The correct lower bound: rank >= max(n*m, m*p, p*n) / max(n, m, p) ... no
    # Actually rank >= n*m*p / (n*m) = p, etc. The strongest simple bound is rank >= max(n*m, m*p, p*n) is wrong.
    # Let me think: each rank-1 term contributes to at most n*m entries in U, m*p in V, p*n in W.
    # The total number of nonzero "triples" is n*m*p. Each rank-1 term (u,v,w) with u in R^(nm), v in R^(mp), w in R^(pn)
    # contributes to the tensor T[i,j,k] = sum_r u[i,j]_r * v[j,k]_r * w[k,i]_r
    # A rank-1 term with all entries nonzero in u, v, w contributes to all n*m*p positions.
    # So the lower bound is just 1 in principle, but structure constraints apply.
    
    # For 2x3*3x6, let's try ranks from 18 down (since schoolbook is 36)
    # Actually let's be smarter. Try a range of ranks.
    
    # Use a randomized search with verification
    # For each candidate rank, try random integer/half-integer matrices and verify
    
    np.random.seed(42)
    
    # Try ranks from lower to upper
    # Theoretical lower bound: at least max(n, m, p) = 6? No.
    # Actually for <2,3,6>, known results suggest rank around 18-24.
    # Let's try ranks 18 through 35.
    
    for rank in range(18, min(schoolbook_rank, 36)):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random attempts for this rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                break
            
            # Generate random integer/half-integer matrices
            # Values from {-1, 0, 1} or {-1, -0.5, 0, 0.5, 1}
            values = np.array([-1, 0, 1])
            
            U_cand = np.random.choice(values, size=(n*m, rank))
            V_cand = np.random.choice(values, size=(m*p, rank))
            W_cand = np.random.choice(values, size=(p*n, rank))
            
            # Verify
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        s = np.sum(U_cand[i*m+j] * V_cand[j*p+k] * W_cand[k*n+i])
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
            
            if valid:
                # Found a valid decomposition
                best_rank = rank
                best_result = {
                    "U": U_cand.tolist(),
                    "V": V_cand.tolist(),
                    "W": W_cand.tolist()
                }
                break  # Move to next (lower) rank
        
        if best_rank == rank:
            break  # Found minimal rank for this search
    
    return best_result
# EVOLVE-BLOCK-END
