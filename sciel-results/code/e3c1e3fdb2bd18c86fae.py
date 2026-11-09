"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook_decomp(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition for multiplying n x m by m x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i*m+j]
                v_row = V_arr[j*p+k]
                w_row = W_arr[k*n+i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication (rank 7)."""
    # Strassen's 7 multiplication algorithm for 2x2 matrices
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # U indices: i*m+j = i*2+j for i,j in {0,1}
    # V indices: j*p+k = j*2+k for j,k in {0,1}
    # W indices: k*n+i = k*2+i for k,i in {0,1}
    
    # M1 = (A00+A11)*(B00+B11)
    # U[0,0]=1 (A00), U[3,0]=1 (A11)  [i*m+j: 0*2+0=0, 1*2+1=3]
    # V[0,0]=1 (B00), V[3,0]=1 (B11)  [j*p+k: 0*2+0=0, 1*2+1=3]
    # W[0,0]=1 (C00), W[3,0]=1 (C11)  [k*n+i: 0*2+0=0, 1*2+1=3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)*B00
    # U[2,1]=1 (A10), U[3,1]=1 (A11)  [1*2+0=2, 1*2+1=3]
    # V[0,1]=1 (B00)  [0*2+0=0]
    # W[2,1]=1 (C10)  [1*2+0=2]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A00*(B01-B11)
    # U[0,2]=1 (A00)  [0*2+0=0]
    # V[1,2]=1 (B01), V[3,2]=-1 (B11)  [0*2+1=1, 1*2+1=3]
    # W[1,2]=1 (C01)  [0*2+1=1]
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A11*(B10-B00)
    # U[3,3]=1 (A11)  [1*2+1=3]
    # V[2,3]=1 (B10), V[0,3]=-1 (B00)  [1*2+0=2, 0*2+0=0]
    # W[0,3]=1 (C00), W[2,3]=1 (C10)  [0*2+0=0, 1*2+0=2]
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)*B11
    # U[0,4]=1 (A00), U[1,4]=1 (A01)  [0*2+0=0, 0*2+1=1]
    # V[3,4]=1 (B11)  [1*2+1=3]
    # W[3,4]=1 (C11)  [1*2+1=3]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[3][4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    # U[2,5]=1 (A10), U[0,5]=-1 (A00)  [1*2+0=2, 0*2+0=0]
    # V[0,5]=1 (B00), V[1,5]=1 (B01)  [0*2+0=0, 0*2+1=1]
    # W[3,5]=1 (C11)  [1*2+1=3]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    # U[1,6]=1 (A01), U[3,6]=-1 (A11)  [0*2+1=1, 1*2+1=3]
    # V[2,6]=1 (B10), V[3,6]=1 (B11)  [1*2+0=2, 1*2+1=3]
    # W[0,6]=1 (C00)  [0*2+0=0]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}

def _pad_decomp(n, m, p, target_n, target_m, target_p):
    """Pad a decomposition from (n,m,p) to (target_n, target_m, target_p)."""
    base = _strassen_2x2x2() if (n, m, p) == (2, 2, 2) else _schoolbook_decomp(n, m, p)
    rank = len(base["U"][0])
    
    U_new = [[0]*rank for _ in range(target_n * target_m)]
    V_new = [[0]*rank for _ in range(target_m * target_p)]
    W_new = [[0]*rank for _ in range(target_p * target_n)]
    
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                U_new[i*target_m + j][r] = base["U"][i*m + j][r]
        for j in range(m):
            for k in range(p):
                V_new[j*target_p + k][r] = base["V"][j*p + k][r]
        for k in range(p):
            for i in range(n):
                W_new[k*target_n + i][r] = base["W"][k*n + i][r]
    
    return {"U": U_new, "V": V_new, "W": W_new}

def _block_decomp_strassen(n, m, p):
    """Try to decompose n x m x p using Strassen on 2x2x2 blocks where possible."""
    # For 3x5x6, we can try various block decompositions
    # Let's try: split n=3 as 2+1, m=5 as 2+2+1, p=6 as 2+2+2
    # This is complex, so let's try a simpler approach
    
    # Pad 3x5x6 to 4x6x6 and use 2x2x2 Strassen blocks
    # 4x6x6 can be decomposed into 2x3x3 blocks of 2x2x2
    # Each 2x2x2 uses rank 7, so 2*3*3*7 = 126
    # But schoolbook is 3*5*6 = 90, so this is worse
    
    # Try padding to 4x8x8: 4x8x8 = 2x4x4 blocks of 2x2x2
    # 2*4*4*7 = 224, worse than 90
    
    # The issue is that Strassen only helps when we can tile the entire space
    # For small dimensions, schoolbook is often better
    
    # Let's try a hybrid: use Strassen for 2x2x2 sub-blocks and schoolbook for the rest
    # This requires careful index mapping
    
    # For now, return None to indicate no improvement
    return None

def _try_kronecker(n, m, p):
    """Try Kronecker product based decomposition."""
    # If n, m, p are all powers of 2, we can use Strassen recursively
    # 3, 5, 6 are not powers of 2, so this won't directly help
    return None

def _randomized_search(n, m, p, target_rank, max_iters=1000):
    """Try to find a rank-target_rank decomposition via randomized search."""
    # This is a heuristic approach - not guaranteed to find optimal
    # For exact integer/half-integer solutions, this is very challenging
    
    # The constraint is: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # This is a system of n*m*p polynomial equations in r*(n*m + m*p + p*n) variables
    
    # For small rank reductions, we can try to find solutions
    # But for large reductions, this becomes intractable
    
    # Simple heuristic: start from schoolbook and try to merge terms
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if there exists
    # a rank-1 update that preserves the tensor
    
    # This is equivalent to finding a rank-1 matrix in the null space
    # of a certain linear map, which is complex
    
    # For now, return None
    return None

def _try_lanczos_or_similar(n, m, p):
    """Try other known low-rank algorithms."""
    # For 3x3x3, the best known is 23 (Laderman)
    # For 4x4x4, the best known is 49
    # For our 3x5x6, we need to find specific decompositions
    
    # Laderman's 23 multiplications for 3x3x3:
    # This is a known result but implementing it for 3x5x6 requires
    # careful block decomposition
    
    # For 3x5x6, we could potentially use:
    # - 3x3x3 Laderman (rank 23) for the 3x3x3 part
    # - Schoolbook for the remaining parts
    # But the dimensions don't align nicely
    
    return None

def _construct_hybrid(n, m, p):
    """Construct a hybrid decomposition using known low-rank algorithms for sub-blocks."""
    # Strategy: decompose 3x5x6 into sub-blocks where we can use known algorithms
    # 
    # Split: n=3, m=5, p=6
    # We can write:
    # C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2]  (2x2x2, use Strassen rank 7)
    # C[0:2, 2:4] = A[0:2, 0:2] * B[0:2, 2:4]  (2x2x2, use Strassen rank 7)
    # C[0:2, 4:6] = A[0:2, 0:2] * B[0:2, 4:6]  (2x2x2, use Strassen rank 7)
    # C[2:3, 0:2] = A[2:3, 0:2] * B[0:2, 0:2]  (1x2x2, schoolbook rank 4)
    # etc.
    # 
    # This gets complicated. Let's try a different approach.
    # 
    # Actually, for the full 3x5x6, the schoolbook rank is 90.
    # The theoretical lower bound is at least max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # And at least the border rank, which is unknown but likely close to schoolbook for these sizes.
    # 
    # Given the complexity, let's just return schoolbook for now.
    return _schoolbook_decomp(n, m, p)

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    deadline = t0 + 115.0  # safety margin
    
    # Start with schoolbook
    best = _schoolbook_decomp(n, m, p)
    best_rank = n * m * p
    
    # Verify the schoolbook decomposition
    if not _verify_decomposition(best["U"], best["V"], best["W"], n, m, p):
        raise RuntimeError("Schoolbook decomposition failed verification")
    
    # Try various approaches to find a lower rank decomposition
    # 1. Try Strassen-based block decomposition
    if time.time() < deadline - 10:
        result = _block_decomp_strassen(n, m, p)
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                rank = len(result["U"][0])
                if rank < best_rank:
                    best = result
                    best_rank = rank
    
    # 2. Try hybrid approach
    if time.time() < deadline - 10:
        result = _construct_hybrid(n, m, p)
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            rank = len(result["U"][0])
            if rank < best_rank:
                best = result
                best_rank = rank
    
    # 3. Try randomized search for small rank reductions
    if time.time() < deadline - 5:
        for target_rank in range(best_rank - 1, max(50, best_rank - 10), -1):
            if time.time() > deadline - 5:
                break
            result = _randomized_search(n, m, p, target_rank)
            if result is not None:
                if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = target_rank
                    break
    
    return best
# EVOLVE-BLOCK-END
