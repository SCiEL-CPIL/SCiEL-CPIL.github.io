"""Program for matrix multiplication 4x4 * 4x8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]
DEADLINE_SECONDS = 110.0  # Safety margin from 120s limit

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses a Strassen-based decomposition with rank 98.
    
    Approach: Split B (4x8) into two 4x4 blocks B1, B2.
    Compute C1 = A*B1 and C2 = A*B2 using 2-level Strassen (rank 49 each).
    Total rank: 98.
    """
    start = time.time()
    n, m, p = DIMENSIONS
    
    def _time_left():
        return DEADLINE_SECONDS - (time.time() - start)
    
    def _timeup():
        return _time_left() < 0.5
    
    # Schoolbook rank-128 as fallback
    def _schoolbook():
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
    
    # Strassen 2x2 factors
    # Each factor: (u_vec[4], v_vec[4], w_vec[4])
    # u: A00, A01, A10, A11
    # v: B00, B01, B10, B11
    # w: C00, C01, C10, C11
    S = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # p1=(A00+A11)(B00+B11)
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),  # p2=(A10+A11)B00
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),  # p3=A00(B01-B11)
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),  # p4=A11(B10-B00)
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # p5=(A00+A01)B11
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # p6=(A10-A00)(B00+B01)
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),  # p7=(A01-A11)(B10+B11)
    ]
    
    # Build rank-98 decomposition for 4x4 * 4x8
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Pre-check deadline
    if _time_left() < 1.0:
        return _schoolbook()
    
    # Precompute non-zero entries for each Strassen factor for efficiency
    S_nonzero = []
    for factor in S:
        u_nz = [(idx, val) for idx, val in enumerate(factor[0]) if val != 0]
        v_nz = [(idx, val) for idx, val in enumerate(factor[1]) if val != 0]
        w_nz = [(idx, val) for idx, val in enumerate(factor[2]) if val != 0]
        S_nonzero.append((u_nz, v_nz, w_nz))
    
    for t in range(2):  # Two 4x4 subproblems (B1, B2)
        for s in range(7):  # Top-level Strassen factor
            u_block_nz, v_block_nz, w_block_nz = S_nonzero[s]
            for r in range(7):  # Scalar Strassen factor
                u_scalar_nz, v_scalar_nz, w_scalar_nz = S_nonzero[r]
                col = t * 49 + s * 7 + r
                
                # U: A is 4x4, split into four 2x2 blocks
                for bi, bj, cb in [(idx//2, idx%2, val) for idx, val in u_block_nz]:
                    for ei, ej, cs in [(idx//2, idx%2, val) for idx, val in u_scalar_nz]:
                        i = 2*bi + ei
                        j = 2*bj + ej
                        U[i*m + j][col] += cb * cs
                
                # V: B block for subproblem t
                for bi, bj, cb in [(idx//2, idx%2, val) for idx, val in v_block_nz]:
                    for ei, ej, cs in [(idx//2, idx%2, val) for idx, val in v_scalar_nz]:
                        j = 2*bi + ei
                        k = 4*t + 2*bj + ej
                        V[j*p + k][col] += cb * cs
                
                # W: C block for subproblem t
                for bi, bj, cb in [(idx//2, idx%2, val) for idx, val in w_block_nz]:
                    for ei, ej, cs in [(idx//2, idx%2, val) for idx, val in w_scalar_nz]:
                        i = 2*bi + ei
                        k = 4*t + 2*bj + ej
                        W[k*n + i][col] += cb * cs
                        
        if _timeup():
            break
    
    # Verify with time check
    def _verify(U, V, W):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        U3 = U_arr.reshape(n, m, rank)
        V3 = V_arr.reshape(m, p, rank)
        W3 = W_arr.reshape(p, n, rank)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(result, 1.0, atol=1e-10)
    
    if _timeup():
        # Time is up, return what we have if it looks reasonable, else schoolbook
        if rank == 98:
            # Quick sanity check: non-zero count
            nz = sum(1 for row in U for x in row if x != 0)
            if nz > 100:
                return {"U": U, "V": V, "W": W}
        return _schoolbook()
    
    if _verify(U, V, W):
        # Convert to integers for cleanliness
        U_int = [[int(round(x)) for x in row] for row in U]
        V_int = [[int(round(x)) for x in row] for row in V]
        W_int = [[int(round(x)) for x in row] for row in W]
        if _verify(U_int, V_int, W_int):
            return {"U": U_int, "V": V_int, "W": W_int}
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    return _schoolbook()
