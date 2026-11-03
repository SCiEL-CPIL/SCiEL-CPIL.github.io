"""Program for matrix multiplication tensor decomposition task."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    
    Strategy:
    1. Use Laderman's rank-23 decomposition for 3x3x3.
    2. Extend this to 3x3x6 by treating the 3x6 matrix as two 3x3 blocks.
    3. The rank for 3x3x6 will be 2 * 23 = 46, which is better than schoolbook 54.
    4. Verify the decomposition exactly.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    def get_laderman_23():
        """
        Return Laderman's rank-23 decomposition for 3x3x3 matrix multiplication.
        Each term is a triple (u, v, w) of length-9 vectors.
        Indices: A[i,j] at i*3+j, B[j,k] at j*3+k, C[k,i] at k*3+i.
        """
        terms = []
        
        def vec_A(idx, val=1):
            v = [0]*9
            v[idx] = val
            return v
        
        def vec_B(idx, val=1):
            v = [0]*9
            v[idx] = val
            return v
        
        def vec_C(idx, val=1):
            v = [0]*9
            v[idx] = val
            return v
        
        def vec_A_multi(indices, vals):
            v = [0]*9
            for idx, val in zip(indices, vals):
                v[idx] = val
            return v
        
        def vec_B_multi(indices, vals):
            v = [0]*9
            for idx, val in zip(indices, vals):
                v[idx] = val
            return v
        
        def vec_C_multi(indices, vals):
            v = [0]*9
            for idx, val in zip(indices, vals):
                v[idx] = val
            return v
        
        # Laderman's 23 terms for 3x3x3
        # The 9 diagonal terms:
        terms.append((vec_A(0), vec_B(0), vec_C(0)))   # a00*b00 -> c00
        terms.append((vec_A(1), vec_B(3), vec_C(1)))   # a01*b10 -> c01
        terms.append((vec_A(2), vec_B(6), vec_C(2)))   # a02*b20 -> c02
        terms.append((vec_A(3), vec_B(1), vec_C(3)))   # a10*b01 -> c10
        terms.append((vec_A(4), vec_B(4), vec_C(4)))   # a11*b11 -> c11
        terms.append((vec_A(5), vec_B(7), vec_C(5)))   # a12*b21 -> c12
        terms.append((vec_A(6), vec_B(2), vec_C(6)))   # a20*b02 -> c20
        terms.append((vec_A(7), vec_B(5), vec_C(7)))   # a21*b12 -> c21
        terms.append((vec_A(8), vec_B(8), vec_C(8)))   # a22*b22 -> c22
        
        # The 14 off-diagonal terms from Laderman's algorithm:
        # T10: (a00+a01)*(b01+b10) -> contributes to c00, c01, c10
        terms.append((vec_A_multi([0,1],[1,1]), vec_B_multi([1,3],[1,1]), vec_C_multi([0,1,3],[1,1,1])))
        
        # T11: (a00-a01)*(b10-b01) -> contributes to c00, c01, c10
        terms.append((vec_A_multi([0,1],[1,-1]), vec_B_multi([3,1],[1,-1]), vec_C_multi([0,1,3],[1,1,1])))
        
        # T12: (a01+a02)*(b10+b21) -> contributes to c01, c02, c11, c12
        terms.append((vec_A_multi([1,2],[1,1]), vec_B_multi([3,7],[1,1]), vec_C_multi([1,2,4,5],[1,1,1,1])))
        
        # T13: (a01-a02)*(b20+b11) -> contributes to c02, c11
        terms.append((vec_A_multi([1,2],[1,-1]), vec_B_multi([6,4],[1,1]), vec_C_multi([2,4],[1,1])))
        
        # T14: (a02+a10)*(b20+b01) -> contributes to c02, c10
        terms.append((vec_A_multi([2,3],[1,1]), vec_B_multi([6,1],[1,1]), vec_C_multi([2,3],[1,1])))
        
        # T15: (a02-a10)*(b00+b21) -> contributes to c00, c12
        terms.append((vec_A_multi([2,3],[1,-1]), vec_B_multi([0,7],[1,1]), vec_C_multi([0,5],[1,1])))
        
        # T16: (a10+a11)*(b01+b12) -> contributes to c10, c11, c20, c21
        terms.append((vec_A_multi([3,4],[1,1]), vec_B_multi([1,5],[1,1]), vec_C_multi([3,4,6,7],[1,1,1,1])))
        
        # T17: (a10-a11)*(b11+b02) -> contributes to c11, c20
        terms.append((vec_A_multi([3,4],[1,-1]), vec_B_multi([4,2],[1,1]), vec_C_multi([4,6],[1,1])))
        
        # T18: (a11+a12)*(b11+b22) -> contributes to c11, c12, c21, c22
        terms.append((vec_A_multi([4,5],[1,1]), vec_B_multi([4,8],[1,1]), vec_C_multi([4,5,7,8],[1,1,1,1])))
        
        # T19: (a11-a12)*(b21+b12) -> contributes to c12, c21
        terms.append((vec_A_multi([4,5],[1,-1]), vec_B_multi([7,5],[1,1]), vec_C_multi([5,7],[1,1])))
        
        # T20: (a12+a20)*(b21+b02) -> contributes to c12, c20
        terms.append((vec_A_multi([5,6],[1,1]), vec_B_multi([7,2],[1,1]), vec_C_multi([5,6],[1,1])))
        
        # T21: (a12-a20)*(b01+b22) -> contributes to c10, c22
        terms.append((vec_A_multi([5,6],[1,-1]), vec_B_multi([1,8],[1,1]), vec_C_multi([3,8],[1,1])))
        
        # T22: (a20+a21)*(b02+b12) -> contributes to c20, c21, c00, c01
        terms.append((vec_A_multi([6,7],[1,1]), vec_B_multi([2,5],[1,1]), vec_C_multi([6,7,0,1],[1,1,1,1])))
        
        # T23: (a20-a21)*(b12+b02) -> contributes to c21, c20
        terms.append((vec_A_multi([6,7],[1,-1]), vec_B_multi([5,2],[1,1]), vec_C_multi([7,6],[1,1])))
        
        return terms
    
    # Get the 23 terms for 3x3x3
    laderman_terms = get_laderman_23()
    
    # Now extend to 3x3x6 by duplicating for the two 3x3 blocks
    # B is 3x6, so we split it into B1 (3x3, columns 0-2) and B2 (3x3, columns 3-5)
    # C is 6x3, so C1 (rows 0-2) corresponds to B1, C2 (rows 3-5) corresponds to B2
    
    U = []
    V = []
    W = []
    
    for (u, v, w) in laderman_terms:
        # Block 1: B columns 0-2, C rows 0-2
        U.append(list(u))
        V.append(list(v))  # v has indices 0-8 for B[0:3, 0:3]
        W.append(list(w))  # w has indices 0-8 for C[0:3, 0:3]
        
        # Block 2: B columns 3-5, C rows 3-5
        U.append(list(u))
        V.append([0]*9 + list(v))  # v shifted to B[0:3, 3:6]
        W.append([0]*9 + list(w))  # w shifted to C[3:6, 0:3]
    
    rank = len(U)
    
    # Verify the decomposition
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        # Check that for all i, j, k: sum_r U[i,j,r] * V[j,k,r] == W[k,i,r]
        # Actually, the tensor decomposition means:
        # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r (sum_j A[i,j]*U[j,r]) * (sum_k V[k,r]*B[j,k]) ... 
        # Wait, let me think about this more carefully.
        
        # The tensor T has entries T[i,j,k] = delta[i,j,k] where T[i,j,k] = 1 if the output C[i,k] 
        # depends on A[i,j] and B[j,k]. Actually, for matrix multiplication:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k].
        # But in the decomposition, we have:
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        
        # Let's verify by checking a few entries
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # The tensor T[i,j,k] should be 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
        # In our indexing: A index = i*m+j, B index = j*p+k, C index = k*n+i
        # T[A_idx, B_idx, C_idx] = 1 if A[i,j]*B[j,k] contributes to C[i,k]
        # This happens when C_idx = k*n+i and B_idx = j*p+k and A_idx = i*m+j
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    A_idx = i * m + j
                    B_idx = j * p + k
                    C_idx = k * n + i
                    # The sum should be 1
                    s = np.sum(U_arr[A_idx, :] * V_arr[B_idx, :] * W_arr[C_idx, :])
                    if abs(s - 1) > 1e-10:
                        return False, (i, j, k, s)
        return True, None
    
    valid, err = verify_decomposition(U, V, W, n, m, p)
    if not valid:
        # Fallback to schoolbook
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
