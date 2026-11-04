import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110

    # Strategy: 6x6 as 3x3 blocks of 2x2 matrices.
    # Use Strassen's 7-multiplication algorithm for each 2x2 block.
    # Rank = 27 * 7 = 189.
    
    def _make_decomp_strassen_block3x3():
        # Strassen for 2x2:
        # C00 = M1 - M2 + M4 - M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M6 - M7
        # where:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # Each Mi contributes rank-1 tensors.
        # For the 6x6 problem, block (I,J) of A is a 2x2 matrix at rows [2I,2I+1] x [2J,2J+1]
        # Similarly for B and C.
        
        # Strassen factors as (U, V, W) where:
        # U has entries for A blocks, V for B blocks, W for C blocks.
        
        # For each of the 27 block positions (I, J, K):
        #   A block at (I, J): rows 2I..2I+1, cols 2J..2J+1
        #   B block at (J, K): rows 2J..2J+1, cols 2K..2K+1
        #   C block at (I, K): rows 2I..2I+1, cols 2K..2K+1
        
        # Strassen term t has:
        #   U factor: which A sub-blocks and coefficients
        #   V factor: which B sub-blocks and coefficients  
        #   W factor: which C sub-blocks and coefficients
        
        # Let's define Strassen's 7 terms as (U_pattern, V_pattern, W_pattern)
        # where each pattern is a dict mapping (block_row, block_col) -> coeff
        # For U: block positions refer to A's 2x2 block structure (top-left, top-right, bottom-left, bottom-right)
        # For V: block positions refer to B's 2x2 block structure
        # For W: block positions refer to C's 2x2 block structure
        
        # A block positions: (0,0)=top-left, (0,1)=top-right, (1,0)=bottom-left, (1,1)=bottom-right
        # Same for B and C.
        
        strassen_terms = [
            # M1 = (A00+A11)*(B00+B11)
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):-1}),
            # M2 = (A10+A11)*B00
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):-1, (1,1):1}),
            # M3 = A00*(B01-B11)
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):-1}),
            # M4 = A11*(B10-B00)
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):-1, (1,0):1}),
            # M5 = (A00+A01)*B11
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            # M6 = (A10-A00)*(B00+B01)
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):-1}),
            # M7 = (A01-A11)*(B10+B11)
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        
        rank = 27 * 7
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for I in range(3):
            for J in range(3):
                for K in range(3):
                    for (u_pat, v_pat, w_pat) in strassen_terms:
                        # U: A block at (I, J), 2x2 sub-blocks
                        for (dr, dc), coeff in u_pat.items():
                            r = 2*I + dr
                            c = 2*J + dc
                            U[r*m + c][col] += coeff
                        
                        # V: B block at (J, K), 2x2 sub-blocks
                        for (dr, dc), coeff in v_pat.items():
                            r = 2*J + dr
                            c = 2*K + dc
                            V[r*p + c][col] += coeff
                        
                        # W: C block at (I, K), 2x2 sub-blocks
                        # W index: k*n + i where k is row of C, i is col of C
                        for (dr, dc), coeff in w_pat.items():
                            k = 2*K + dc  # col of C
                            i = 2*I + dr  # row of C
                            W[k*n + i][col] += coeff
                        
                        col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def _verify(U, V, W, n, m, p, atol=1e-9):
        rank = len(U[0])
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        U_r = U_np.reshape(n, m, rank)
        V_r = V_np.reshape(m, p, rank)
        W_r = W_np.reshape(p, n, rank)
        T = np.einsum('ijt,jkt,kit->ijk', U_r, V_r, W_r)
        # T[i,j,k] should equal 1 for all i,j,k (the matmul tensor is sum over t of e_{i,j} e_{j,k} e_{k,i})
        # Actually the matmul tensor M[i,j,k] = 1 if j==j (always), so it's the identity in the middle index.
        # Wait: M[i,j,k] = delta_{j,j} = 1 for all i,j,k? No.
        # The matrix multiplication tensor: C[i,k] = sum_j A[i,j]*B[j,k]
        # So the tensor is M[i,j,k] = 1 for all i,j,k (it's just 1 * e_i ⊗ e_j ⊗ e_k summed over j... no)
        # Actually M[i,j,k] = delta_{j,j} which is always 1. That can't be right.
        # Let me reconsider: the bilinear form is C_{ik} = sum_j A_{ij} B_{jk}
        # The tensor is M_{ijk} such that C_{ik} = sum_j M_{ijk} A_{ij} B_{jk}
        # So M_{ijk} = delta_{j,j} = 1? No, M_{ijk} = 1 for all i,j,k.
        # Hmm, that means the tensor is the all-ones tensor? That doesn't seem right either.
        # Actually: C_{ik} = sum_j A_{ij} B_{jk}, so M_{ijk} = 1 for all i,j,k.
        # Yes, the matrix multiplication tensor for n×n×n is the all-ones tensor in the appropriate index structure.
        # Wait no. Let me be more careful.
        # C = AB, C_{ik} = sum_j A_{ij} B_{jk}
        # The "tensor" M is defined by M_{i,j,k} = 1 for all i,j,k, because
        # C_{ik} = sum_j (1) * A_{ij} * B_{jk}
        # So yes, M is the all-ones array of shape (n,m,p).
        return np.allclose(T, 1.0, atol=atol)
    
    # Build the 189-rank decomposition
    decomp = _make_decomp_strassen_block3x3()
    
    # Verify
    if _verify(decomp["U"], decomp["V"], decomp["W"], n, m, p):
        return decomp
    
    # Fallback: schoolbook rank 216
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
