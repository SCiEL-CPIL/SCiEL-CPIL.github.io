"""
Rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
Uses Strassen 7-rank scheme for 2x2x2 blocks combined with schoolbook for strips.
"""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def _strassen_2x2():
    """
    Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen).
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    Strassen's 7 multiplications:
    M1 = (a11+a22)(b11+b22)
    M2 = (a21+a22)b11
    M3 = a11(b12-b22)
    M4 = a22(b21-b11)
    M5 = (a11+a12)b22
    M6 = (a21-a11)(b11+b12)
    M7 = (a12-a22)(b21+b22)
    
    C11 = M1 - M4 + M6 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M3 + M2 - M5
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    # a11=U[0], a12=U[1], a21=U[2], a22=U[3]
    # b11=V[0], b12=V[1], b21=V[2], b22=V[3]
    # c11=W[0], c12=W[1], c21=W[2], c22=W[3]
    
    # M1 = (a11+a22)(b11+b22)
    # U[0] += 1, U[3] += 1
    # V[0] += 1, V[3] += 1
    # W[0] += 1, W[1] += 1, W[2] += 1, W[3] += 1 (from C11=M1-..., C12=...+M5, C21=...+M4, C22=M1-...)
    # Actually let me be more careful.
    
    # Let me define each term explicitly.
    # Term c corresponds to a rank-1 tensor: u_vec * v_vec * w_vec
    # where (u_vec)_i = U[i, c], (v_vec)_j = V[j, c], (w_vec)_k = W[k, c]
    
    # The MM tensor is: T[i,j,k] = delta_{i,k} (for C[i,k] = sum_j A[i,j] B[j,k])
    # Wait, the convention is W[k*n+i, c] so the output index is (k, i) meaning C[k, i]?
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # And from the schoolbook in the seed: W[k*n+i, c] = 1 when matching i,j,k
    # So the tensor being decomposed is: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = delta_{i,k}
    # This means C[i,k] = sum_j A[i,j] B[j,k], which is standard MM.
    
    # Strassen terms for 2x2:
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B = [[c00, c01], [c10, c11]]
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    
    # Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # c00 = M1 - M4 + M6 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M3 + M2 - M5
    
    # Now express each M_t as a rank-1 term:
    # M_t = (sum_i alpha_i * A[i, j_t]) * (sum_j beta_j * B[j_t, k]) ... no.
    # Each M_t is a product of a linear form in A and a linear form in B.
    # M_t = (sum_{i,j} u_{i,j}^t A[i,j]) * (sum_{j,k} v_{j,k}^t B[j,k])
    # Then the contribution to C[i,k] is w_{i,k}^t * M_t where w_{i,k}^t is the coefficient.
    
    # In our convention:
    # U[i*m+j, c] = u_{i,j}^c
    # V[j*p+k, c] = v_{j,k}^c  
    # W[k*n+i, c] = w_{k,i}^c (note: W index is k*n+i, so W[k,i])
    
    # For 2x2: n=m=p=2
    # U indices: i*2+j for i,j in {0,1}
    # V indices: j*2+k for j,k in {0,1}
    # W indices: k*2+i for k,i in {0,1}
    
    # M1 = (a00+a11)(b00+b11)
    # u: A[0,0]=1, A[1,1]=1 -> U[0]=1, U[3]=1
    # v: B[0,0]=1, B[1,1]=1 -> V[0]=1, V[3]=1
    # w: contributes to c00, c01, c10, c11 with coeffs from the formulas
    # c00 = M1 - M4 + M6 + M7 -> W[0,0] += 1 (k=0,i=0)
    # c01 = M3 + M5 -> no M1 contribution
    # c10 = M2 + M4 -> no M1 contribution  
    # c11 = M1 - M3 + M2 - M5 -> W[1,1] += 1 (k=1,i=1)
    # Wait, W[k*n+i] so for c00: k=0,i=0 -> W[0*2+0]=W[0]
    # for c01: k=0,i=1 -> W[0*2+1]=W[1]
    # for c10: k=1,i=0 -> W[1*2+0]=W[2]
    # for c11: k=1,i=1 -> W[1*2+1]=W[3]
    
    # M1 contributes: c00 += 1*M1, c11 += 1*M1
    # So W[0] += 1, W[3] += 1
    
    # M2 = (a10+a11)b00
    # u: A[1,0]=1, A[1,1]=1 -> U[2]=1, U[3]=1
    # v: B[0,0]=1 -> V[0]=1
    # c10 = M2 + M4 -> W[2] += 1
    # c11 = M1 - M3 + M2 - M5 -> W[3] += 1
    
    # M3 = a00(b01-b11)
    # u: A[0,0]=1 -> U[0]=1
    # v: B[0,1]=1, B[1,1]=-1 -> V[1]=1, V[3]=-1
    # c01 = M3 + M5 -> W[1] += 1
    # c11 = M1 - M3 + M2 - M5 -> W[3] += -1
    
    # M4 = a11(b10-b00)
    # u: A[1,1]=1 -> U[3]=1
    # v: B[1,0]=1, B[0,0]=-1 -> V[2]=1, V[0]=-1
    # c00 = M1 - M4 + M6 + M7 -> W[0] += -1
    # c10 = M2 + M4 -> W[2] += 1
    
    # M5 = (a00+a01)b11
    # u: A[0,0]=1, A[0,1]=1 -> U[0]=1, U[1]=1
    # v: B[1,1]=1 -> V[3]=1
    # c01 = M3 + M5 -> W[1] += 1
    # c11 = M1 - M3 + M2 - M5 -> W[3] += -1
    
    # M6 = (a10-a00)(b00+b01)
    # u: A[1,0]=1, A[0,0]=-1 -> U[2]=1, U[0]=-1
    # v: B[0,0]=1, B[0,1]=1 -> V[0]=1, V[1]=1
    # c00 = M1 - M4 + M6 + M7 -> W[0] += 1
    
    # M7 = (a01-a11)(b10+b11)
    # u: A[0,1]=1, A[1,1]=-1 -> U[1]=1, U[3]=-1
    # v: B[1,0]=1, B[1,1]=1 -> V[2]=1, V[3]=1
    # c00 = M1 - M4 + M6 + M7 -> W[0] += 1
    
    # Now build the matrices:
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: c=0
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: c=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3: c=2
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # M4: c=3
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # M5: c=4
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = -1
    
    # M6: c=5
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1
    
    # M7: c=6
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def construct():
    """
    Construct the tensor decomposition for 4x5 * 5x6 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Strategy: Partition into blocks
    # A is 4x5, B is 5x6, C is 4x6
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # We'll use Strassen for 2x2 sub-blocks where possible,
    # and schoolbook for the rest.
    
    # Partition:
    # i in {0,1,2,3}, j in {0,1,2,3,4}, k in {0,1,2,3,4,5}
    
    # For j in {0,1}: use Strassen-like approach on 2x2x2 blocks
    # For j in {2,3,4}: schoolbook
    
    # Actually, let's think more carefully.
    # The tensor is: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = delta_{i,k}
    
    # We need to decompose this tensor.
    # Schoolbook gives rank = n*m*p = 4*5*6 = 120.
    
    # Let's try to reduce by using Strassen on 2x2x2 blocks.
    
    # Consider the sub-problem where i in {0,1}, j in {0,1}, k in {0,1}.
    # This is a 2x2x2 MM tensor, which has rank 7 (Strassen) instead of 8.
    
    # But the full problem is 4x5x6, not a product of 2x2x2 blocks.
    
    # Let me try a different approach: use the fact that we can apply
    # Strassen to any 2x2x2 sub-block.
    
    # For each pair (i1,i2) in rows of A, (j1,j2) in cols of A/rows of B,
    # and (k1,k2) in cols of B, we have a 2x2x2 sub-problem.
    
    # But these sub-problems overlap, so we can't just take direct sums.
    
    # Let me try a simpler approach:
    # Use Strassen for the top-left 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    # and schoolbook for everything else.
    
    # This gives: 7 + (120 - 8) = 7 + 112 = 119. Only saves 1.
    
    # Better: partition into non-overlapping 2x2x2 blocks.
    # i in {0,1,2,3} -> two blocks: {0,1} and {2,3}
    # j in {0,1,2,3,4} -> two blocks: {0,1} and {2,3}, plus j=4
    # k in {0,1,2,3,4,5} -> three blocks: {0,1}, {2,3}, {4,5}
    
    # Non-overlapping 2x2x2 blocks:
    # Block (i-block, j-block, k-block) where:
    # i-blocks: [0,1], [2,3]
    # j-blocks: [0,1], [2,3]
    # k-blocks: [0,1], [2,3], [4,5]
    
    # That's 2*2*3 = 12 blocks of 2x2x2, each with rank 7 instead of 8.
    # Total from these: 12*7 = 84 instead of 12*8 = 96. Save 12.
    
    # Remaining: j=4 strip (i in {0,1,2,3}, j=4, k in {0,1,2,3,4,5})
    # That's 4*1*6 = 24 terms with schoolbook.
    
    # Total rank: 84 + 24 = 108.
    
    # Let's implement this.
    
    all_terms = []  # list of (u_idx, v_idx, w_idx, u_val, v_val, w_val)
    
    # Get Strassen decomposition for 2x2x2
    U_s, V_s, W_s = _strassen_2x2()
    
    # For each 2x2x2 block, apply Strassen
    i_blocks = [(0, 1), (2, 3)]
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    for i_start, i_end in i_blocks:
        for j_start, j_end in j_blocks:
            for k_start, k_end in k_blocks:
                # This block covers:
                # i in {i_start, i_start+1}
                # j in {j_start, j_start+1}
                # k in {k_start, k_start+1}
                
                # For each of the 7 Strassen terms:
                for c in range(7):
                    # U_s has shape (4, 7), indexed by i_local*2 + j_local
                    # V_s has shape (4, 7), indexed by j_local*2 + k_local
                    # W_s has shape (4, 7), indexed by k_local*2 + i_local
                    
                    for i_local in range(2):
                        u_val = U_s[i_local * 2 + 0, c]  # Wait, U_s[i*2+j]
                        if u_val != 0:
                            i_global = i_start + i_local
                            # Need to find which j_local this corresponds to
                            # U_s index is i_local*2 + j_local
                            # So for a given U_s entry at position (i_local*2+j_local, c):
                            pass
                    
                    # Let me redo this more carefully.
                    # U_s[a, c] where a = i_local*2 + j_local, i_local in {0,1}, j_local in {0,1}
                    # This corresponds to global index: i_global = i_start + i_local, j_global = j_start + j_local
                    # So U[i_global*m + j_global, c_global] += U_s[a, c]
                    
                    for a in range(4):
                        u_val = U_s[a, c]
                        if u_val != 0:
                            i_local = a // 2
                            j_local = a % 2
                            i_global = i_start + i_local
                            j_global = j_start + j_local
                            u_idx = i_global * m + j_global
                            # We'll accumulate in a dict or list
                            all_terms.append((u_idx, -1, -1, u_val, 0, 0))  # placeholder
                    
                    for a in range(4):
                        v_val = V_s[a, c]
                        if v_val != 0:
                            j_local = a // 2
                            k_local = a % 2
                            j_global = j_start + j_local
                            k_global = k_start + k_local
                            v_idx = j_global * p + k_global
                            # This is getting messy. Let me use a different approach.
    
    # This approach is too complex. Let me use a cleaner method.
    # Build U, V, W directly.
    
    rank = 108
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen blocks
    for i_start, i_end in i_blocks:
        for j_start, j_end in j_blocks:
            for k_start, k_end in k_blocks:
                for c in range(7):
                    # U_s[a, c] where a = i_local*2 + j_local
                    for a in range(4):
                        u_val = U_s[a, c]
                        if u_val != 0:
                            i_local = a // 2
                            j_local = a % 2
                            i_global = i_start + i_local
                            j_global = j_start + j_local
                            U[i_global * m + j_global, col] += u_val
                    
                    for a in range(4):
                        v_val = V_s[a, c]
                        if v_val != 0:
                            j_local = a // 2
                            k_local = a % 2
                            j_global = j_start + j_local
                            k_global = k_start + k_local
                            V[j_global * p + k_global, col] += v_val
                    
                    for a in range(4):
                        w_val = W_s[a, c]
                        if w_val != 0:
                            k_local = a // 2
                            i_local = a % 2
                            k_global = k_start + k_local
                            i_global = i_start + i_local
                            W[k_global * n + i_global, col] += w_val
                    
                    col += 1
    
    # Schoolbook for j=4 strip
    for i in range(n):
        for k in range(p):
            j = 4
            U[i * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    # Verify correctness
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(col):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                expected = 1.0 if i == k else 0.0
                if abs(val - expected) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
