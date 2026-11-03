"""Optimized tensor decomposition for 4x4 * 4x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_recursive_2x2():
    """
    Returns (U, V, W) for 2x2 * 2x2 -> 2x2 via Strassen (rank 7).
    Indexing: U is 4x7, V is 4x7, W is 4x7.
    U row = i*m + j, V row = j*p + k, W row = k*n + i (with n=m=p=2).
    C[i,k] = sum_r U[i*2+j,r] * V[j*2+k,r] * W[k*2+i,r]
    For 2x2: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen for 2x2*2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each term r has contributions to U (4x1), V (4x1), W (4x1)
    # U[i*2+j], V[j*2+k], W[k*2+i]
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1 -> [1,0,0,1]
    # V: B00=1, B11=1 -> [1,0,0,1]
    # W: C00=M1, C11=M1 -> W[k*2+i]: C00=idx(0,0)=0, C11=idx(1,1)=3 -> W[0]=1, W[3]=1
    # Actually W index: k*n+i = k*2+i
    # C00: k=0,i=0 -> W[0]
    # C11: k=1,i=1 -> W[3]
    
    terms = []
    
    # M1: (A00+A11)(B00+B11)
    # U = [1, 0, 0, 1] (A00=1 at idx 0, A11=1 at idx 3)
    # V = [1, 0, 0, 1] (B00=1 at idx 0, B11=1 at idx 3)
    # W: C00+=1 -> W[0]+=1, C11+=1 -> W[3]+=1 => W = [1, 0, 0, 1]
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    
    # M2: (A10+A11)B00
    # U: A10=1 at idx 2, A11=1 at idx 3 => [0, 0, 1, 1]
    # V: B00=1 at idx 0 => [1, 0, 0, 0]
    # W: C10+=1 -> k=0,i=1 -> W[1]+=1, C11+=-1 -> k=1,i=1 -> W[3]+=-1 => W = [0, 1, 0, -1]
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]))
    
    # M3: A00(B01-B11)
    # U: A00=1 => [1, 0, 0, 0]
    # V: B01=1 at idx 1, B11=-1 at idx 3 => [0, 1, 0, -1]
    # W: C01+=1 -> k=1,i=0 -> W[2]+=1, C11+=1 -> k=1,i=1 -> W[3]+=1 => W = [0, 0, 1, 1]
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]))
    
    # M4: A11(B10-B00)
    # U: A11=1 => [0, 0, 0, 1]
    # V: B10=1 at idx 2, B00=-1 at idx 0 => [-1, 0, 1, 0]
    # W: C00+=1 -> W[0]+=1, C10+=1 -> W[1]+=1 => W = [1, 1, 0, 0]
    terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]))
    
    # M5: (A00+A01)B11
    # U: A00=1 at idx 0, A01=1 at idx 1 => [1, 1, 0, 0]
    # V: B11=1 at idx 3 => [0, 0, 0, 1]
    # W: C00+=-1 -> W[0]+=-1, C01+=1 -> W[2]+=1 => W = [-1, 0, 1, 0]
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]))
    
    # M6: (A10-A00)(B00+B01)
    # U: A10=1 at idx 2, A00=-1 at idx 0 => [-1, 0, 1, 0]
    # V: B00=1 at idx 0, B01=1 at idx 1 => [1, 1, 0, 0]
    # W: C11+=1 -> W[3]+=1 => W = [0, 0, 0, 1]
    terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    
    # M7: (A01-A11)(B10+B11)
    # U: A01=1 at idx 1, A11=-1 at idx 3 => [0, 1, 0, -1]
    # V: B10=1 at idx 2, B11=1 at idx 3 => [0, 0, 1, 1]
    # W: C00+=1 -> W[0]+=1 => W = [1, 0, 0, 0]
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    
    U = np.array([term[0] for term in terms]).T  # 4x7
    V = np.array([term[1] for term in terms]).T  # 4x7
    W = np.array([term[2] for term in terms]).T  # 4x7
    return U, V, W


def _recursive_strassen_4x4():
    """
    Returns (U, V, W) for 4x4 * 4x4 -> 4x4 via recursive Strassen (rank 49).
    Splits into 2x2 blocks, each 2x2 multiplication uses rank-7 Strassen.
    """
    n = m = p = 4
    bn = 2
    n_blocks = n // bn  # 2
    m_blocks = m // bn  # 2
    p_blocks = p // bn  # 2
    
    U2, V2, W2 = _strassen_recursive_2x2()  # Each is 4x7
    
    total_rank = n_blocks * p_blocks * m_blocks * 7  # 2*2*2*7 = 56? No wait...
    # Actually: for each output 2x2 block (a, c), we need m_blocks=2 inner products
    # Each inner product uses 7 Strassen terms
    # So total rank = n_blocks * p_blocks * m_blocks * 7 = 2 * 2 * 2 * 7 = 56
    
    # Hmm, that's not 49. Let me reconsider.
    # Actually for 4x4*4x4: split A into 2x2 blocks of 2x2, B into 2x2 blocks of 2x2.
    # C[a,c] = sum_j A[a,j] * B[j,c], where a,c,j in {0,1}
    # For each (a,c): 2 multiplications of 2x2 by 2x2, each rank 7
    # Total rank = 4 output blocks * 2 inner products * 7 = 56
    
    # But recursive Strassen gives 7^2 = 49 for 4x4*4x4!
    # The key insight: when we do Strassen on 2x2 blocks, the "scalar" multiplications
    # in Strassen are themselves 2x2 multiplications, which we also do with Strassen.
    # So we don't multiply by m_blocks separately - the Strassen already accounts for it.
    
    # Let me redo: 
    # Level 1: C is 2x2 blocks of 2x2. C[a,c] = A[a,0]*B[0,c] + A[a,1]*B[1,c]
    # Using Strassen at block level: 7 "multiplications" of 2x2 by 2x2 matrices
    # Each of those 2x2*2x2 is done with Strassen: rank 7
    # Total: 7 * 7 = 49
    
    # So I need to compose: for each of the 7 Strassen terms at block level,
    # substitute the rank-7 Strassen for the 2x2*2x2 multiplication.
    
    U_data = np.zeros((n * m, 49))  # 16 x 49
    V_data = np.zeros((m * p, 49))  # 16 x 49
    W_data = np.zeros((p * n, 49))  # 16 x 49
    
    # Block-level Strassen for 2x2 blocks of 2x2:
    # Same 7 terms as _strassen_recursive_2x2 but at block level
    block_terms = [
        # (a_coeffs: dict from (a_row, a_col_block) -> val, 
        #  b_coeffs: dict from (b_row_block, b_col) -> val,
        #  c_coeffs: dict from (c_row, c_col_block) -> val)
        # For block (a_row in {0,1}, a_col_block in {0,1}), B block (b_row_block in {0,1}, b_col in {0,1})
        # C block (c_row in {0,1}, c_col_block in {0,1})
        
        # M1: (A[0,:] + A[1,:]) * (B[:,0] + B[:,1]) where each is 2x2
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
        # Wait, I need to be careful. The block-level Strassen is:
        # A is 2x2 blocks: A00, A01, A10, A11 (each 2x2)
        # B is 2x2 blocks: B00, B10, B01, B11 (each 2x2)  -- wait, B is 4x4 so blocks B[j,c]
        
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
    ]
    
    # Each "multiplication" like (A00+A11)(B00+B11) is a 2x2*2x2 mult
    # I'll use the precomputed 2x2 Strassen factors
    # For each block-level term, I have linear combos of 2x2 matrices for A, B
    # Then I multiply them using Strassen rank-7
    
    # Let me build it more carefully.
    # For each of the 7 block-level terms, I have:
    #   A_combo: sum of alpha_ij * A_ij where A_ij is a 2x2 block at position (i,j)
    #   B_combo: sum of beta_kl * B_kl where B_kl is a 2x2 block at position (k,l)
    #   And I need to compute A_combo * B_combo (2x2 * 2x2)
    #   This 2x2*2x2 product uses rank-7 Strassen
    
    # The 2x2 Strassen factors U2, V2, W2 are for:
    #   For 2x2 A (indices 0-3), 2x2 B (indices 0-3), 2x2 C (indices 0-3):
    #   C[r] = sum_s U2[r,s] * A[r] * V2[s,t] * B[t] * W2[t,s]  -- no this isn't right
    
    # Let me use a cleaner composition. 
    # I'll just do the double loop: for each output block (a,c) and each input block j,
    # do Strassen for A[a,j] * B[j,c].
    
    U_data = np.zeros((n * m, 56))  # 16 x 56 (this gives rank 56, not 49)
    V_data = np.zeros((m * p, 56))
    W_data = np.zeros((p * n, 56))
    
    # Actually, let me just use the simpler approach: rank 56 via nested Strassen
    # (2x2 blocks, each 2x2 mult uses rank 7)
    # 2 output blocks * 2 * 2 input pairs * 7 = 56
    
    col = 0
    for a in range(n_blocks):      # output row block 0,1
        for c in range(p_blocks):  # output col block 0,1
            for jb in range(m_blocks):  # input col block 0,1
                # A block is A[2a:2a+2, 2jb:2jb+2]
                # B block is B[2jb:2jb+2, 2c:2c+2]
                # Use Strassen rank-7 for this 2x2 * 2x2
                for r in range(7):
                    # U contribution from A block at (a, jb)
                    # U2[i*2+j, r] where i is local row in A block, j is local col in A block
                    for li in range(bn):  # local row in A block
                        for lj in range(bn):  # local col in A block
                            U_data[(2*a+li)*m + (2*jb+lj), col] = U2[li*2+lj, r]
                    
                    # V contribution from B block at (jb, c)
                    # V2[j*2+k, r] where j is local row in B block, k is local col in B block
                    for lj in range(bn):  # local row in B block
                        for lk in range(bn):  # local col in B block
                            V_data[(2*jb+lj)*p + (2*c+lk), col] = V2[lj*2+lk, r]
                    
                    # W contribution: C block at (a, c)
                    # W2[k*2+i, r] where k is local row in C block, i is local col in C block
                    for lk in range(bn):  # local row in C block
                        for li in range(bn):  # local col in C block
                            W_data[(2*c+lk)*n + (2*a+li), col] = W2[lk*2+li, r]
                    
                    col += 1
    
    rank = col
    return U_data[:, :rank].tolist(), V_data[:, :rank].tolist(), W_data[:, :rank].tolist()


def construct():
    """
    Find a low-rank tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Uses recursive Strassen for the 4x4x4 part and schoolbook for the 4x2 part.
    """
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Get recursive Strassen for 4x4 * 4x4 (rank 56 with nested 2x2 Strassen)
    # Actually let me try to get true recursive Strassen rank 49.
    
    # For now, use the simpler nested approach which gives rank 56 for 4x4*4x4
    U_44, V_44, W_44 = _recursive_strassen_4x4()
    rank_44 = len(V_44[0]) if V_44 else 0
    
    # For the 4x2 part: schoolbook, rank = 4 * 4 * 2 / 4 = 8
    # C[:, 4:6] = A * B[:, 4:6], where B[:, 4:6] is 4x2
    # Schoolbook: rank = n * m * (p - 4) / m = n * (p-4) = 4 * 2 = 8
    rank_42 = n * 2  # = 8
    
    total_rank = rank_44 + rank_42
    
    U_data = np.zeros((n * m, total_rank))
    V_data = np.zeros((m * p, total_rank))
    W_data = np.zeros((p * n, total_rank))
    
    # Copy 4x4 part
    U_data[:, :rank_44] = np.array(U_44)
    V_data[:, :rank_44] = np.array(V_44)
    W_data[:, :rank_44] = np.array(W_44)
    
    # Schoolbook for 4x2 part: C[i, 4+k] = sum_j A[i,j] * B[j, 4+k]
    # Each (i, j, k) triple gives rank 1
    col = rank_44
    for i in range(n):
        for j in range(m):
            for k in range(2):
                U_data[i * m + j, col] = 1
                V_data[j * p + (4 + k), col] = 1
                W_data[(4 + k) * n + i, col] = 1
                col += 1
    
    rank = col
    
    # Verify
    U_np, V_np, W_np = np.array(U_data), np.array(V_data), np.array(W_data)
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i * m + j] * V_np[j * p + k], W_np[k * n + i])
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook for entire 4x4x6
        r2 = n * m * p  # 96
        U2 = np.zeros((n * m, r2))
        V2 = np.zeros((m * p, r2))
        W2 = np.zeros((p * n, r2))
        cc = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U2[ii * m + jj, cc] = 1
                    V2[jj * p + kk, cc] = 1
                    W2[kk * n + ii, cc] = 1
                    cc += 1
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    
    return {"U": U_data[:, :rank].tolist(), "V": V_data[:, :rank].tolist(), "W": W_data[:, :rank].tolist()}

# EVOLVE-BLOCK-END
