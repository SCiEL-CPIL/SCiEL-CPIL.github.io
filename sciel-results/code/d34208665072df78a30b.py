import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for safety

    n, m, p = DIMENSIONS  # 4, 4, 8
    nm = n * m   # 16
    mp = m * p   # 32
    pn = p * n   # 32

    # Schoolbook fallback (rank 128)
    def _schoolbook():
        rank = n * m * p  # 128
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W

    # ---- Strassen 2x2*2x2 decomposition (rank 7) ----
    # U2[i, c], V2[j, c], W2[k, c] such that
    # sum_c U2[i,c]*V2[j,c]*W2[k,c] = 1 if i==k else 0  (for 2x2 matmul)
    # Wait, that's not right. For matrix multiplication C = A*B where A,B are 2x2:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] = 1 for all i,j,k (since every product contributes)
    # No wait - the tensor decomposition satisfies:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    # For 2x2*2x2: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k in {0,1}
    # This is the all-ones 2x2x2 tensor which has rank... let me think.
    # Actually no. The matrix multiplication tensor T_matmul has T[i,j,k] = 1
    # only when the indices are "compatible" in the sense that j is the shared index.
    # But for a fixed problem size, ALL (i,j,k) triples are valid, so T[i,j,k] = 1.
    # 
    # Hmm, but then rank 1 would suffice: U=1, V=1, W=1. That's clearly wrong.
    # 
    # Let me re-examine. The tensor for matrix multiplication C=A*B is:
    # T[i, j, k] = 1  (the coefficient of A[i,j]*B[j,k] in C[i,k])
    # The decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k] = 1
    # 
    # So yes, it IS the all-ones tensor. But the "rank" here is the tensor rank
    # of the all-ones tensor, which for 2x2x2 is... 
    # 
    # Actually wait. I think I'm confusing myself. Let me look at this differently.
    # The standard result is that the rank of the matrix multiplication tensor
    # <n,m,p> is the minimum number of scalar multiplications needed.
    # For 2x2x2, it's 7 (Strassen). For 1x1x1, it's 1.
    # 
    # The tensor T has entries T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # A rank-1 term is u(i,j) * v(j,k) * w(k,i) where these are functions.
    # The sum over r of these must equal 1 for all (i,j,k).
    # 
    # For 2x2x2: we need sum_r u_r(i,j) * v_r(j,k) * w_r(k,i) = 1 for all i,j,k.
    # This is 8 equations. Each rank-1 term has 4+4+4=12 parameters (up to scale).
    # The all-ones tensor of size 2x2x2 has tensor rank 7? No, that's not right either.
    # 
    # Let me just verify: the standard Strassen algorithm uses 7 multiplications
    # for 2x2 matrix multiplication. The "multiplications" correspond to rank-1 terms.
    # So the rank of the 2x2x2 matrix multiplication tensor is 7.
    # 
    # But the all-ones 2x2x2 tensor should have rank 1 (just u=v=w=1 everywhere).
    # 
    # I think the issue is that the matrix multiplication tensor is NOT the all-ones tensor
    # in the way I'm thinking. Let me re-read the problem statement.
    # 
    # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix
    #  by a 4 by 8 matrix. The output factor follows the official trace convention W[k*n+i,r]."
    # 
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # 
    # So U has n*m rows (indexed by (i,j) for A), V has m*p rows (indexed by (j,k) for B),
    # W has p*n rows (indexed by (k,i) for C, using trace convention k*n+i).
    # 
    # The constraint is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    # 
    # This IS the all-ones tensor. And the all-ones tensor of size (n*m) x (m*p) x (p*n)
    # has rank 1: just set all entries to 1.
    # 
    # But that gives rank 1, which is absurdly low. There must be something I'm missing.
    # 
    # OH WAIT. I think the issue is that the tensor is NOT all ones. The matrix multiplication
    # tensor T[i,j,k] is 1 only when the (i,j) entry of A and (j,k) entry of B contribute
    # to the (i,k) entry of C. This is ALWAYS true for valid indices. So T[i,j,k] = 1.
    # 
    # Unless... the problem is asking for something different. Let me re-read.
    # 
    # Actually, I think I've been overthinking this. The standard result in algebraic
    # complexity theory is that the rank of the matrix multiplication tensor <n,m,p>
    # (the tensor that encodes matrix multiplication) is the bilinear complexity.
    # For <2,2,2> it's 7, for <4,4,4> it's at most 49 (Strassen) and known to be <= 49.
    # 
    # The key insight I was missing: the tensor is NOT the all-ones tensor in the
    # (n*m, m*p, p*n) space. It's a specific tensor where T[i*m+j, j*p+k, k*n+i] = 1
    # but only for "compatible" indices. Wait, all indices are compatible...
    # 
    # Let me just look at this from the Strassen perspective. In Strassen's algorithm
    # for 2x2*2x2, we compute 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A10)(B11)
    # M6 = (A01+A11)(B00-B10)
    # M7 = (A00+A01)(B11-B01)
    # 
    # Each M is a scalar product of two linear combinations. In tensor decomposition terms:
    # M1 = sum_{i,j} (delta_{i,0}delta_{j,0} + delta_{i,1}delta_{j,1}) * A[i,j] * 
    #        sum_{j',k'} (delta_{j',0}delta_{k',0} + delta_{j',1}delta_{k',1}) * B[j',k']
    # 
    # So the rank-1 term for M1 has:
    # U[i,j] = delta_{i,0}delta_{j,0} + delta_{i,1}delta_{j,1}
    # V[j,k] = delta_{j,0}delta_{k,0} + delta_{j,1}delta_{k,1}
    # W[k,i] = delta_{k,0}delta_{i,0} + delta_{k,1}delta_{i,1}  (for C00) 
    #          + delta_{k,1}delta_{i,1}  (for C11)
    # 
    # Wait, W[k,i] should encode how M1 contributes to C. M1 contributes to both C00 and C11.
    # C00 = M1 - M4 + M5 - M7
    # C11 = M1 - M2 + M3 + M6
    # 
    # So for M1: W[0,0] += 1, W[1,1] += 1
    # 
    # The full decomposition: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all i,j,k.
    # 
    # Let me verify with M1 alone:
    # U_1[i,j] = 1 if (i,j) in {(0,0),(1,1)} else 0
    # V_1[j,k] = 1 if (j,k) in {(0,0),(1,1)} else 0
    # W_1[k,i] = 1 if (k,i) in {(0,0),(1,1)} else 0
    # 
    # M1 contribution to (i,j,k): U_1[i,j] * V_1[j,k] * W_1[k,i]
    # For (0,0,0): 1*1*1 = 1
    # For (0,0,1): 1*0*0 = 0 (V_1[0,1] = 0)
    # For (0,1,0): 0*... = 0
    # For (1,1,1): 1*1*1 = 1
    # For (1,1,0): 1*0*... = 0
    # 
    # So M1 alone contributes 1 to (0,0,0) and (1,1,1), 0 elsewhere.
    # 
    # The sum of all 7 terms should give 1 for all 8 (i,j,k) combinations.
    # 
    # OK so the tensor IS all-ones, but the rank-1 terms have specific structure
    # (they're "block diagonal" in some sense) that makes the rank 7 rather than 1.
    # 
    # Wait, that doesn't make sense. If the tensor is all-ones, then the rank-1 term
    # with U=1, V=1, W=1 everywhere gives exactly 1 for all (i,j,k). So rank 1 works!
    # 
    # I must be misunderstanding the problem. Let me re-read very carefully.
    # 
    # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix
    #  by a 4 by 8 matrix."
    # 
    # I think the issue is that the tensor is NOT all-ones. The matrix multiplication
    # tensor has a specific sparsity pattern. Let me think about this more carefully.
    # 
    # For C = A * B where A is n x m, B is m x p:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # 
    # The tensor T is a 3-way array with dimensions (n*m) x (m*p) x (p*n) where
    # T[i*m+j, j*p+k, k*n+i] = 1 if and only if... hmm, but for any valid (i,j,k),
    # A[i,j]*B[j,k] does contribute to C[i,k]. So T = 1 everywhere.
    # 
    # UNLESS the problem is using a different convention where the tensor is sparse.
    # 
    # Actually, I just realized: maybe the "tensor" in this context is not the
    # all-ones tensor, but rather the tensor that defines the bilinear map.
    # The bilinear map (A, B) -> C has the property that C[i,k] is a bilinear
    # form in A and B. The tensor of this bilinear map has entries:
    # T[(i,k), (i,j), (j,k)] = 1
    # 
    # In other words, T is indexed by (output, A-input, B-input) and is 1 only
    # when the indices are consistent. In the flattened representation:
    # T[k*n+i, i*m+j, j*p+k] = 1
    # 
    # This is a DIFFERENT tensor from the all-ones tensor! It's a sparse tensor
    # where T[a, b, c] = 1 only when a, b, c are "compatible" indices.
    # 
    # So the constraint is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i, j, k
    # 
    # Which is the same as what I had before. And this IS the all-ones tensor
    # in the (i,j,k) space. So rank 1 should work...
    # 
    # I'm going in circles. Let me just trust the known results:
    # - Rank of <2,2,2> matrix multiplication tensor = 7 (Strassen)
    # - Rank of <4,4,4> <= 49 (Strassen recursion)
    # - Rank of <4,4,8> <= 98 (apply 4x4x4 Strassen to two 4x4x4 blocks)
    # 
    # And just implement the Strassen-based decomposition.

    # Strassen 2x2*2x2 (rank 7)
    # Using the standard Strassen formulas:
    # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    # M2 = (A10+A11)(B00)     -> C10 += 1, C11 -= 1
    # M3 = (A00)(B01-B11)     -> C01 += 1, C11 += 1
    # M4 = (A11)(B10-B00)     -> C00 -= 1, C10 += 1
    # M5 = (A00+A10)(B11)     -> C00 += 1, C01 += 1
    # M6 = (A01+A11)(B00-B10) -> C11 += 1
    # M7 = (A00+A01)(B11-B01) -> C00 -= 1
    
    U2 = np.zeros((4, 7))  # indexed by (i*2+j, term)
    V2 = np.zeros((4, 7))  # indexed by (j*2+k, term)
    W2 = np.zeros((4, 7))  # indexed by (k*2+i, term)  [trace convention: k*2+i]

    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1  # A00, A11
    V2[0, 0] = 1; V2[3, 0] = 1  # B00, B11
    W2[0, 0] = 1; W2[3, 0] = 1  # C00 (k=0,i=0), C11 (k=1,i=1)

    # M2 = (A10+A11)(B00)
    U2[2, 1] = 1; U2[3, 1] = 1  # A10, A11
    V2[0, 1] = 1                # B00
    W2[2, 1] = 1; W2[3, 1] = -1 # C10 (k=0,i=1), C11 (k=1,i=1) with -1

    # M3 = (A00)(B01-B11)
    U2[0, 2] = 1                # A00
    V2[1, 2] = 1; V2[3, 2] = -1 # B01, -B11
    W2[1, 2] = 1; W2[3, 2] = 1  # C01 (k=1,i=0), C11 (k=1,i=1)

    # M4 = (A11)(B10-B00)
    U2[3, 3] = 1                # A11
    V2[2, 3] = 1; V2[0, 3] = -1 # B10, -B00
    W2[0, 3] = -1; W2[2, 3] = 1 # C00 (k=0,i=0) with -1, C10 (k=0,i=1)

    # M5 = (A00+A10)(B11)
    U2[0, 4] = 1; U2[2, 4] = 1  # A00, A10
    V2[3, 4] = 1                # B11
    W2[0, 4] = 1; W2[1, 4] = 1  # C00 (k=0,i=0), C01 (k=1,i=0)

    # M6 = (A01+A11)(B00-B10)
    U2[1, 5] = 1; U2[3, 5] = 1  # A01, A11
    V2[0, 5] = 1; V2[2, 5] = -1 # B00, -B10
    W2[3, 5] = 1                # C11 (k=1,i=1)

    # M7 = (A00+A01)(B11-B01)
    U2[0, 6] = 1; U2[1, 6] = 1  # A00, A01
    V2[3, 6] = 1; V2[1, 6] = -1 # B11, -B01
    W2[0, 6] = -1               # C00 (k=0,i=0) with -1

    # Verify 2x2 Strassen
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = np.sum(U2[i*2+j, :] * V2[j*2+k, :] * W2[k*2+i, :])
                if abs(val - 1.0) > 1e-10:
                    # Fallback to schoolbook
                    U, V, W = _schoolbook()
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now build 4x4*4x8 decomposition using Strassen recursion.
    # Split B (4x8) into two 4x4 blocks: B_left = B[:, 0:4], B_right = B[:, 4:8]
    # For each, compute A (4x4) * B_block (4x4) using 4x4x4 Strassen (rank 49).
    # Total rank: 49 + 49 = 98.
    
    # 4x4x4 Strassen: split A and B into 2x2 blocks, use 7 Strassen products,
    # each computed with 2x2x2 Strassen (rank 7). Total: 7*7 = 49 terms.
    
    U_list = []
    V_list = []
    W_list = []

    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term to the decomposition."""
        U_row = np.zeros(nm)
        V_row = np.zeros(mp)
        W_row = np.zeros(pn)
        for (i, j), val in u_dict.items():
            U_row[i * m + j] += val
        for (j, k), val in v_dict.items():
            V_row[j * p + k] += val
        for (k, i), val in w_dict.items():
            W_row[k * n + i] += val
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)

    # 4x4x4 Strassen specs
    # A blocks: A00=[0:2,0:2], A01=[0:2,2:4], A10=[2:4,0:2], A11=[2:4,2:4]
    # B blocks: B00=[0:2,0:2], B01=[0:2,2:4], B10=[2:4,0:2], B11=[2:4,2:4]
    # C blocks: C00=[0:2,0:2], C01=[0:2,2:4], C10=[2:4,0:2], C11=[2:4,2:4]
    
    # For each of the 7 Strassen products P_k, we need:
    # - A_eff: linear combination of A blocks
    # - B_eff: linear combination of B blocks
    # - C contribution: which C blocks get which coefficient
    
    # P0: A_eff = A00+A11, B_eff = B00+B11, C: C00+=1, C11+=1
    # P1: A_eff = A10+A11, B_eff = B00,      C: C10+=1, C11-=1
    # P2: A_eff = A00,      B_eff = B01-B11,  C: C01+=1, C11+=1
    # P3: A_eff = A11,      B_eff = B10-B00,  C: C00-=1, C10+=1
    # P4: A_eff = A00+A01,  B_eff = B11,      C: C00+=1, C01+=1
    # P5: A_eff = A01+A11,  B_eff = B00-B10,  C: C11+=1
    # P6: A_eff = A00+A01,  B_eff = B11-B01,  C: C00-=1

    # Block definitions: (row_start, col_start) for 2x2 blocks
    A_blocks = {
        'A00': (0, 0), 'A01': (0, 2),
        'A10': (2, 0), 'A11': (2, 2)
    }
    B_blocks = {
        'B00': (0, 0), 'B01': (0, 2),
        'B10': (2, 0), 'B11': (2, 2)
    }
    C_blocks = {
        'C00': (0, 0), 'C01': (0, 2),
        'C10': (2, 0), 'C11': (2, 2)
    }

    # Strassen product specs: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: list of (block_name, coeff)
    # B_coeffs: list of (block_name, coeff)
    # C_coeffs: dict from block_name to coeff
    strassen_4x4_specs = [
        # P0
        (
            [('A00', 1), ('A11', 1)],
            [('B00', 1), ('B11', 1)],
            {'C00': 1, 'C11': 1}
        ),
        # P1
        (
            [('A10', 1), ('A11', 1)],
            [('B00', 1)],
            {'C10': 1, 'C11': -1}
        ),
        # P2
        (
            [('A00', 1)],
            [('B01', 1), ('B11', -1)],
            {'C01': 1, 'C11': 1}
        ),
        # P3
        (
            [('A11', 1)],
            [('B10', 1), ('B00', -1)],
            {'C00': -1, 'C10': 1}
        ),
        # P4
        (
            [('A00', 1), ('A01', 1)],
            [('B11', 1)],
            {'C00': 1, 'C01': 1}
        ),
        # P5
        (
            [('A01', 1), ('A11', 1)],
            [('B00', 1), ('B10', -1)],
            {'C11': 1}
        ),
        # P6
        (
            [('A00', 1), ('A01', 1)],
            [('B11', 1), ('B01', -1)],
            {'C00': -1}
        ),
    ]

    def build_4x4x4(A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off):
        """Build 49 terms for 4x4*4x4 Strassen decomposition with given offsets."""
        for k_p in range(7):
            a_coeffs, b_coeffs, c_coeffs = strassen_4x4_specs[k_p]
            
            for m_s in range(7):
                u_dict = {}
                v_dict = {}
                w_dict = {}

                # U: A_eff indices
                for (block_name, coeff) in a_coeffs:
                    br, bc = A_blocks[block_name]
                    for r in range(2):
                        for c in range(2):
                            # Local 2x2 index: (r, c) -> U2 index r*2+c
                            u_val = U2[r * 2 + c, m_s]
                            if u_val != 0:
                                gi = A_row_off + br + r
                                gj = A_col_off + bc + c
                                u_dict[(gi, gj)] = u_dict.get((gi, gj), 0) + coeff * u_val

                # V: B_eff indices
                for (block_name, coeff) in b_coeffs:
                    br, bc = B_blocks[block_name]
                    for r in range(2):
                        for c in range(2):
                            # Local 2x2 index: (r, c) -> V2 index r*2+c
                            v_val = V2[r * 2 + c, m_s]
                            if v_val != 0:
                                gj = B_row_off + br + r
                                gk = B_col_off + bc + c
                                v_dict[(gj, gk)] = v_dict.get((gj, gk), 0) + coeff * v_val

                # W: C contribution
                for block_name, coeff_c in c_coeffs.items():
                    cr, cc = C_blocks[block_name]
                    for r in range(2):
                        for c in range(2):
                            # Local 2x2 index: (r, c) -> W2 index r*2+c
                            # But W is indexed by (k, i) in trace convention: k*2+i
                            # In the local 2x2 C block, C[r, c] means row r, col c
                            # So k=r, i=c in local coords
                            w_val = W2[r * 2 + c, m_s]
                            if w_val != 0:
                                gi = C_row_off + cr + r  # i index
                                gk = C_col_off + cc + c  # k index
                                w_dict[(gk, gi)] = w_dict.get((gk, gi), 0) + coeff_c * w_val

                if u_dict and v_dict and w_dict:
                    add_term(u_dict, v_dict, w_dict)

    # Apply for B_left (cols 0-3) and B_right (cols 4-7)
    build_4x4x4(0, 0, 0, 0, 0, 0)
    build_4x4x4(0, 0, 0, 4, 0, 4)

    rank = len(U_list)
    
    if rank == 0:
        U, V, W = _schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U = np.array(U_list).T  # shape (nm, rank)
    V = np.array(V_list).T  # shape (mp, rank)
    W = np.array(W_list).T  # shape (pn, rank)

    # Verify the decomposition
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-8:
        # Fallback to schoolbook
        U, V, W = _schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Check time
    if time.time() > deadline:
        # Shouldn't happen, but just in case
        pass

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
