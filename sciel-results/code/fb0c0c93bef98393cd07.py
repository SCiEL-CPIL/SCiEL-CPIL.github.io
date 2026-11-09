"""Strategy: Hardcode the rank-49 Strassen-based decomposition for 4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def _strassen_4x4():
    """
    Constructs the exact rank-49 decomposition for 4x4 matrix multiplication
    using the Strassen algorithm applied to 2x2 blocks.
    
    The 4x4 matrices are viewed as 2x2 blocks of 2x2 matrices.
    Strassen computes the product using 7 multiplications of 2x2 matrices.
    Each 2x2 multiplication can be done with rank 7 (Strassen for 2x2) or rank 4 (schoolbook).
    Using rank 7 for each of the 7 Strassen products gives 7 * 7 = 49 total rank.
    
    The tensors U, V, W have shapes:
    U: (16, 49)  # n*m = 4*4
    V: (16, 49)  # m*p = 4*4
    W: (16, 49)  # p*n = 4*4
    
    Indices:
    U[i, c] where i is row-major index of (i,j) in 4x4 A.
    V[j, c] where j is row-major index of (j,k) in 4x4 B.
    W[k, c] where k is row-major index of (k,i) in 4x4 C. Note: W[k*n+i, c] in prompt means 
    index is k*4 + i? 
    Prompt says: "output factor follows the official trace convention W[k*n+i,r]"
    Usually matrix multiplication tensor T_{ijk} = A_{ij} B_{jk} C_{ki}? No, standard is C_{ik} = sum_j A_{ij} B_{jk}.
    The tensor is T_{i,j,k} such that C_{ik} = sum_j A_{ij} B_{jk}.
    The decomposition is T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}.
    So W index is (k,i). Row-major index for W is k*4 + i.
    """
    rank = 49
    U = np.zeros((16, rank), dtype=object)
    V = np.zeros((16, rank), dtype=object)
    W = np.zeros((16, rank), dtype=object)
    
    # We need to map the 7 Strassen products (each of rank 7) into the global rank 49.
    # Let s be the Strassen step (0..6) and t be the internal Strassen rank (0..6).
    # Global index c = s * 7 + t.
    
    # Strassen formulas for 2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We are decomposing the 4x4 multiplication into 7 block multiplications.
    # Each block multiplication is 2x2 * 2x2 -> 2x2.
    # We use the rank-7 Strassen decomposition for each 2x2 block multiplication.
    # The rank-7 decomposition for 2x2 multiplication (indices a,b for A 2x2, c,d for B 2x2, e,f for C 2x2):
    # T_{a,b,c} = sum_{r=1..7} u_{a,r} v_{b,r} w_{c,r}
    
    # Let's define the rank-7 factors for a single 2x2 multiplication.
    # A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]], C = [[c0, c1], [c2, c3]]
    # c0 = a0*b0 + a1*b2
    # c1 = a0*b1 + a1*b3
    # c2 = a2*b0 + a3*b2
    # c3 = a2*b1 + a3*b3
    
    # Strassen 2x2 rank 7 factors:
    # r=0: (a0+a3)*(b0+b3) -> M1
    # r=1: (a2+a3)*b0 -> M2
    # r=2: a0*(b1-b3) -> M3
    # r=3: a3*(b2-b0) -> M4
    # r=4: (a0+a1)*b3 -> M5
    # r=5: (a2-a0)*(b0+b1) -> M6
    # r=6: (a1-a3)*(b2+b3) -> M7
    
    # C0 = M1 + M4 - M5 + M7
    # C1 = M3 + M5
    # C2 = M2 + M4
    # C3 = M1 - M2 + M3 + M6
    
    # Let's construct U_2x2, V_2x2, W_2x2 for the single 2x2 multiplication.
    # U_2x2[a, r], V_2x2[b, r], W_2x2[c, r]
    
    U2 = np.zeros((4, 7), dtype=object)
    V2 = np.zeros((4, 7), dtype=object)
    W2 = np.zeros((4, 7), dtype=object)
    
    # r=0: M1 = (a0+a3)(b0+b3)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C0 and C3 get M1
    
    # r=1: M2 = (a2+a3)b0
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1 # C2 gets M2
    W2[3, 1] = -1 # C3 gets -M2
    
    # r=2: M3 = a0(b1-b3)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1 # C1 gets M3
    W2[3, 2] = 1 # C3 gets M3
    
    # r=3: M4 = a3(b2-b0)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1 # C0 gets M4
    W2[2, 3] = 1 # C2 gets M4
    
    # r=4: M5 = (a0+a1)b3
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1 # C0 gets -M5
    W2[1, 4] = 1 # C1 gets M5
    
    # r=5: M6 = (a2-a0)(b0+b1)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C3 gets M6
    
    # r=6: M7 = (a1-a3)(b2+b3)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C0 gets M7
    
    # Now, we have 7 such block multiplications for the 4x4 case.
    # The 4x4 matrix A is composed of blocks A00, A01, A10, A11.
    # Block A00 corresponds to indices 0,1,4,5 in the 4x4 matrix (row-major).
    # Block A01 corresponds to indices 2,3,6,7.
    # Block A10 corresponds to indices 8,9,12,13.
    # Block A11 corresponds to indices 10,11,14,15.
    
    # Let's define the mapping from 4x4 indices to 2x2 block indices.
    # Block (bi, bj) where bi, bj in {0,1}.
    # Element (r, c) in block (bi, bj) corresponds to 4x4 index (2*bi + r, 2*bj + c).
    # Row-major index in 4x4: (2*bi + r)*4 + (2*bj + c).
    
    # Strassen steps for 4x4:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to fill U, V, W for the 4x4 case.
    # For each Strassen step s (0..6), and each internal rank t (0..6):
    # Global rank index c = s * 7 + t.
    
    # We will iterate over s and t, and determine which 4x4 indices are non-zero in U, V, W.
    
    for s in range(7):
        for t in range(7):
            c = s * 7 + t
            
            # Determine the factors for U, V, W based on s and t.
            # u_val, v_val, w_val are the coefficients for the specific 2x2 block elements.
            
            # Let's define a helper to get the 2x2 factors for a given combination of blocks.
            # This is complex to do generically, so let's hardcode the logic for each s.
            
            # It's easier to think of it as:
            # U_{4x4, c} is non-zero for elements in specific blocks.
            # V_{4x4, c} is non-zero for elements in specific blocks.
            # W_{4x4, c} is non-zero for elements in specific blocks.
            
            # Let's define the block combinations for U, V, W for each s.
            # For U:
            # s=0: A00+A11. U2 factors apply to both A00 and A11.
            # s=1: A10+A11. U2 factors apply to both A10 and A11.
            # s=2: A00. U2 factors apply to A00.
            # s=3: A11. U2 factors apply to A11.
            # s=4: A00+A01. U2 factors apply to both A00 and A01.
            # s=5: A10-A00. U2 factors apply to A10 (positive) and A00 (negative).
            # s=6: A01-A11. U2 factors apply to A01 (positive) and A11 (negative).
            
            # For V:
            # s=0: B00+B11.
            # s=1: B00.
            # s=2: B01-B11.
            # s=3: B10-B00.
            # s=4: B11.
            # s=5: B00+B01.
            # s=6: B10+B11.
            
            # For W:
            # s=0: C00, C11. (M1 contributes to C00 and C11)
            # s=1: C10 (M2), C11 (-M2).
            # s=2: C01 (M3), C11 (M3).
            # s=3: C00 (M4), C10 (M4).
            # s=4: C00 (-M5), C01 (M5).
            # s=5: C11 (M6).
            # s=6: C00 (M7).
            
            # Let's implement this by iterating over the 2x2 block indices (bi, bj) for U and V,
            # and the 2x2 output block indices (ci, cj) for W.
            
            # Block indices in 4x4:
            # Block (bi, bj) has elements at 4x4 indices:
            # (2*bi, 2*bj), (2*bi, 2*bj+1), (2*bi+1, 2*bj), (2*bi+1, 2*bj+1)
            # Row-major indices:
            # 4*(2*bi) + 2*bj
            # 4*(2*bi) + 2*bj + 1
            # 4*(2*bi+1) + 2*bj
            # 4*(2*bi+1) + 2*bj + 1
            
            def block_indices(bi, bj):
                return [
                    4*(2*bi) + 2*bj,
                    4*(2*bi) + 2*bj + 1,
                    4*(2*bi+1) + 2*bj,
                    4*(2*bi+1) + 2*bj + 1
                ]
            
            # For each s, we define the set of blocks and signs for U, V, W.
            # u_blocks: list of (block_idx, sign)
            # v_blocks: list of (block_idx, sign)
            # w_blocks: list of (block_idx, sign)
            
            if s == 0:
                u_blocks = [(0, 1), (3, 1)] # A00, A11
                v_blocks = [(0, 1), (3, 1)] # B00, B11
                w_blocks = [(0, 1), (3, 1)] # C00, C11
            elif s == 1:
                u_blocks = [(2, 1), (3, 1)] # A10, A11
                v_blocks = [(0, 1)] # B00
                w_blocks = [(2, 1), (3, -1)] # C10, C11
            elif s == 2:
                u_blocks = [(0, 1)] # A00
                v_blocks = [(1, 1), (3, -1)] # B01, B11
                w_blocks = [(1, 1), (3, 1)] # C01, C11
            elif s == 3:
                u_blocks = [(3, 1)] # A11
                v_blocks = [(2, 1), (0, -1)] # B10, B00
                w_blocks = [(0, 1), (2, 1)] # C00, C10
            elif s == 4:
                u_blocks = [(0, 1), (1, 1)] # A00, A01
                v_blocks = [(3, 1)] # B11
                w_blocks = [(0, -1), (1, 1)] # C00, C01
            elif s == 5:
                u_blocks = [(2, 1), (0, -1)] # A10, A00
                v_blocks = [(0, 1), (1, 1)] # B00, B01
                w_blocks = [(3, 1)] # C11
            elif s == 6:
                u_blocks = [(1, 1), (3, -1)] # A01, A11
                v_blocks = [(2, 1), (3, 1)] # B10, B11
                w_blocks = [(0, 1)] # C00
            
            # Now apply the 2x2 factors U2[:, t], V2[:, t], W2[:, t] to these blocks.
            
            for (bi, bj), sign_u in u_blocks:
                inds = block_indices(bi, bj)
                for local_i in range(4):
                    global_i = inds[local_i]
                    U[global_i, c] += sign_u * U2[local_i, t]
            
            for (bi, bj), sign_v in v_blocks:
                inds = block_indices(bi, bj)
                for local_j in range(4):
                    global_j = inds[local_j]
                    V[global_j, c] += sign_v * V2[local_j, t]
            
            for (ci, cj), sign_w in w_blocks:
                inds = block_indices(ci, cj)
                for local_k in range(4):
                    # W index is (k, i) in the tensor convention?
                    # The prompt says W[k*n+i, r].
                    # In the 2x2 block decomposition, the output block is C_{ci, cj}.
                    # The elements of C_{ci, cj} are C_{2*ci + r, 2*cj + c}.
                    # The tensor index for W is (k, i) where k is the column of C and i is the row of C?
                    # No, T_{i,j,k} = A_{ij} B_{jk} W_{k,i}.
                    # So W index is (k, i).
                    # In the block C_{ci, cj}, the row is i = 2*ci + r, column is k = 2*cj + c.
                    # So the W index is (2*cj + c, 2*ci + r).
                    # Row-major index for W: (2*cj + c)*4 + (2*ci + r).
                    
                    # Let's re-verify the W index mapping.
                    # U is indexed by (i, j) of A.
                    # V is indexed by (j, k) of B.
                    # W is indexed by (k, i) of C.
                    
                    # In the 2x2 block C_{ci, cj}, the local row is r (0..1), local col is c (0..1).
                    # Global row i = 2*ci + r.
                    # Global col k = 2*cj + c.
                    # W index is (k, i).
                    # Row-major index = k*4 + i = (2*cj + c)*4 + (2*ci + r).
                    
                    global_w_idx = (2*cj + local_k % 2)*4 + (2*ci + local_k // 2)
                    # Wait, local_k in block_indices is the row-major index of the 2x2 block.
                    # local_k = 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1).
                    # So local row r = local_k // 2, local col c = local_k % 2.
                    
                    W[global_w_idx, c] += sign_w * W2[local_k, t]
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=0):
    """Exact integer verification."""
    rank = U.shape[1]
    U = np.asarray(U, dtype=object)
    V = np.asarray(V, dtype=object)
    W = np.asarray(W, dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                expected = 1 if i == k else 0
                if s != expected:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Generate the rank-49 decomposition
    U, V, W = _strassen_4x4()
    
    # Verify it
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=object)
        V = np.zeros((m * p, rank), dtype=object)
        W = np.zeros((p * n, rank), dtype=object)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
    
    # Convert to lists
    U_list = [[int(x) for x in row] for row in np.asarray(U, dtype=object)]
    V_list = [[int(x) for x in row] for row in np.asarray(V, dtype=object)]
    W_list = [[int(x) for x in row] for row in np.asarray(W, dtype=object)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END