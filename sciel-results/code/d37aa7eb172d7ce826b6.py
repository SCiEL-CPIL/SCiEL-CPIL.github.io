"""Rewritten program for matrix multiplication 4,4,8 rank-minimal tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    deadline = time.time() + 130.0  # safety margin
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 128
    
    def _make_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    def _verify_exact(U_list, V_list, W_list, rank):
        """Verify T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k."""
        if len(U_list) != n * m or len(V_list) != m * p or len(W_list) != p * n:
            return False
        if any(len(row) != rank for row in U_list) or any(len(row) != rank for row in V_list) or any(len(row) != rank for row in W_list):
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(rank):
                        total += U_list[i * m + j][c] * V_list[j * p + k][c] * W_list[k * n + i][c]
                    if total != Fraction(1):
                        return False
        return True
    
    def _verify_float(U, V, W, rank, tol=1e-9):
        """Quick float check."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def _to_halfint_fraction(mat):
        """Convert numpy array to list of Fractions (round to nearest 0.5)."""
        result = []
        for row in mat:
            new_row = []
            for val in row:
                h = round(val * 2) / 2.0
                new_row.append(Fraction(h).limit_denominator(2))
            result.append(new_row)
        return result
    
    def _error_matrix(U, V, W, rank):
        """Compute error tensor E[i,j,k] = 1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]."""
        E = np.ones((n, m, p))
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        E[i, j, k] -= U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
        return E
    
    def _total_error(U, V, W, rank):
        E = _error_matrix(U, V, W, rank)
        return np.sum(E * E)
    
    # Start with schoolbook
    best_U, best_V, best_W = _make_schoolbook()
    best_rank = schoolbook_rank
    
    # --- Strategy 1: Try to reduce rank via column elimination / sparsification ---
    # Start from schoolbook and try to zero out entries while maintaining correctness.
    # This is a form of "peeling" the tensor.
    
    # Convert to numpy for computation
    U_np = np.array(best_U, dtype=float)
    V_np = np.array(best_V, dtype=float)
    W_np = np.array(best_W, dtype=float)
    current_rank = best_rank
    
    # Try to eliminate one term at a time: set a column to zero and compensate
    # Actually, a better approach: try random rank-r decompositions for r < 128
    
    # --- Strategy 2: Randomized search for lower rank ---
    # For matrix multiplication C = A*B where A is (n x m), B is (m x p), C is (n x p)
    # The tensor is T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # We need: sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    # Lower bound: max(n*m, m*p, p*n) = max(16, 32, 32) = 32
    # Known results: For 4x4*4x4, rank is at most 49 (or 48 with improvements).
    # For 4x4*4x8, we can think of it as 2 independent 4x4*4x4 products, so rank <= 2*49 = 98
    # But we can do better by sharing structure.
    
    # Let's try a structured approach: partition B into two 4x4 blocks B1, B2
    # C = A*B1 + A*B2
    # Each 4x4*4x4 multiplication has rank <= 49 (using known optimal or near-optimal)
    # Total rank <= 98
    
    # Even better: use the fact that A is shared.
    # C1 = A*B1, C2 = A*B2
    # T1[i,j,k] for k in 0..3, T2[i,j,k] for k in 4..7
    
    # Let's implement a direct construction for 4x4*4x8 with rank < 128.
    # Approach: Use the Lade胜 algorithm or similar for rectangular matrix multiplication.
    
    # Actually, let me try a simple but effective approach:
    # Use the schoolbook for the innermost 4x4 products but share the A-factor.
    
    # Partition: B = [B1 | B2] where B1, B2 are 4x4
    # C = [C1 | C2] where C1 = A*B1, C2 = A*B2
    # For C1: T1[i,j,k] = sum_j A[i,j]*B1[j,k] = 1 for i,j,k in [0,4)
    # For C2: T2[i,j,k] = sum_j A[i,j]*B2[j,k] = 1 for i,j,k in [0,4)
    
    # Schoolbook for each: rank 16 each, total 32... wait that's wrong.
    # Schoolbook for 4x4*4x4: rank = 4*4*4 = 64 per product, total 128.
    
    # Let me use a known rank-49 decomposition for 4x4*4x4.
    # The Strassen algorithm gives rank 7 for 2x2*2x2.
    # For 4x4*4x4, we can partition into 2x2 blocks and use Strassen: rank 7*8 = 56.
    # There are better algorithms: rank 49 is known for 4x4*4x4.
    
    # Let me implement the rank-56 Strassen-based decomposition for 4x4*4x4,
    # then use it twice for the two halves of B.
    
    # Strassen for 2x2*2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4*4x4 with 2x2 block partition:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]] where each block is 2x2
    # C = [[C11, C12], [C21, C22]] where each block is 2x2
    # Each block product (e.g., A11*B11) is a 2x2*2x2 product, computed by Strassen (rank 7)
    # But Strassen combines them: the 7 Strassen products each involve 2x2*2x2 multiplications
    # Wait, no. In Strassen for 4x4, the "scalar" multiplications are replaced by 2x2*2x2 multiplications.
    # Each 2x2*2x2 multiplication uses schoolbook (rank 8) or Strassen (rank 7).
    # If we use schoolbook for the 2x2*2x2: total rank = 7 * 8 = 56
    # If we use Strassen for the 2x2*2x2: total rank = 7 * 7 = 49
    
    # Let's build the rank-49 decomposition for 4x4*4x4 using nested Strassen.
    
    def _build_4x4_4x4_decomp(rank_inner=8):
        """
        Build U, V, W for 4x4*4x4 matrix multiplication.
        n_inner = m_inner = p_inner = 4
        Returns (U, V, W) as numpy arrays with shapes (16, R), (16, R), (16, R)
        where R is the total rank.
        
        Uses 2x2 block Strassen with inner 2x2*2x2 products.
        If rank_inner=8, total rank = 56.
        If rank_inner=7, total rank = 49.
        """
        # The 7 Strassen combinations for the outer 2x2 level:
        # Each combination is: X_t * Y_t where X_t, Y_t are 2x2 block matrices
        # X_t is a linear combo of A blocks, Y_t is a linear combo of B blocks
        
        # Strassen products (outer level):
        # P1 = (A11+A22) * (B11+B22)
        # P2 = (A21+A22) * B11
        # P3 = A11 * (B12-B22)
        # P4 = A22 * (B21-B11)
        # P5 = (A11+A12) * B22
        # P6 = (A21-A11) * (B11+B12)
        # P7 = (A12-A22) * (B21+B22)
        
        # Results:
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # Each Pt is a 2x2 matrix = result of a 2x2*2x2 product.
        # A 2x2*2x2 product via schoolbook has rank 8.
        
        # For each Pt, we need a decomposition of the 2x2*2x2 tensor.
        # Schoolbook 2x2*2x2: 8 terms.
        # Strassen 2x2*2x2: 7 terms.
        
        if rank_inner == 8:
            # Schoolbook 2x2*2x2
            def _inner_2x2_2x2_schoolbook():
                """Return (u, v, w) for 2x2*2x2 schoolbook, shapes (4,8),(4,8),(4,8)"""
                u = np.zeros((4, 8))
                v = np.zeros((4, 8))
                w = np.zeros((4, 8))
                col = 0
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            u[i * 2 + j, col] = 1
                            v[j * 2 + k, col] = 1
                            w[k * 2 + i, col] = 1
                            col += 1
                return u, v, w
        elif rank_inner == 7:
            # Strassen 2x2*2x2
            def _inner_2x2_2x2_strassen():
                """Return (u, v, w) for 2x2*2x2 Strassen, shapes (4,7),(4,7),(4,7)"""
                # Strassen:
                # M1 = (a11+a22)(b11+b22)
                # M2 = (a21+a22)b11
                # M3 = a11(b12-b22)
                # M4 = a22(b21-b11)
                # M5 = (a11+a12)b22
                # M6 = (a21-a11)(b11+b12)
                # M7 = (a12-a22)(b21+b22)
                # c11 = M1 + M4 - M5 + M7
                # c12 = M3 + M5
                # c21 = M2 + M4
                # c22 = M1 - M2 + M3 + M6
                
                u = np.zeros((4, 7))
                v = np.zeros((4, 7))
                w = np.zeros((4, 7))
                
                # Indices: A: [0]=a11, [1]=a12, [2]=a21, [3]=a22
                #          B: [0]=b11, [1]=b12, [2]=b21, [3]=b22
                #          C: [0]=c11, [1]=c12, [2]=c21, [3]=c22
                
                # M1 = (a11+a22)(b11+b22)
                u[0, 0] = 1; u[3, 0] = 1
                v[0, 0] = 1; v[3, 0] = 1
                w[0, 0] = 1; w[1, 0] = 1; w[2, 0] = 1; w[3, 0] = 1  # c11+=1, c12+=1, c21+=1, c22+=1
                
                # M2 = (a21+a22)b11
                u[2, 1] = 1; u[3, 1] = 1
                v[0, 1] = 1
                w[2, 1] = 1; w[3, 1] = -1  # c21+=1, c22+=-1
                
                # M3 = a11(b12-b22)
                u[0, 2] = 1
                v[1, 2] = 1; v[3, 2] = -1
                w[1, 2] = 1; w[3, 2] = 1  # c12+=1, c22+=1
                
                # M4 = a22(b21-b11)
                u[3, 3] = 1
                v[2, 3] = 1; v[0, 3] = -1
                w[0, 3] = 1; w[2, 3] = 1  # c11+=1, c21+=1
                
                # M5 = (a11+a12)b22
                u[0, 4] = 1; u[1, 4] = 1
                v[3, 4] = 1
                w[0, 4] = -1; w[1, 4] = 1  # c11+=-1, c12+=1
                
                # M6 = (a21-a11)(b11+b12)
                u[2, 5] = 1; u[0, 5] = -1
                v[0, 5] = 1; v[1, 5] = 1
                w[3, 5] = 1  # c22+=1
                
                # M7 = (a12-a22)(b21+b22)
                u[1, 6] = 1; u[3, 6] = -1
                v[2, 6] = 1; v[3, 6] = 1
                w[0, 6] = 1  # c11+=1
                
                return u, v, w
        
        # Now build the 4x4*4x4 decomposition
        # A blocks: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3), A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
        # In flattened (i*m+j) form: A11: [0,1,4,5], A12: [2,3,6,7], A21: [8,9,12,13], A22: [10,11,14,15]
        # B blocks: B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3), B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3)
        # In flattened (j*p+k) form: B11: [0,1,4,5], B12: [2,3,6,7], B21: [8,9,12,13], B22: [10,11,14,15]
        # C blocks: C11 (rows 0-1, cols 0-1), C12 (rows 0-1, cols 2-3), C21 (rows 2-3, cols 0-1), C22 (rows 2-3, cols 2-3)
        # In flattened (k*n+i) form: C11: [0,1,4,5], C12: [2,3,6,7], C21: [8,9,12,13], C22: [10,11,14,15]
        
        if rank_inner == 8:
            u_inner, v_inner, w_inner = _inner_2x2_2x2_schoolbook()
        else:
            u_inner, v_inner, w_inner = _inner_2x2_2x2_strassen()
        
        inner_rank = rank_inner
        total_rank = 7 * inner_rank
        
        U = np.zeros((16, total_rank))
        V = np.zeros((16, total_rank))
        W = np.zeros((16, total_rank))
        
        # Block index mappings for inner 2x2 within each 2x2 block:
        # For a 2x2 block at position (br, bc) in the 4x4 matrix:
        # The 4 elements are at positions:
        #   A: (br*2+0)*4 + (bc*2+0), (br*2+0)*4 + (bc*2+1), (br*2+1)*4 + (bc*2+0), (br*2+1)*4 + (bc*2+1)
        #   In flattened i*m+j form: (br*2+0)*4+(bc*2+0), (br*2+0)*4+(bc*2+1), (br*2+1)*4+(bc*2+0), (br*2+1)*4+(bc*2+1)
        # Inner 2x2 indices: [0]=top-left, [1]=top-right, [2]=bottom-left, [3]=bottom-right
        
        def _block_indices_A(br, bc):
            """Return 4 indices into flattened A (i*4+j) for block (br,bc)."""
            return [
                (br * 2 + 0) * 4 + (bc * 2 + 0),
                (br * 2 + 0) * 4 + (bc * 2 + 1),
                (br * 2 + 1) * 4 + (bc * 2 + 0),
                (br * 2 + 1) * 4 + (bc * 2 + 1),
            ]
        
        def _block_indices_B(br, bc):
            """Return 4 indices into flattened B (j*4+k) for block (br,bc)."""
            return [
                (br * 2 + 0) * 4 + (bc * 2 + 0),
                (br * 2 + 0) * 4 + (bc * 2 + 1),
                (br * 2 + 1) * 4 + (bc * 2 + 0),
                (br * 2 + 1) * 4 + (bc * 2 + 1),
            ]
        
        def _block_indices_C(br, bc):
            """Return 4 indices into flattened C (k*4+i) for block (br,bc)."""
            return [
                (br * 2 + 0) * 4 + (bc * 2 + 0),
                (br * 2 + 0) * 4 + (bc * 2 + 1),
                (br * 2 + 1) * 4 + (bc * 2 + 0),
                (br * 2 + 1) * 4 + (bc * 2 + 1),
            ]
        
        # A blocks: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
        A11_idx = _block_indices_A(0, 0)
        A12_idx = _block_indices_A(0, 1)
        A21_idx = _block_indices_A(1, 0)
        A22_idx = _block_indices_A(1, 1)
        
        # B blocks: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
        B11_idx = _block_indices_B(0, 0)
        B12_idx = _block_indices_B(0, 1)
        B21_idx = _block_indices_B(1, 0)
        B22_idx = _block_indices_B(1, 1)
        
        # C blocks: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
        C11_idx = _block_indices_C(0, 0)
        C12_idx = _block_indices_C(0, 1)
        C21_idx = _block_indices_C(1, 0)
        C22_idx = _block_indices_C(1, 1)
        
        # For each Strassen product t=0..6, for each inner product c=0..inner_rank-1:
        # We need to set U, V, W entries.
        
        # Strassen combinations (outer level):
        # t=0: P1 = (A11+A22)*(B11+B22)
        # t=1: P2 = (A21+A22)*B11
        # t=2: P3 = A11*(B12-B22)
        # t=3: P4 = A22*(B21-B11)
        # t=4: P5 = (A11+A12)*B22
        # t=5: P6 = (A21-A11)*(B11+B12)
        # t=6: P7 = (A12-A22)*(B21+B22)
        
        # For each product t, the A-side is a linear combination of A blocks.
        # Each A block is 2x2, so the "A-side matrix" for product t is 2x2.
        # The inner 2x2*2x2 decomposition works on this 2x2 A-side and 2x2 B-side.
        
        # For product t, the A-side 2x2 matrix has entries:
        # A_t[0,0] = sum of coefficients * A_block[0,0]
        # A_t[0,1] = sum of coefficients * A_block[0,1]
        # A_t[1,0] = sum of coefficients * A_block[1,0]
        # A_t[1,1] = sum of coefficients * A_block[1,1]
        
        # Similarly for B-side.
        
        # Let me define for each product t, the coefficients for each A block and B block.
        # coeff_A[t][block] = coefficient for that block in the A-side linear combo
        # coeff_B[t][block] = coefficient for that block in the B-side linear combo
        
        # Block order: 0=A11/B11, 1=A12/B12, 2=A21/B21, 3=A22/B22
        
        coeff_A = [
            [1, 0, 0, 1],  # P1: A11 + A22
            [0, 0, 1, 1],  # P2: A21 + A22
            [1, 0, 0, 0],  # P3: A11
            [0, 0, 0, 1],  # P4: A22
            [1, 1, 0, 0],  # P5: A11 + A12
            [-1, 0, 1, 0],  # P6: A21 - A11
            [0, 1, 0, -1],  # P7: A12 - A22
        ]
        
        coeff_B = [
            [1, 0, 0, 1],  # P1: B11 + B22
            [1, 0, 0, 0],  # P2: B11
            [0, 1, 0, -1],  # P3: B12 - B22
            [0, 0, 1, -1],  # P4: B21 - B11
            [0, 0, 0, 1],  # P5: B22
            [1, 1, 0, 0],  # P6: B11 + B12
            [0, 0, 1, 1],  # P7: B21 + B22
        ]
        
        # Result contributions:
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # result_coeff[t][result_block] = coefficient
        # result_block: 0=C11, 1=C12, 2=C21, 3=C22
        
        result_coeff = [
            [1, 0, 0, 0],   # P1 -> C11
            [0, 0, 1, -1],  # P2 -> C21, C22
            [0, 1, 0, 1],   # P3 -> C12, C22
            [1, 0, 1, 0],   # P4 -> C11, C21
            [-1, 1, 0, 0],  # P5 -> C11, C12
            [0, 0, 0, 1],   # P6 -> C22
            [1, 0, 0, 0],   # P7 -> C11
        ]
        
        # Now fill U, V, W
        for t in range(7):
            for c in range(inner_rank):
                col = t * inner_rank + c
                
                # A-side: for each inner element (ii, jj) in the 2x2 A-side matrix:
                # A_t[ii, jj] = sum over blocks b of coeff_A[t][b] * A_b[ii, jj]
                # The inner 2x2 index for A is ii*2+jj
                # U[row, col] where row is the flattened index in A
                for ii in range(2):
                    for jj in range(2):
                        inner_idx = ii * 2 + jj
                        # u_inner[inner_idx, c] gives the coefficient for this inner element
                        u_coeff = u_inner[inner_idx, c]
                        if u_coeff == 0:
                            continue
                        for b in range(4):
                            a_coeff = coeff_A[t][b]
                            if a_coeff == 0:
                                continue
                            # Get the actual index in the 4x4 A matrix
                            if b == 0:
                                block_idx = A11_idx
                            elif b == 1:
                                block_idx = A12_idx
                            elif b == 2:
                                block_idx = A21_idx
                            else:
                                block_idx = A22_idx
                            # inner_idx maps to position within the 2x2 block
                            U[block_idx[inner_idx], col] += u_coeff * a_coeff
                
                # B-side: similar
                for ii in range(2):
                    for jj in range(2):
                        inner_idx = ii * 2 + jj
                        v_coeff = v_inner[inner_idx, c]
                        if v_coeff == 0:
                            continue
                        for b in range(4):
                            b_coeff = coeff_B[t][b]
                            if b_coeff == 0:
                                continue
                            if b == 0:
                                block_idx = B11_idx
                            elif b == 1:
                                block_idx = B12_idx
                            elif b == 2:
                                block_idx = B21_idx
                            else:
                                block_idx = B22_idx
                            V[block_idx[inner_idx], col] += v_coeff * b_coeff
                
                # C-side: for each inner element in the 2x2 C-side matrix:
                for ii in range(2):
                    for jj in range(2):
                        inner_idx = ii * 2 + jj
                        w_coeff = w_inner[inner_idx, c]
                        if w_coeff == 0:
                            continue
                        for rb in range(4):
                            r_coeff = result_coeff[t][rb]
                            if r_coeff == 0:
                                continue
                            if rb == 0:
                                block_idx = C11_idx
                            elif rb == 1:
                                block_idx = C12_idx
                            elif rb == 2:
                                block_idx = C21_idx
                            else:
                                block_idx = C22_idx
                            W[block_idx[inner_idx], col] += w_coeff * r_coeff
        
        return U, V, W, total_rank
    
    # Build the 4x4*4x4 decomposition
    U44, V44, W44, rank44 = _build_4x4_4x4_decomp(rank_inner=7)
    
    # Verify the 4x4*4x4 decomposition
    def _verify_4x4(U, V, W, rank):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * 4 + j, c] * V[j * 4 + k, c] * W[k * 4 + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False, (i, j, k, s)
        return True, None
    
    ok, info = _verify_4x4(U44, V44, W44, rank44)
    
    if ok:
        # Now build the 4x4*4x8 decomposition using two 4x4*4x4 decompositions
        # B = [B1 | B2], each 4x4
        # C = [C1 | C2], each 4x4
        # C1 = A * B1, C2 = A * B2
        
        total_rank = 2 * rank44  # 98
        
        U_full = np.zeros((16, total_rank))
        V_full = np.zeros((32, total_rank))
        W_full = np.zeros((32, total_rank))
        
        # For C1 (k=0..3): use U44, V44 for B1 (j=0..3, k=0..3), W44 for C1
        # U44 has shape (16, rank44)
        # V44 has shape (16, rank44) -> but we need V for B1 which is 4x4, indices j*4+k for j,k in 0..3
        # W44 has shape (16, rank44) -> for C1, k*4+i for k,i in 0..3
        
        # Copy U44 to first half
        U_full[:, :rank44] = U44
        
        # V for B1: rows 0..15 (j*4+k for j,k in 0..3)
        V_full[:16, :rank44] = V44
        
        # W for C1: rows 0..15 (k*4+i for k,i in 0..3)
        W_full[:16, :rank44] = W44
        
        # For C2 (k=4..7): use U44, V44 for B2 (j=0..3, k=4..7), W44 for C2
        U_full[:, rank44:] = U44
        
        # V for B2: rows 16..31 (j*4+k for j in 0..3, k in 4..7) -> index = j*4 + (k-4) + 16... 
        # Actually, V index is j*p+k = j*4+k. For B2, k ranges 4..7.
        # So V indices for B2 are j*4+4, j*4+5, j*4+6, j*4+7 for j=0..3
        # That's rows 4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18,19
        # Wait, j ranges 0..3, k ranges 4..7:
        # j=0: 4,5,6,7
        # j=1: 8,9,10,11
        # j=2: 12,13,14,15
        # j=3: 16,17,18,19
        
        # V44 has indices j*4+k' where k'=0..3 (for the local 4x4 block)
        # So V44[j*4+k'] corresponds to V_full[j*4+k'+4, rank44+c]
        
        for j in range(4):
            for k_prime in range(4):
                v_idx = j * 4 + k_prime
                v_full_idx = j * 4 + (k_prime + 4)  # = j*4 + k_prime + 4
                V_full[v_full_idx, rank44:] = V44[v_idx, :]
        
        # W for C2: k ranges 4..7, i ranges 0..3
        # W index is k*n+i = k*4+i
        # For C2: k=4..7, i=0..3
        # W indices: 16..31
        # W44 has indices k'*4+i where k'=0..3
        # So W44[k'*4+i] corresponds to W_full[(k'+4)*4+i, rank44+c]
        
        for k_prime in range(4):
            for i in range(4):
                w_idx = k_prime * 4 + i
                w_full_idx = (k_prime + 4) * 4 + i
                W_full[w_full_idx, rank44:] = W44[w_idx, :]
        
        # Verify the full 4x4*4x8 decomposition
        def _verify_full(U, V, W, rank):
            for i in range(4):
                for j in range(4):
                    for k in range(8):
                        s = 0.0
                        for c in range(rank):
                            s += U[i * 4 + j, c] * V[j * 8 + k, c] * W[k * 4 + i, c]
                        if abs(s - 1.0) > 1e-9:
                            return False, (i, j, k, s)
            return True, None
        
        ok_full, info_full = _verify_full(U_full, V_full, W_full, total_rank)
        
        if ok_full:
            # Convert to fractions
            U_frac = _to_halfint_fraction(U_full)
            V_frac = _to_halfint_fraction(V_full)
            W_frac = _to_halfint_fraction(W_full)
            
            # Verify with fractions
            if _verify_exact(U_frac, V_frac, W_frac, total_rank):
                return {"U": U_frac, "V": V_frac, "W": W_frac}
            else:
                # Float verification passed but fraction didn't - this shouldn't happen
                # with integer/half-integer entries
                pass
    
    # Fallback: try schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
