import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = 3, 4, 8
    
    def verify_decomp(U, V, W, n, m, p, tol=1e-9):
        """U: (n*m, r), V: (m*p, r), W: (p*n, r) flat. Verify tensor decomposition."""
        r = U.shape[1]
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        rng = np.random.default_rng(42)
        for _ in range(5):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            C = A @ B
            
            u = np.einsum('ijr,ij->ir', U3, A)
            v = np.einsum('jkr,jk->kr', V3, B)
            C_dec = np.einsum('kir,ir,kr->ik', W3, u, v)
            
            if np.max(np.abs(C - C_dec)) > tol:
                return False
        return True
    
    def get_strassen_222():
        """Return U, V, W for 2x2x2 Strassen decomposition.
        U: (2,2,7), V: (2,2,7), W: (2,2,7)
        T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
        """
        # Strassen's 7 multiplications for 2x2 matrices
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M7
        
        r = 7
        U = np.zeros((2, 2, r))
        V = np.zeros((2, 2, r))
        W = np.zeros((2, 2, r))
        
        # r=0: M1 = (A11+A22)(B11+B22)
        # U contributes to A linear form: A11+A22 -> U[0,0,0]=1, U[1,1,0]=1
        # V contributes to B linear form: B11+B22 -> V[0,0,0]=1, V[1,1,0]=1
        # W: C11 += M1, C22 += M1 -> W[0,0,0]=1, W[1,1,0]=1
        U[0,0,0]=1; U[1,1,0]=1
        V[0,0,0]=1; V[1,1,0]=1
        W[0,0,0]=1; W[1,1,0]=1
        
        # r=1: M2 = (A21+A22)B11
        U[1,0,1]=1; U[1,1,1]=1
        V[0,0,1]=1
        W[1,0,1]=1  # C21 += M2
        
        # r=2: M3 = A11(B12-B22)
        U[0,0,2]=1
        V[0,1,2]=1; V[1,1,2]=-1
        W[0,1,2]=1  # C12 += M3
        
        # r=3: M4 = A22(B21-B11)
        U[1,1,3]=1
        V[1,0,3]=1; V[0,0,3]=-1
        W[0,0,3]=1; W[1,0,3]=1  # C11 += M4, C21 += M4
        
        # r=4: M5 = (A11+A12)B22
        U[0,0,4]=1; U[0,1,4]=1
        V[1,1,4]=1
        W[0,0,4]=-1  # C11 -= M5
        
        # r=5: M6 = (A21-A11)(B11+B12)
        U[1,0,5]=1; U[0,0,5]=-1
        V[0,0,5]=1; V[0,1,5]=1
        # M6 is not used in standard Strassen 2x2... let me recheck.
        # Actually standard Strassen doesn't use M6 for 2x2. Let me use the correct 7.
        
        # Let me redo with the standard 7 Strassen formulas:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # Reset
        U = np.zeros((2, 2, r))
        V = np.zeros((2, 2, r))
        W = np.zeros((2, 2, r))
        
        # M1 = (A11+A22)(B11+B22)
        U[0,0,0]=1; U[1,1,0]=1
        V[0,0,0]=1; V[1,1,0]=1
        W[0,0,0]=1; W[1,1,0]=1
        
        # M2 = (A21+A22)B11
        U[1,0,1]=1; U[1,1,1]=1
        V[0,0,1]=1
        W[1,0,1]=1
        
        # M3 = A11(B12-B22)
        U[0,0,2]=1
        V[0,1,2]=1; V[1,1,2]=-1
        W[0,1,2]=1; W[1,1,2]=1
        
        # M4 = A22(B21-B11)
        U[1,1,3]=1
        V[1,0,3]=1; V[0,0,3]=-1
        W[0,0,3]=1; W[1,0,3]=1
        
        # M5 = (A11+A12)B22
        U[0,0,4]=1; U[0,1,4]=1
        V[1,1,4]=1
        W[0,0,4]=-1; W[0,1,4]=1
        
        # M6 = (A21-A11)(B11+B12)
        U[1,0,5]=1; U[0,0,5]=-1
        V[0,0,5]=1; V[0,1,5]=1
        # M6 not used in standard 2x2 Strassen output. 
        # Actually, I'm confusing myself. Let me use the well-known 7-term formula.
        
        # Standard Strassen 2x2:
        # p1 = (a00+a11)(b00+b11)
        # p2 = (a10+a11)b00
        # p3 = a00(b01-b11)
        # p4 = a11(b10-b00)
        # p5 = (a00+a01)b11
        # p6 = (a10-a00)(b00+b01)
        # p7 = (a01-a11)(b10+b11)
        # c00 = p1+p4-p5+p7
        # c01 = p3+p5
        # c10 = p2+p4
        # c11 = p1-p2+p3+p7
        
        # Wait, this has 7 terms but p6 isn't used in the output. That's wrong.
        # The correct Strassen has all 7 used. Let me look up properly.
        
        # Correct Strassen 2x2:
        # p1 = (a00+a11)(b00+b11)
        # p2 = (a10+a11)b00
        # p3 = a00(b01-b11)
        # p4 = a11(b10-b00)
        # p5 = (a00+a01)b11
        # p6 = (a10-a00)(b00+b01)
        # p7 = (a01-a11)(b10+b11)
        # c00 = p1 + p4 - p5 + p7
        # c01 = p3 + p5
        # c10 = p2 + p4
        # c11 = p1 - p2 + p3 + p7
        
        # p6 is not used! That means this isn't the right formula.
        # The actual Strassen algorithm:
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)b11
        # m3 = a11(b12-b22)
        # m4 = a22(b21-b11)
        # m5 = (a11+a12)b22
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        # c11 = m1+m4-m5+m7
        # c12 = m3+m5
        # c21 = m2+m4
        # c22 = m1-m2+m3+m7
        
        # Yes, m6 is not used in the final combination. This is correct for Strassen.
        # The tensor rank is 7 because we need 7 scalar multiplications, even though
        # one of the linear forms (m6) doesn't appear in the output. Wait, that can't be right.
        # Actually m6 IS used - I'm misremembering. Let me just verify by construction.
        
        # Let me just hardcode the known correct Strassen factors and verify.
        # I'll use a different well-known representation.
        
        # Actually, the simplest approach: just verify by construction.
        # For the tensor T[i,j,k]=1 for 2x2x2, we need:
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        # Let me use the following known rank-7 decomposition:
        # This is from the literature. I'll verify it numerically.
        
        U = np.zeros((2, 2, 7))
        V = np.zeros((2, 2, 7))
        W = np.zeros((2, 2, 7))
        
        # Using the standard Strassen formulation where:
        # U[i,j,r] gives the coefficient of A[i,j] in the r-th multiplication
        # V[j,k,r] gives the coefficient of B[j,k] in the r-th multiplication  
        # W[k,i,r] gives the coefficient of the r-th multiplication in C[k,i]... 
        # Wait, W[k,i,r] means W is indexed by (output_col, output_row, r)
        # C[i,k] = sum_r W[k,i,r] * (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k])
        
        # So for C[i,k], the contribution from term r is W[k,i,r] * u_r(i) * v_r(k)
        # where u_r(i) = sum_j U[i,j,r] A[i,j] and v_r(k) = sum_j V[j,k,r] B[j,k]
        
        # Let me define the 7 Strassen terms:
        # Term 0: (A00+A11)(B00+B11) -> contributes to C00, C11
        # Term 1: (A10+A11)B00 -> contributes to C10
        # Term 2: A00(B01-B11) -> contributes to C01, C11
        # Term 3: A11(B10-B00) -> contributes to C00, C10
        # Term 4: (A00+A01)B11 -> contributes to C00(-1), C01
        # Term 5: (A10-A00)(B00+B01) -> contributes to... 
        # Term 6: (A01-A11)(B10+B11) -> contributes to C00, C11
        
        # Hmm, I keep getting confused. Let me just use a verified set.
        
        # Known correct Strassen 2x2x2 factors (verified in many sources):
        # I'll use the formulation where:
        # U[0,0,:] = [1, 0, 1, 0, 1, -1, 0]  (coefficients of A00)
        # U[0,1,:] = [0, 0, 0, 0, 1, 0, 1]   (coefficients of A01)
        # U[1,0,:] = [0, 1, 0, 0, 0, 1, 0]   (coefficients of A10)
        # U[1,1,:] = [1, 1, 0, 1, 0, 0, -1]  (coefficients of A11)
        
        # V[0,0,:] = [1, 1, 0, -1, 0, 1, 0]  (coefficients of B00)
        # V[0,1,:] = [0, 0, 1, 0, 0, 1, 0]   (coefficients of B01)
        # V[1,0,:] = [0, 0, 0, 1, 0, 0, 1]   (coefficients of B10)
        # V[1,1,:] = [1, 0, -1, 0, 1, 0, 1]  (coefficients of B11)
        
        # W[0,0,:] = [1, 0, 0, 1, -1, 0, 1]  (coefficients for C00)
        # W[0,1,:] = [0, 0, 1, 0, 1, 0, 0]   (coefficients for C01)
        # W[1,0,:] = [0, 1, 0, 1, 0, 0, 0]   (coefficients for C10)
        # W[1,1,:] = [1, -1, 1, 0, 0, 0, 1]  (coefficients for C11)
        
        # Let me verify: T[i,j,k] should equal 1 for all i,j,k.
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        # For i=0,j=0,k=0:
        # sum_r U[0,0,r] V[0,0,r] W[0,0,r]
        # = 1*1*1 + 0*1*0 + 1*0*0 + 0*(-1)*1 + 1*0*(-1) + (-1)*1*0 + 0*0*1
        # = 1 + 0 + 0 + 0 + 0 + 0 + 0 = 1 ✓
        
        # For i=0,j=0,k=1:
        # sum_r U[0,0,r] V[0,1,r] W[1,0,r]
        # = 1*0*0 + 0*0*1 + 1*1*0 + 0*0*1 + 1*0*0 + (-1)*1*0 + 0*0*0
        # = 0 + 0 + 0 + 0 + 0 + 0 + 0 = 0 ✗ (should be 1)
        
        # That's wrong. The indices must be different. Let me reconsider.
        
        # The tensor is T[i,j,k] where C[i,k] = sum_j A[i,j] B[j,k]
        # So T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor).
        # Decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        # Let me just verify Strassen works by checking the matrix multiplication property
        # rather than the tensor equation directly.
        
        # Actually, let me take a completely different approach.
        # I'll construct the rank-88 decomposition by direct block combination
        # and verify it works for matrix multiplication, not the tensor equation.
        
        return None  # Placeholder
    
    def build_rank88():
        """Build the rank-88 decomposition using Strassen on 2x4x8 blocks.
        
        3x4x8 = 1x4x8 (schoolbook, rank 32) + 2x4x8 (Strassen, rank 56)
        
        For 2x4x8:
        - Split into 2x2x4 and 2x2x4 blocks
        - Each 2x2x4 uses Strassen 2x2x2 twice: rank 7+7=14
        - Total for 2x4x8: 28+28=56
        
        Actually, 2x4x8 can be seen as 4 blocks of 2x2x2:
        A = [A11 A12; A21 A22] where each Aij is 2x2
        B = [B11 B12 B13 B14] where each Bij is 2x2
        C = [C11 C12 C13 C14; C21 C22 C23 C24] where each Cij is 2x2
        
        C11 = A11*B11 + A12*B21 -> 2 Strassen blocks: 14
        C12 = A11*B12 + A12*B22 -> 14
        C13 = A11*B13 + A12*B23 -> 14
        C14 = A11*B14 + A12*B24 -> 14
        C21 = A21*B11 + A22*B21 -> 14
        C22 = A21*B12 + A22*B22 -> 14
        C23 = A21*B13 + A22*B23 -> 14
        C24 = A21*B14 + A22*B24 -> 14
        Total: 8*14 = 112. That's worse than schoolbook 64!
        
        Wait, that's not right. Strassen for 2x2x2 has rank 7, not 14.
        For 2x2x4: A(2x2) * B(2x4) = C(2x4)
        This is 2 blocks of 2x2x2: rank 7+7 = 14.
        For 2x4x8: A(2x4) * B(4x8) = C(2x8)
        Split A into [A1; A2] each 2x2, B into 4 blocks each 2x2.
        C1 = A1*B1 + A2*B2 (where B1 is first 2 cols of B, etc.)
        Actually, let me think more carefully.
        
        A is 2x4, B is 4x8.
        A = [A_left | A_right] where A_left, A_right are 2x2
        B = [B_top; B_bottom] where B_top, B_bottom are 4x4
        No, that doesn't work either.
        
        Better: A(2x4) * B(4x8)
        Split the middle dimension (4) into two parts of 2:
        A = [A1 A2] where A1, A2 are 2x2
        B = [B1; B2] where B1, B2 are 2x8
        C = A1*B1 + A2*B2
        
        A1*B1: 2x2 * 2x8 = 2x8. This is 4 blocks of 2x2x2. Rank 4*7=28.
        A2*B2: 2x2 * 2x8 = 2x8. Rank 4*7=28.
        Total for 2x4x8: 56. ✓
        
        So 3x4x8 = 1x4x8 (rank 32) + 2x4x8 (rank 56) = 88. ✓
        """
        n, m, p = 3, 4, 8
        rank = 88
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Part 1: 1x4x8 schoolbook (rows 0 of A, all of B)
        # A[0, :] * B = C[0, :]
        # Schoolbook: for each j,k: C[0,k] += A[0,j]*B[j,k]
        # U[0*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+0, r] = 1
        idx = 0
        for j in range(4):
            for k in range(8):
                U[0*4 + j, idx] = 1  # A[0,j]
                V[j*8 + k, idx] = 1  # B[j,k]
                W[k*3 + 0, idx] = 1  # C[0,k]
                idx += 1
        # idx should be 32
        
        # Part 2: 2x4x8 using Strassen
        # A[1:3, :] * B = C[1:3, :]
        # Split middle dim: A[1:3, 0:2] * B[0:2, :] + A[1:3, 2:4] * B[2:4, :]
        
        # Sub-part 2a: A[1:3, 0:2] * B[0:2, :]
        # A_sub is 2x2 (rows 1,2 of A; cols 0,1)
        # B_sub is 2x8 (rows 0,1 of B)
        # C_sub is 2x8 (rows 1,2 of C)
        # This is 4 blocks of 2x2x2 (for k=0,1,2,3 of the 8 output cols)
        # Wait, B_sub is 2x8, so we split into 4 blocks of 2x2.
        
        # For each output column pair (k0, k0+1) in B[0:2, k0:k0+2]:
        # C_sub[:, k0:k0+2] = A_sub @ B_sub[:, k0:k0+2]
        # This is a 2x2x2 matmul for each k0 in {0, 2, 4, 6}
        
        for k0 in [0, 2, 4, 6]:
            # Strassen 2x2x2 for this block
            # A_block = A[1:3, 0:2] (2x2)
            # B_block = B[0:2, k0:k0+2] (2x2)
            # C_block = C[1:3, k0:k0+2] (2x2)
            
            # Strassen factors (verified):
            # I'll use the standard Strassen and verify numerically.
            # For now, use schoolbook for this block (rank 4) and verify.
            # Later optimize to Strassen (rank 7).
            
            # Actually, let me just use schoolbook for everything first to get a
            # working rank-96, then optimize.
            
            for ii in range(2):  # A row: 1+ii
                for jj in range(2):  # A col: jj (0 or 1)
                    for kk in range(2):  # B col: k0+kk
                        # U: A[1+ii, jj]
                        U[(1+ii)*4 + jj, idx] = 1
                        # V: B[jj, k0+kk]
                        V[jj*8 + (k0+kk), idx] = 1
                        # W: C[1+ii, k0+kk]
                        W[(k0+kk)*3 + (1+ii), idx] = 1
                        idx += 1
        
        # Sub-part 2b: A[1:3, 2:4] * B[2:4, :]
        for k0 in [0, 2, 4, 6]:
            for ii in range(2):  # A row: 1+ii
                for jj in range(2):  # A col: 2+jj
                    for kk in range(2):  # B col: k0+kk
                        U[(1+ii)*4 + (2+jj), idx] = 1
                        V[(2+jj)*8 + (k0+kk), idx] = 1
                        W[(k0+kk)*3 + (1+ii), idx] = 1
                        idx += 1
        
        # idx should be 32 + 32 + 32 = 96? No, that's schoolbook.
        # 1x4x8: 32
        # 2x4x8 schoolbook: 2*4*8 = 64
        # Total: 96. That's just schoolbook!
        
        # I need to actually use Strassen to get rank 56 for the 2x4x8 part.
        # Let me implement proper Strassen.
        
        return None  # Need to redo with actual Strassen
    
    # Let me implement this properly from scratch with verified Strassen.
    
    def strassen_222_factors():
        """Return (U, V, W) for 2x2x2 Strassen, shapes (2,2,7) each.
        Verified: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k in {0,1}."""
        
        # I'll use the well-known Strassen decomposition and verify it.
        # Reference: standard Strassen algorithm
        # 
        # Let me use a different, verified representation.
        # The key insight: for the all-ones tensor T[i,j,k]=1,
        # we need T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        #
        # One known rank-7 solution (from Bini et al.):
        
        U = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1]
        ])  # U[i,j,r], i=0,1; j=0,1
        
        V = np.array([
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 1]
        ])  # V[j,k,r], j=0,1; k=0,1
        
        W = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0]
        ])  # W[k,i,r], k=0,1; i=0,1
        
        # Verify: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = sum(U[i,j,r]*V[j,k,r]*W[k,i,r] for r in range(7))
                    if abs(val - 1) > 1e-10:
                        # This set is wrong, try another
                        pass
        
        # Let me try a different known set.
        # Actually, let me just construct it by solving the linear system.
        # T[i,j,k] = 1 = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Fix U and V to be random, solve for W.
        
        rng = np.random.default_rng(42)
        U = rng.standard_normal((2, 2, 7))
        V = rng.standard_normal((2, 2, 7))
        
        # For each (i,j,k): 1 = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # 8 equations, 14 unknowns (W[0,0,0..6], W[0,1,0..6], W[1,0,0..6], W[1,1,0..6])
        # Under-determined, use least squares.
        
        A_mat = np.zeros((8, 14))
        b_vec = np.ones(8)
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    eq = i*4 + j*2 + k
                    for r in range(7):
                        # W[k,i,r] is at index k*2*7 + i*7 + r = k*14 + i*7 + r
                        col = k*14 + i*7 + r
                        A_mat[eq, col] = U[i,j,r] * V[j,k,r]
        
        W_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        W = W_flat.reshape(2, 2, 7)
        
        # Verify
        ok = True
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = sum(U[i,j,r]*V[j,k,r]*W[k,i,r] for r in range(7))
                    if abs(val - 1) > 1e-8:
                        ok = False
        
        if not ok:
            # Try with different random seeds
            for seed in range(100):
                rng = np.random.default_rng(seed)
                U = rng.standard_normal((2, 2, 7))
                V = rng.standard_normal((2, 2, 7))
                A_mat = np.zeros((8, 14))
                b_vec = np.ones(8)
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            eq = i*4 + j*2 + k
                            for r in range(7):
                                col = k*14 + i*7 + r
                                A_mat[eq, col] = U[i,j,r] * V[j,k,r]
                W_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W = W_flat.reshape(2, 2, 7)
                ok = True
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            val = sum(U[i,j,r]*V[j,k,r]*W[k,i,r] for r in range(7))
                            if abs(val - 1) > 1e-8:
                                ok = False
                                break
                        if not ok: break
                    if not ok: break
                if ok:
                    break
        
        if not ok:
            # Fallback: use schoolbook rank 8 for 2x2x2
            U = np.zeros((2, 2, 8))
            V = np.zeros((2, 2, 8))
            W = np.zeros((2, 2, 8))
            idx = 0
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        U[i,j,idx] = 1
                        V[j,k,idx] = 1
                        W[k,i,idx] = 1
                        idx += 1
            return U, V, W, 8
        
        return U, V, W, 7
    
    # Get Strassen factors
    U_s, V_s, W_s, strassen_rank = strassen_222_factors()
    
    # Build the full 3x4x8 decomposition
    n, m, p = 3, 4, 8
    
    if strassen_rank == 7:
        # 2x4x8 using Strassen: rank 56
        # 1x4x8 schoolbook: rank 32
        # Total: 88
        total_rank = 32 + 56
    else:
        # Fallback to schoolbook
        total_rank = 96
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    idx = 0
    
    # Part 1: 1x4x8 schoolbook
    for j in range(4):
        for k in range(8):
            U[0*4 + j, idx] = 1
            V[j*8 + k, idx] = 1
            W[k*3 + 0, idx] = 1
            idx += 1
    
    if strassen_rank == 7:
        # Part 2: 2x4x8 using Strassen
        # A[1:3, :] * B = C[1:3, :]
        # Split middle: A[1:3,0:2]*B[0:2,:] + A[1:3,2:4]*B[2:4,:]
        # Each is 2x2 * 2x8 = 2x8, which is 4 blocks of 2x2x2
        
        for part in range(2):  # part 0: cols 0-1, part 1: cols 2-3 of A
            a_col_offset = part * 2
            b_row_offset = part * 2
            
            for k0 in range(0, 8, 2):  # Each 2x2x2 block handles 2 output cols
                # A_block: rows 1,2 of A; cols a_col_offset, a_col_offset+1
                # B_block: rows b_row_offset, b_row_offset+1 of B; cols k0, k0+1
                # C_block: rows 1,2 of C; cols k0, k0+1
                
                for r in range(7):
                    # U: A[1+ii, a_col_offset+jj]
                    for ii in range(2):
                        for jj in range(2):
                            U[(1+ii)*4 + (a_col_offset+jj), idx] = U_s[ii, jj, r]
                    
                    # V: B[b_row_offset+jj, k0+kk]
                    for jj in range(2):
                        for kk in range(2):
                            V[(b_row_offset+jj)*8 + (k0+kk), idx] = V_s[jj, kk, r]
                    
                    # W: C[1+ii, k0+kk]
                    for ii in range(2):
                        for kk in range(2):
                            W[(k0+kk)*3 + (1+ii), idx] = W_s[kk, ii, r]
                    
                    idx += 1
    else:
        # Part 2: 2x4x8 schoolbook
        for ii in range(2):
            for jj in range(4):
                for kk in range(8):
                    U[(1+ii)*4 + jj, idx] = 1
                    V[jj*8 + kk, idx] = 1
                    W[kk*3 + (1+ii), idx] = 1
                    idx += 1
    
    # Verify
    if verify_decomp(U, V, W, n, m, p, tol=1e-8):
        # Clean to integers/half-integers
        U_clean = np.round(U)
        V_clean = np.round(V)
        W_clean = np.round(W)
        
        if verify_decomp(U_clean, V_clean, W_clean, n, m, p, tol=1e-9):
            return {"U": U_clean.tolist(), "V": V_clean.tolist(), "W": W_clean.tolist()}
        
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        
        if verify_decomp(U_half, V_half, W_half, n, m, p, tol=1e-9):
            return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
        
        # If not clean, try to find a clean version by adjusting
        # For Strassen, the factors should be integers. Let me check.
        if strassen_rank == 7:
            # The Strassen factors from lstsq might not be clean.
            # Let me try to find clean Strassen factors.
            for seed in range(200):
                if time.time() > deadline - 20:
                    break
                rng = np.random.default_rng(seed)
                U_t = rng.standard_normal((2, 2, 7))
                V_t = rng.standard_normal((2, 2, 7))
                
                A_mat = np.zeros((8, 14))
                b_vec = np.ones(8)
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            eq = i*4 + j*2 + k
                            for r in range(7):
                                col = k*14 + i*7 + r
                                A_mat[eq, col] = U_t[i,j,r] * V_t[j,k,r]
                
                W_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W_t = W_flat.reshape(2, 2, 7)
                
                # Check if close to integers or half-integers
                U_t_clean = np.round(U_t)
                V_t_clean = np.round(V_t)
                W_t_clean = np.round(W_t)
                
                ok = True
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            val = sum(U_t_clean[i,j,r]*V_t_clean[j,k,r]*W_t_clean[k,i,r] for r in range(7))
                            if abs(val - 1) > 1e-9:
                                ok = False
                                break
                        if not ok: break
                    if not ok: break
                
                if ok:
                    # Rebuild with clean factors
                    U = np.zeros((n*m, 88))
                    V = np.zeros((m*p, 88))
                    W = np.zeros((p*n, 88))
                    idx = 0
                    for j in range(4):
                        for k in range(8):
                            U[0*4 + j, idx] = 1
                            V[j*8 + k, idx] = 1
                            W[k*3 + 0, idx] = 1
                            idx += 1
                    for part in range(2):
                        a_col_offset = part * 2
                        b_row_offset = part * 2
                        for k0 in range(0, 8, 2):
                            for r in range(7):
                                for ii in range(2):
                                    for jj in range(2):
                                        U[(1+ii)*4 + (a_col_offset+jj), idx] = U_t_clean[ii, jj, r]
                                for jj in range(2):
                                    for kk in range(2):
                                        V[(b_row_offset+jj)*8 + (k0+kk), idx] = V_t_clean[jj, kk, r]
                                for ii in range(2):
                                    for kk in range(2):
                                        W[(k0+kk)*3 + (1+ii), idx] = W_t_clean[kk, ii, r]
                                idx += 1
                    if verify_decomp(U, V, W, n, m, p, tol=1e-9):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 96
    U_sb = np.zeros((n*m, 96))
    V_sb = np.zeros((m*p, 96))
    W_sb = np.zeros((p*n, 96))
    idx = 0
    for i in range(3):
        for j in range(4):
            for k in range(8):
                U[i*4 + j, idx] = 1
                V[j*8 + k, idx] = 1
                W[k*3 + i, idx] = 1
                idx += 1
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

if __name__ == "__main__":
    res = construct()
    rank = len(res['U'][0])
    print(f"Found rank: {rank}")
