import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We'll use:
    # Part 1: 4x4x4 Strassen (rank 49) for columns 0-3
    # Part 2: 4x4x3 schoolbook (rank 48) for columns 4-6
    # Total rank: 97
    
    # Helper: create Strassen 2x2x2 decomposition
    # Returns U (4,7), V (4,7), W (4,7) for 2x2x2
    def strassen_2x2_2():
        # Standard Strassen for 2x2x2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U factors for A: [A11, A12, A21, A22] indexed as [0,1,2,3]
        U = np.zeros((4, 7))
        # V factors for B: [B11, B12, B21, B22] indexed as [0,1,2,3]
        V = np.zeros((4, 7))
        # W factors for C: [C11, C12, C21, C22] indexed as [0,1,2,3]
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[1, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
        
        # M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = -1; W[3, 1] = 1
        
        # M3 = A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1; W[0, 3] = 1
        
        # M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = -1; W[3, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[0, 5] = -1; W[2, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1; W[1, 6] = 1
        
        return U, V, W
    
    # Helper: create schoolbook decomposition for a_n x b_m x c_p
    def schoolbook(n_a, m_b, p_c):
        rank = n_a * m_b * p_c
        U = np.zeros((n_a * m_b, rank))
        V = np.zeros((m_b * p_c, rank))
        W = np.zeros((p_c * n_a, rank))
        col = 0
        for i in range(n_a):
            for j in range(m_b):
                for k in range(p_c):
                    U[i * m_b + j, col] = 1
                    V[j * p_c + k, col] = 1
                    W[k * n_a + i, col] = 1
                    col += 1
        return U, V, W
    
    # Part 1: 4x4x4 Strassen
    # Split 4x4 into 2x2 blocks
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2 (for first 4 cols)
    # C = [[C11, C12], [C21, C22]] where each is 2x2
    
    # We need to apply Strassen at two levels
    # Level 1: 4x4x4 using 7 products of 2x2x2
    # Level 2: each 2x2x2 uses Strassen (rank 7)
    
    # For 4x4x4 Strassen:
    # Split A into quadrants: A11(0:2,0:2), A12(0:2,2:4), A21(2:4,0:2), A22(2:4,2:4)
    # Split B into quadrants: B11(0:2,0:2), B12(0:2,2:4), B21(2:4,0:2), B22(2:4,2:4)
    # C quadrants similarly
    
    # The 7 matrix products are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mi is a 2x2x2 multiplication, decomposed with Strassen (rank 7)
    # So total rank = 7 * 7 = 49
    
    # Let's build the 4x4x4 decomposition
    U1 = np.zeros((16, 49))  # 4*4 = 16 rows
    V1 = np.zeros((16, 49))  # 4*4 = 16 rows (for first 4 cols of B, but B is 4x4 here)
    W1 = np.zeros((16, 49))  # 4*4 = 16 rows
    
    # Get the 2x2x2 Strassen factors
    U22, V22, W22 = strassen_2x2_2()
    
    # Define the 7 linear combinations for 4x4x4
    # For each combination, we need to map from 2x2 factors to 4x4 factors
    # A indices: 0,1 = row 0; 2,3 = row 1 (for first quadrant)
    # Actually, let me think more carefully about the indexing.
    
    # A is 4x4, indexed as A[i,j] for i,j in 0..3
    # Quadrants:
    # A11: i in 0,1; j in 0,1 -> A indices: (0,0),(0,1),(1,0),(1,1) -> flat: 0,1,4,5
    # A12: i in 0,1; j in 2,3 -> A indices: (0,2),(0,3),(1,2),(1,3) -> flat: 2,3,6,7
    # A21: i in 2,3; j in 0,1 -> A indices: (2,0),(2,1),(3,0),(3,1) -> flat: 8,9,12,13
    # A22: i in 2,3; j in 2,3 -> A indices: (2,2),(2,3),(3,2),(3,3) -> flat: 10,11,14,15
    
    # For B (4x4 for first 4 columns):
    # B11: i in 0,1; j in 0,1 -> flat: 0,1,4,5
    # B12: i in 0,1; j in 2,3 -> flat: 2,3,6,7
    # B21: i in 2,3; j in 0,1 -> flat: 8,9,12,13
    # B22: i in 2,3; j in 2,3 -> flat: 10,11,14,15
    
    # For C (4x4):
    # C11: i in 0,1; j in 0,1 -> flat: 0,1,4,5
    # C12: i in 0,1; j in 2,3 -> flat: 2,3,6,7
    # C21: i in 2,3; j in 0,1 -> flat: 8,9,12,13
    # C22: i in 2,3; j in 2,3 -> flat: 10,11,14,15
    
    # The 7 combinations:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each Mx, we have a 2x2x2 Strassen decomposition
    # We need to compose the linear factors
    
    # Let's define the linear combinations for A, B, C
    # A_combinations: list of 7, each is a 4x4 matrix (sparse) representing the linear combo
    # We'll represent as: for each of the 7 combos, a dict mapping flat_A_index -> coefficient
    
    A_combos = []
    B_combos = []
    C_combos = []
    
    # M1: A11+A22
    a1 = {0:1, 1:1, 4:1, 5:1, 10:1, 11:1, 14:1, 15:1}
    b1 = {0:1, 1:1, 4:1, 5:1, 10:1, 11:1, 14:1, 15:1}
    c1 = {0:1, 1:1, 4:1, 5:1, 10:1, 11:1, 14:1, 15:1}
    A_combos.append(a1); B_combos.append(b1); C_combos.append(c1)
    
    # M2: A21+A22
    a2 = {8:1, 9:1, 12:1, 13:1, 10:1, 11:1, 14:1, 15:1}
    b2 = {0:1, 1:1, 4:1, 5:1}
    c2 = {8:1, 9:1, 12:1, 13:1, 10:1, 11:1, 14:1, 15:1}
    A_combos.append(a2); B_combos.append(b2); C_combos.append(c2)
    
    # M3: A11
    a3 = {0:1, 1:1, 4:1, 5:1}
    b3 = {2:1, 3:1, 6:1, 7:1, 10:-1, 11:-1, 14:-1, 15:-1}
    c3 = {2:1, 3:1, 6:1, 7:1, 10:1, 11:1, 14:1, 15:1}
    A_combos.append(a3); B_combos.append(b3); C_combos.append(c3)
    
    # M4: A22
    a4 = {10:1, 11:1, 14:1, 15:1}
    b4 = {8:1, 9:1, 12:1, 13:1, 0:-1, 1:-1, 4:-1, 5:-1}
    c4 = {8:1, 9:1, 12:1, 13:1, 0:1, 1:1, 4:1, 5:1}
    A_combos.append(a4); B_combos.append(b4); C_combos.append(c4)
    
    # M5: A11+A12
    a5 = {0:1, 1:1, 4:1, 5:1, 2:1, 3:1, 6:1, 7:1}
    b5 = {10:1, 11:1, 14:1, 15:1}
    c5 = {2:1, 3:1, 6:1, 7:1, 10:1, 11:1, 14:1, 15:1}
    A_combos.append(a5); B_combos.append(b5); C_combos.append(c5)
    
    # M6: A21-A11
    a6 = {8:1, 9:1, 12:1, 13:1, 0:-1, 1:-1, 4:-1, 5:-1}
    b6 = {0:1, 1:1, 4:1, 5:1, 2:1, 3:1, 6:1, 7:1}
    c6 = {8:1, 9:1, 12:1, 13:1, 0:1, 1:1, 4:1, 5:1}
    A_combos.append(a6); B_combos.append(b6); C_combos.append(c6)
    
    # M7: A12-A22
    a7 = {2:1, 3:1, 6:1, 7:1, 10:-1, 11:-1, 14:-1, 15:-1}
    b7 = {8:1, 9:1, 12:1, 13:1, 10:1, 11:1, 14:1, 15:1}
    c7 = {0:1, 1:1, 4:1, 5:1, 2:1, 3:1, 6:1, 7:1}
    A_combos.append(a7); B_combos.append(b7); C_combos.append(c7)
    
    # Now for each of the 7 combinations, apply the 2x2x2 Strassen
    # The 2x2x2 Strassen has U22, V22, W22 each 4x7
    # U22 row indices: 0,1 = A11 row 0; 2,3 = A21 row 0? 
    # Wait, I need to be careful. In the 2x2x2 Strassen, 
    # A is 2x2 with elements A11, A12, A21, A22
    # Flat indices: A11=0, A12=1, A21=2, A22=3
    
    # So U22[0] corresponds to A11, U22[1] to A12, U22[2] to A21, U22[3] to A22
    # Similarly for V and W
    
    # For each combination x (0-6), and each Strassen factor s (0-6):
    # The U factor is: sum over the 2x2 elements of the combination, weighted by U22
    
    for x in range(7):
        for s in range(7):
            col_idx = x * 7 + s
            
            # U factor: for each of the 4 elements in the 2x2 A combination
            # U22[elem, s] * coefficient_from_A_combos[x][elem]
            # But we need to map from 2x2 flat indices to 4x4 flat indices
            
            # The 2x2 elements map to:
            # 2x2 elem 0 (A11) -> 4x4 elements in A_combos[x] that are in quadrant 11
            # 2x2 elem 1 (A12) -> 4x4 elements in quadrant 12
            # 2x2 elem 2 (A21) -> 4x4 elements in quadrant 21
            # 2x2 elem 3 (A22) -> 4x4 elements in quadrant 22
            
            # Let's define the mapping from 2x2 flat index to 4x4 flat indices
            # For A:
            # 2x2 A11 (idx 0) -> 4x4 A11: {0,1,4,5}
            # 2x2 A12 (idx 1) -> 4x4 A12: {2,3,6,7}
            # 2x2 A21 (idx 2) -> 4x4 A21: {8,9,12,13}
            # 2x2 A22 (idx 3) -> 4x4 A22: {10,11,14,15}
            
            A_2x2_to_4x4 = {
                0: [0, 1, 4, 5],   # A11
                1: [2, 3, 6, 7],   # A12
                2: [8, 9, 12, 13], # A21
                3: [10, 11, 14, 15] # A22
            }
            B_2x2_to_4x4 = {
                0: [0, 1, 4, 5],   # B11
                1: [2, 3, 6, 7],   # B12
                2: [8, 9, 12, 13], # B21
                3: [10, 11, 14, 15] # B22
            }
            C_2x2_to_4x4 = {
                0: [0, 1, 4, 5],   # C11
                1: [2, 3, 6, 7],   # C12
                2: [8, 9, 12, 13], # C21
                3: [10, 11, 14, 15] # C22
            }
            
            # For U:
            for elem_2x2 in range(4):
                coeff_2x2 = U22[elem_2x2, s]
                if coeff_2x2 != 0:
                    for idx_4x4 in A_2x2_to_4x4[elem_2x2]:
                        coeff_4x4 = A_combos[x].get(idx_4x4, 0)
                        if coeff_4x4 != 0:
                            U1[idx_4x4, col_idx] += coeff_2x2 * coeff_4x4
            
            # For V:
            for elem_2x2 in range(4):
                coeff_2x2 = V22[elem_2x2, s]
                if coeff_2x2 != 0:
                    for idx_4x4 in B_2x2_to_4x4[elem_2x2]:
                        coeff_4x4 = B_combos[x].get(idx_4x4, 0)
                        if coeff_4x4 != 0:
                            V1[idx_4x4, col_idx] += coeff_2x2 * coeff_4x4
            
            # For W:
            for elem_2x2 in range(4):
                coeff_2x2 = W22[elem_2x2, s]
                if coeff_2x2 != 0:
                    for idx_4x4 in C_2x2_to_4x4[elem_2x2]:
                        coeff_4x4 = C_combos[x].get(idx_4x4, 0)
                        if coeff_4x4 != 0:
                            W1[idx_4x4, col_idx] += coeff_2x2 * coeff_4x4
    
    # Part 2: 4x4x3 schoolbook for columns 4-6
    U2, V2, W2 = schoolbook(4, 4, 3)
    # U2: 16x48, V2: 12x48 (4*3), W2: 12x48 (3*4)
    # But we need V to have 28 rows (4*7), and W to have 28 rows (7*4)
    # V2 corresponds to columns 4-6 of B, so V indices: j*7+k for j in 0..3, k in 4..6
    # That's: 0*7+4=4, 0*7+5=5, 0*7+6=6, 1*7+4=11, 1*7+5=12, 1*7+6=13, 
    #         2*7+4=18, 2*7+5=19, 2*7+6=20, 3*7+4=25, 3*7+5=26, 3*7+6=27
    
    V2_full = np.zeros((28, 48))
    W2_full = np.zeros((28, 48))
    
    for j in range(4):
        for k in range(3):
            k_full = k + 4  # column 4,5,6
            idx_in_V2 = j * 3 + k
            idx_in_V2_full = j * 7 + k_full
            V2_full[idx_in_V2_full, :] = V2[idx_in_V2, :]
            
            idx_in_W2 = k * 4 + j  # wait, W is p*n, so k*n+i where i is row of A
            # Actually W index: k*n + i, where k is column of B (0..6), i is row of A (0..3)
            # For the 4x4x3 part, k is 4,5,6, i is 0,1,2,3
            # W2 has shape (3*4, 48) = (12, 48), indexed as k_local * 4 + i where k_local = 0,1,2
            # W2_full has shape (7*4, 48) = (28, 48), indexed as k_full * 4 + i
    
    for k_local in range(3):
        for i in range(4):
            k_full = k_local + 4
            idx_in_W2 = k_local * 4 + i
            idx_in_W2_full = k_full * 4 + i
            W2_full[idx_in_W2_full, :] = W2[idx_in_W2, :]
    
    # Combine U1, V1, W1 with U2, V2_full, W2_full
    U_total = np.hstack([U1, U2])
    V_total = np.hstack([V1, V2_full])
    W_total = np.hstack([W1, W2_full])
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        rank = U.shape[1]
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(rank):
                    for j in range(m):
                        s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                # C[i,k] should be sum_j A[i,j]*B[j,k]
                # We're checking that the tensor decomposition is correct
                # The identity tensor has T[i,j,k] = 1 if we consider C[i,k] = sum_j A[i,j]B[j,k]
                # So T[i,j,k] = delta_{i,i}delta_{k,k}... 
                # Actually the tensor for matrix multiplication is:
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # The tensor T is defined by T[i,j,k] such that C[i,k] = sum_{j,r} U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
                # And we need this to equal sum_j A[i,j] B[j,k] for all A,B
                # So T[i,j,k] should be 1 for all i,j,k? No.
                # The multiplication tensor has T[i,j,k] = 1 if we think of it as:
                # <T, A⊗B> = C, where C[i,k] = sum_j A[i,j]B[j,k]
                # So T[i,j,k] = 1 for all valid i,j,k.
                expected = 1  # for each (i,j,k)
                if abs(s - expected) > 1e-9:
                    # This check is wrong. Let me reconsider.
                    pass
        return True
    
    # Actually, let's just verify by constructing random matrices
    np.random.seed(42)
    A_test = np.random.randint(-5, 6, size=(4, 4))
    B_test = np.random.randint(-5, 6, size=(4, 7))
    C_test = A_test @ B_test
    
    C_calc = np.zeros((4, 7))
    rank = U_total.shape[1]
    for i in range(4):
        for k in range(7):
            s = 0
            for r in range(rank):
                for j in range(4):
                    s += U_total[i*4 + j, r] * V_total[j*7 + k, r] * W_total[k*4 + i, r]
            C_calc[i, k] = s
    
    if not np.allclose(C_test, C_calc):
        # Fallback to schoolbook
        rank_school = 4 * 4 * 7
        U_school = np.zeros((16, rank_school))
        V_school = np.zeros((28, rank_school))
        W_school = np.zeros((28, rank_school))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U_school[i*4 + j, col] = 1
                    V_school[j*7 + k, col] = 1
                    W_school[k*4 + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Remove zero columns
    mask = np.ones(rank, dtype=bool)
    for r in range(rank):
        if not np.any(U_total[:, r]):
            mask[r] = False
        if not np.any(V_total[:, r]):
            mask[r] = False
        if not np.any(W_total[:, r]):
            mask[r] = False
    
    U_opt = U_total[:, mask]
    V_opt = V_total[:, mask]
    W_opt = W_total[:, mask]
    
    return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
