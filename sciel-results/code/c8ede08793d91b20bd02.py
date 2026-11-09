"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import scipy.sparse as sp

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Build 2x2 Strassen decomposition
    # For 2x2 matrices: A * B = C
    # C11 = (A11+A22)*(B11+B22) - C21
    # C12 = A11*(B12-B22) + C22
    # C21 = (A22-A11)*(B11+B12) + C11
    # C22 = (A11+A22)*(B22-B11) - C12
    # Where C11, C12, C21, C22 are the 4 quadrants of C
    # And the 7 products are:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A11+A22)*(B22-B11)
    # M3 = (A22-A11)*(B11+B12)
    # M4 = A11*(B12-B22)
    # M5 = A22*(B11-B12)
    # M6 = (A11+A12)*B22
    # M7 = (A21+A22)*B11
    
    # For 2x2 Strassen:
    # C11 = M1 - M2 + M4 - M5 + M7
    # C12 = M3 - M4 + M5 - M6
    # C21 = M1 - M3 + M6 - M7
    # C22 = M2 + M3 - M4 + M5
    
    # Actually let me use the standard Strassen formulas:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = (A11 + A12) * B22
    # M4 = A11 * (B12 - B22)
    # M5 = A22 * (B21 - B11)
    # M6 = (A11 + A12) * (B21 + B22)
    # M7 = (A21 + A22) * (B11 + B12)
    
    # C11 = M1 - M6 + M7 + M4
    # C12 = M3 + M6 - M7 - M5
    # C21 = M2 + M5 - M3 - M4
    # C22 = M1 - M2 + M5 + M7
    
    # Build the 3x3 schoolbook decomposition
    def build_schoolbook_3x3():
        """Build 3x3 schoolbook decomposition"""
        rank = 27
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
        
        return U, V, W
    
    # Build 2x2 Strassen decomposition
    def build_strassen_2x2():
        """Build 2x2 Strassen decomposition"""
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # M1 = (A11 + A22) * (B11 + B22)
        # A indices: 0=A11, 1=A12, 2=A21, 3=A22
        # B indices: 0=B11, 1=B12, 2=B21, 3=B22
        # C indices: 0=C11, 1=C12, 2=C21, 3=C22
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = -1
        
        # M2 = (A21 + A22) * B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3 = (A11 + A12) * B22
        U[0, 2] = 1; U[1, 2] = 1
        V[3, 2] = 1
        W[1, 2] = 1; W[2, 2] = -1
        
        # M4 = A11 * (B12 - B22)
        U[0, 3] = 1
        V[1, 3] = 1; V[3, 3] = -1
        W[0, 3] = 1; W[2, 3] = -1
        
        # M5 = A22 * (B21 - B11)
        U[3, 4] = 1
        V[2, 4] = 1; V[0, 4] = -1
        W[1, 4] = -1; W[3, 4] = 1
        
        # M6 = (A11 + A12) * (B21 + B22)
        U[0, 5] = 1; U[1, 5] = 1
        V[2, 5] = 1; V[3, 5] = 1
        W[0, 5] = -1; W[1, 5] = 1
        
        # M7 = (A21 + A22) * (B11 + B12)
        U[2, 6] = 1; U[3, 6] = 1
        V[0, 6] = 1; V[1, 6] = 1
        W[0, 6] = 1; W[2, 6] = -1
        
        return U, V, W
    
    # Verify Strassen decomposition
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W correctly decompose matrix multiplication"""
        # For each (i,j,k), sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] should be 1
        # Otherwise 0
        
        # Use tensor contraction
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        
        # Create sparse tensors
        U_sp = sp.csr_matrix(U)
        V_sp = sp.csr_matrix(V)
        W_sp = sp.csr_matrix(W)
        
        rank = U.shape[1]
        
        # Build the full tensor
        T = np.zeros((n, m, p))
        
        for c in range(rank):
            u_col = U[:, c]
            v_col = V[:, c]
            w_col = W[:, c]
            
            # Only process non-zero entries
            u_nz = np.nonzero(u_col)[0]
            v_nz = np.nonzero(v_col)[0]
            w_nz = np.nonzero(w_col)[0]
            
            if len(u_nz) == 0 or len(v_nz) == 0 or len(w_nz) == 0:
                continue
            
            for idx_u in u_nz:
                i = idx_u // m
                j = idx_u % m
                for idx_v in v_nz:
                    j2 = idx_v // p
                    k = idx_v % p
                    if j2 != j:
                        continue
                    for idx_w in w_nz:
                        k2 = idx_w // n
                        i2 = idx_w % n
                        if k2 != k or i2 != i:
                            continue
                        T[i, j, k] += u_col[idx_u] * v_col[idx_v] * w_col[idx_w]
        
        # Check that T[i,j,k] = 1 if i==j==k (wait, no)
        # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor should be T[i,j,k] = delta(i,j,k) where the multiplication is defined
        # T[i,j,k] = 1 if this is a valid term in the product, else 0
        # For matrix multiplication, every (i,j,k) triple contributes, so T[i,j,k] = 1 for all i,j,k
        
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-10)
    
    # Kronecker product of decompositions
    def kronecker_product(U1, V1, W1, U2, V2, W2):
        """Kronecker product of two tensor decompositions"""
        rank1, rank2 = U1.shape[1], U2.shape[1]
        rank = rank1 * rank2
        
        U = np.zeros((U1.shape[0] * U2.shape[0], rank))
        V = np.zeros((V1.shape[0] * V2.shape[0], rank))
        W = np.zeros((W1.shape[0] * W2.shape[0], rank))
        
        for c1 in range(rank1):
            for c2 in range(rank2):
                c = c1 * rank2 + c2
                U[:, c] = np.kron(U1[:, c1], U2[:, c2])
                V[:, c] = np.kron(V1[:, c1], V2[:, c2])
                W[:, c] = np.kron(W1[:, c1], W2[:, c2])
        
        return U, V, W
    
    # Build 3x3 schoolbook and 2x2 Strassen
    U3, V3, W3 = build_schoolbook_3x3()
    U2, V2, W2 = build_strassen_2x2()
    
    # Verify individual decompositions
    if not verify_decomposition(U3, V3, W3, 3, 3, 3):
        print("Warning: 3x3 schoolbook verification failed")
    if not verify_decomposition(U2, V2, W2, 2, 2, 2):
        print("Warning: 2x2 Strassen verification failed")
    
    # Kronecker product: 6x6x6 = (3x3) ⊗ (2x2)
    U6, V6, W6 = kronecker_product(U3, V3, W3, U2, V2, W2)
    
    # Verify the 6x6x6 decomposition
    if not verify_decomposition(U6, V6, W6, 6, 6, 6):
        print("Warning: 6x6x6 Kronecker decomposition verification failed")
        # Fallback to schoolbook
        U6, V6, W6 = build_schoolbook_3x3()
        # Scale up to 6x6x6
        rank_sb = 216
        U6 = np.zeros((36, rank_sb))
        V6 = np.zeros((36, rank_sb))
        W6 = np.zeros((36, rank_sb))
        col = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    U6[i*6+j, col] = 1
                    V6[j*6+k, col] = 1
                    W6[k*6+i, col] = 1
                    col += 1
    
    # Convert to lists of lists with integer/half-integer entries
    def to_list_of_lists(matrix):
        """Convert numpy array to list of lists, rounding to nearest half-integer"""
        result = []
        for row in matrix:
            row_list = []
            for val in row:
                # Round to nearest 0.5
                rounded = round(val * 2) / 2
                # If it's very close to an integer, make it an int
                if abs(rounded - round(rounded)) < 1e-10:
                    row_list.append(int(round(rounded)))
                else:
                    row_list.append(rounded)
            result.append(row_list)
        return result
    
    U_list = to_list_of_lists(U6)
    V_list = to_list_of_lists(V6)
    W_list = to_list_of_lists(W6)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
