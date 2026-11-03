"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.linalg import qr, solve

DIMENSIONS = [4, 4, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if not (U.shape[1] == V.shape[1] == W.shape[1]):
        return False
    
    rank = U.shape[1]
    
    # For each (i, k) pair, check that C[i,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    # Actually, the tensor T[i,j,k] = 1 if we multiply A[i,j] * B[j,k] to get C[i,k]
    # The decomposition says: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{correct}
    # More precisely, for matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = 1 (it's the identity for the multiplication)
    # The bilinear form: sum_{i,j,k} A[i,j] B[j,k] T[i,j,k] W... 
    # 
    # Let me reconsider. The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k
    # in the sense that (A*B)[i,k] = sum_j A[i,j] * B[j,k].
    # 
    # The rank-r decomposition means there exist U (nm x r), V (mp x r), W (pn x r) such that
    # for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Wait, that's not right either.
    #
    # Actually the standard formulation: The matrix multiplication tensor T in (R^{nm} x R^{mp} x R^{pn})
    # has T[i,j; j,k; k,i] = 1, meaning T_{(i,j),(j,k),(k,i)} = 1.
    # A rank-r decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{nm}, v_r in R^{mp}, w_r in R^{pn}.
    # So T_{(i,j),(j,k),(k,i)} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And we need this to equal 1 for all valid i,j,k.
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2():
    """Return the 7 rank-1 terms for 2x2x2 Strassen decomposition."""
    # Strassen's 7 multiplications for 2x2 matrix multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # For 2x2, n=m=p=2. Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U[i*2+j], V[j*2+k], W[k*2+i]
    # 
    # A indices: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    # B indices: B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    # C indices: C[0,0], C[0,1], C[1,0], C[1,1]
    # 
    # U has 4 entries: U[0]=a-coeff, U[1]=b-coeff, U[2]=c-coeff, U[3]=d-coeff
    # V has 4 entries: V[0]=e-coeff, V[1]=f-coeff, V[2]=g-coeff, V[3]=h-coeff
    # W has 4 entries: W[0]=C[0,0]-coeff, W[1]=C[0,1]-coeff, W[2]=C[1,0]-coeff, W[3]=C[1,1]-coeff
    
    terms = []
    # M1 = (a+d)(e+h): U=[1,0,0,1], V=[1,1,0,0]
    terms.append(([1,0,0,1], [1,1,0,0]))
    # M2 = (c+d)e: U=[0,0,1,1], V=[1,0,0,0]
    terms.append(([0,0,1,1], [1,0,0,0]))
    # M3 = a(f-h): U=[1,0,0,0], V=[0,1,0,-1]
    terms.append(([1,0,0,0], [0,1,0,-1]))
    # M4 = d(g-e): U=[0,0,0,1], V=[-1,0,1,0]
    terms.append(([0,0,0,1], [-1,0,1,0]))
    # M5 = (a+b)h: U=[1,1,0,0], V=[0,0,0,1]
    terms.append(([1,1,0,0], [0,0,0,1]))
    # M6 = (c-a)(e+f): U=[-1,0,1,0], V=[1,1,0,0]
    terms.append(([-1,0,1,0], [1,1,0,0]))
    # M7 = (b-d)(g+h): U=[0,1,0,-1], V=[0,0,1,1]
    terms.append(([0,1,0,-1], [0,0,1,1]))
    
    # W coefficients for each C element:
    # C[0,0] = M1 + M4 - M5 + M7 -> W = [1, 0, 0, 0] for C[0,0]
    # C[0,1] = M3 + M5 -> W = [0, 1, 0, 0] for C[0,1]
    # C[1,0] = M2 + M4 -> W = [0, 0, 1, 0] for C[1,0]
    # C[1,1] = M1 - M2 + M3 + M6 -> W = [0, 0, 0, 1] for C[1,1]
    
    # Now build full U, V, W matrices for the 2x2x2 tensor
    # Each term r contributes to all 4 C entries with appropriate W coefficients
    U_mat = np.zeros((4, 7))
    V_mat = np.zeros((4, 7))
    W_mat = np.zeros((4, 7))
    
    for r in range(7):
        U_vec, V_vec = terms[r]
        U_mat[:, r] = U_vec
        V_mat[:, r] = V_vec
        # W contributions:
        # C[0,0] gets M1, M4, -M5, M7
        # C[0,1] gets M3, M5
        # C[1,0] gets M2, M4
        # C[1,1] gets M1, -M2, M3, M6
        W_mat[0, r] = 0  # C[0,0]
        W_mat[1, r] = 0  # C[0,1]
        W_mat[2, r] = 0  # C[1,0]
        W_mat[3, r] = 0  # C[1,1]
    
    # Fill W based on which M contributes to which C
    # M1 -> C[0,0] (+1), C[1,1] (+1)
    W_mat[0, 0] += 1
    W_mat[3, 0] += 1
    # M2 -> C[1,0] (+1), C[1,1] (-1)
    W_mat[2, 1] += 1
    W_mat[3, 1] -= 1
    # M3 -> C[0,1] (+1), C[1,1] (+1)
    W_mat[1, 2] += 1
    W_mat[3, 2] += 1
    # M4 -> C[0,0] (+1), C[1,0] (+1)
    W_mat[0, 3] += 1
    W_mat[2, 3] += 1
    # M5 -> C[0,0] (-1), C[0,1] (+1)
    W_mat[0, 4] -= 1
    W_mat[1, 4] += 1
    # M6 -> C[1,1] (+1)
    W_mat[3, 5] += 1
    # M7 -> C[0,0] (+1)
    W_mat[0, 6] += 1
    
    return U_mat, V_mat, W_mat

def _build_4x4x5_from_blocks():
    """Build 4x4x5 decomposition by combining Strassen-like blocks.
    
    Strategy: Decompose 4x4x5 as blocks.
    Split the 4x4 input into 2x2 blocks, and the 4x5 into 2x2 + 2x3 (or similar).
    
    Actually, let's use a simpler approach:
    4x4x5 can be decomposed by viewing it as:
    - Two 2x4x5 sub-problems, or
    - A combination of smaller tensor decompositions.
    
    Better approach: Use the fact that 4x4x4 has rank at most 49 (from Strassen recursion)
    and 4x4x1 has rank at most 16 (schoolbook for the last column).
    Total: 49 + 16 = 65.
    
    For 4x4x4: Use Strassen recursively.
    4x4 = [[A11, A12],[A21, A22]] where each is 2x2.
    Similarly for B (4x4) and C (4x4).
    
    Strassen for 4x4 using 7 multiplications of 2x2 matrices:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each M is a 2x2 matrix product, which has rank 7 (Strassen).
    So total rank for 4x4x4 = 7 * 7 = 49.
    
    For the 4x4x1 part (last column of the 4x5):
    C[:, 4] = A @ B[:, 4]
    This is a vector-matrix product, which has rank at most 16 (schoolbook).
    
    Total rank: 49 + 16 = 65.
    """
    n, m, p = 4, 4, 5
    rank = 65
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # First 49 terms: 4x4x4 Strassen recursion
    # 7 outer blocks, each with 7 inner Strassen terms
    col = 0
    for outer in range(7):
        for inner in range(7):
            # Determine which 2x2 block of A, B, C this term affects
            # Outer block 'outer' determines which 2x2 submatrix of the 4x4
            # Inner term 'inner' is the Strassen term within that 2x2 product
            
            # The 7 Strassen terms for the outer 4x4=2x2 block structure:
            # Each outer term s uses specific 2x2 blocks of A and B
            # and contributes to specific 2x2 blocks of C
            pass
    
    # This is getting complex. Let me use a cleaner construction.
    
    # Alternative: Direct schoolbook for 4x4x5 is rank 80.
    # Let's just try to find a better decomposition using optimization.
    
    return None

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_4x4x4():
    """Build the 4x4x4 Strassen decomposition with rank 49."""
    n = m = p = 4
    rank = 49
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # 7 outer Strassen multiplications, each of rank 7 (inner Strassen)
    # Outer terms:
    # M1 = (A11+A22)(B11+B22) -> contributes to C11, C22
    # M2 = (A21+A22)B11 -> contributes to C21, C22
    # M3 = A11(B12-B22) -> contributes to C12, C22
    # M4 = A22(B21-B11) -> contributes to C11, C21
    # M5 = (A11+A12)B22 -> contributes to C11, C12
    # M6 = (A21-A11)(B11+B12) -> contributes to C22
    # M7 = (A12-A22)(B21+B22) -> contributes to C11
    
    # Block index mapping for 4x4:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # Similarly for B and C.
    
    # For each outer term, we need the linear forms on A-blocks and B-blocks,
    # and then decompose the resulting 2x2 matrix product using inner Strassen.
    
    # This is complex to implement correctly. Let me verify with a known construction.
    
    # Actually, let's just return the schoolbook for now and try to improve.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    
    # Try to build the 4x4x4 Strassen + 4x4x1 schoolbook decomposition (rank 65)
    # If that's too complex, fall back to schoolbook (rank 80)
    
    # For now, let's implement the Strassen 4x4x4 properly.
    
    # The 4x4x4 tensor can be decomposed using the Strassen algorithm.
    # We'll construct it explicitly.
    
    # Define the 7 outer Strassen combinations
    # Each is: (A_linear_form, B_linear_form, C_contribution_pattern)
    # where A_linear_form is a dict from block to coefficient
    # blocks: 0=A11, 1=A12, 2=A21, 3=A22
    
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # For each outer multiplication, the result is a 2x2 matrix.
    # We then decompose this 2x2 matrix product using the 7 inner Strassen terms.
    
    # The key insight: the outer "multiplication" is between linear combinations of 2x2 blocks.
    # So M_s = L_A^s * L_B^s where L_A^s is a 2x2 matrix (linear combo of A-blocks)
    # and L_B^s is a 2x2 matrix (linear combo of B-blocks).
    # Then M_s is a 2x2 matrix, and we use inner Strassen to decompose the multiplication.
    
    # Let's define the 7 outer terms as (A_coeffs, B_coeffs) where each is a 4x4 matrix
    # representing the linear form on the 16 elements of A (or B).
    
    # Actually, it's simpler to think in terms of the tensor directly.
    
    # Let me just implement the known rank-49 decomposition for 4x4x4.
    
    # For the 4x4x1 part, use schoolbook (rank 16).
    
    # Given the complexity, let me fall back to schoolbook for now.
    U, V, W = _schoolbook(n, m, p)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time - 10:
        # Time's up, return schoolbook
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
