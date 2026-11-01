import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def _build_trivial(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1.0
                V[b_idx, r] = 1.0
                W[c_idx, r] = 1.0
                r += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is valid."""
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_target[a_idx, b_idx, c_idx] = 1.0
    return np.allclose(T_comp, T_target, atol=tol)

def _add_strassen_block(U, V, W, rank_start, i0, j0, k0):
    """
    Adds a 2x2x2 Strassen decomposition for the sub-matrix defined by 
    rows i0..i0+1, cols j0..j0+1 of A, rows j0..j0+1, cols k0..k0+1 of B,
    and rows k0..k0+1, cols i0..i0+1 of C.
    
    Strassen formulas for C = A*B where A,B,C are 2x2:
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # Indices in the global vectors
    # A is 2x4, B is 4x7, C is 7x2 (flattened)
    # A indices: (i, j) -> i*4 + j
    # B indices: (j, k) -> j*7 + k
    # C indices: (k, i) -> k*2 + i
    
    # We are decomposing the tensor slice T[i0:i0+2, j0:j0+2, k0:k0+2]
    # But the dimensions are fixed: n=2, m=4, p=7.
    # So i0 must be 0, j0 can be 0, 2, k0 can be 0, 2, 4, 6?
    # Wait, Strassen requires square blocks. 
    # If we pick i0=0, j0=0, k0=0:
    # A block is A[0:2, 0:2]
    # B block is B[0:2, 0:2]
    # C block is C[0:2, 0:2]
    
    # Let's map the local indices to global indices.
    # Local A: (li, lj) where li, lj in {0,1}
    # Global A: (i0+li, j0+lj)
    
    # The Strassen algorithm computes 7 products.
    # Each product corresponds to a rank-1 term in the decomposition.
    # For each Strassen term Mr, we have linear combinations of A and B elements.
    # U[:, r] corresponds to the linear combination of A elements.
    # V[:, r] corresponds to the linear combination of B elements.
    # W[:, r] corresponds to the linear combination of C elements.
    
    # Term 1: M1 = (A11 + A22) * (B11 + B22)
    # A part: A11, A22
    # B part: B11, B22
    # C part: contributes to C11, C22
    
    # Let's define the coefficients for each term.
    # We will add 7 new columns to U, V, W.
    
    # Helper to get global indices
    def a_idx(li, lj):
        return (i0 + li) * 4 + (j0 + lj)
    def b_idx(lj, lk):
        return (j0 + lj) * 7 + (k0 + lk)
    def c_idx(lk, li):
        return (k0 + lk) * 2 + (i0 + li)

    # Term 1
    r = rank_start
    U[a_idx(0,0), r] += 1.0
    U[a_idx(1,1), r] += 1.0
    V[b_idx(0,0), r] += 1.0
    V[b_idx(1,1), r] += 1.0
    W[c_idx(0,0), r] += 1.0  # C11
    W[c_idx(1,1), r] += 1.0  # C22
    r += 1
    
    # Term 2: M2 = (A21 + A22) * B11
    U[a_idx(1,0), r] += 1.0
    U[a_idx(1,1), r] += 1.0
    V[b_idx(0,0), r] += 1.0
    W[c_idx(0,1), r] += 1.0  # C21
    r += 1
    
    # Term 3: M3 = A11 * (B12 - B22)
    U[a_idx(0,0), r] += 1.0
    V[b_idx(0,1), r] += 1.0
    V[b_idx(1,1), r] += -1.0
    W[c_idx(0,1), r] += 1.0  # C12
    W[c_idx(1,1), r] += 1.0  # C22
    r += 1
    
    # Term 4: M4 = A22 * (B21 - B11)
    U[a_idx(1,1), r] += 1.0
    V[b_idx(1,0), r] += 1.0
    V[b_idx(0,0), r] += -1.0
    W[c_idx(0,0), r] += 1.0  # C11
    W[c_idx(1,0), r] += 1.0  # C21
    r += 1
    
    # Term 5: M5 = (A11 + A12) * B22
    U[a_idx(0,0), r] += 1.0
    U[a_idx(0,1), r] += 1.0
    V[b_idx(1,1), r] += 1.0
    W[c_idx(0,0), r] += -1.0 # C11
    W[c_idx(0,1), r] += 1.0  # C12
    r += 1
    
    # Term 6: M6 = (A21 - A11) * (B11 + B12)
    U[a_idx(1,0), r] += 1.0
    U[a_idx(0,0), r] += -1.0
    V[b_idx(0,0), r] += 1.0
    V[b_idx(0,1), r] += 1.0
    W[c_idx(1,1), r] += 1.0  # C22
    r += 1
    
    # Term 7: M7 = (A12 - A22) * (B21 + B22)
    U[a_idx(0,1), r] += 1.0
    U[a_idx(1,1), r] += -1.0
    V[b_idx(1,0), r] += 1.0
    V[b_idx(1,1), r] += 1.0
    W[c_idx(0,0), r] += 1.0  # C11
    r += 1
    
    return r

def _add_schoolbook_block(U, V, W, rank_start, i0, j0, k0):
    """
    Adds a 2x2x2 schoolbook decomposition (4 terms).
    Actually, if we just use 4 terms for a 2x2x2 block, it's rank 4.
    We can also use 1x1x1 blocks (rank 1 each).
    For a 2x2x2 block, schoolbook is 4 terms.
    """
    def a_idx(li, lj):
        return (i0 + li) * 4 + (j0 + lj)
    def b_idx(lj, lk):
        return (j0 + lj) * 7 + (k0 + lk)
    def c_idx(lk, li):
        return (k0 + lk) * 2 + (i0 + li)

    r = rank_start
    # C11 = A11*B11
    U[a_idx(0,0), r] += 1.0
    V[b_idx(0,0), r] += 1.0
    W[c_idx(0,0), r] += 1.0
    r += 1
    # C12 = A12*B22 ? No, C12 = A11*B12 + A12*B22
    # Let's just do individual 1x1x1 terms for simplicity and correctness.
    # Actually, 2x2x2 schoolbook is 4 terms:
    # C11 = A11 B11
    # C12 = A11 B12 + A12 B22 -> This is not rank 1.
    # So we must decompose 2x2x2 into four 1x1x1 blocks if we want simple rank 4.
    # Or we can just call 4 times a 1x1x1 adder.
    
    # Let's implement 1x1x1 addition directly in the main loop for remaining parts.
    # This function is a placeholder if we wanted a specific 2x2x2 schoolbook, 
    # but 4x 1x1x1 is equivalent and easier to manage in the general loop.
    return r

def _add_single_term(U, V, W, rank_start, i, j, k):
    """
    Adds a single 1x1x1 term: A[i,j] * B[j,k] -> C[k,i]
    """
    a_idx = i * 4 + j
    b_idx = j * 7 + k
    c_idx = k * 2 + i
    
    U[a_idx, rank_start] += 1.0
    V[b_idx, rank_start] += 1.0
    W[c_idx, rank_start] += 1.0
    
    return rank_start + 1

def construct():
    start_time = time.time()
    deadline = 145
    
    n, m, p = DIMENSIONS # 2, 4, 7
    
    # Strategy:
    # We have a 2x4 matrix A and 4x7 matrix B.
    # We can partition the columns of A (and rows of B) into blocks of size 2.
    # Since m=4, we have two blocks of size 2: [0:2] and [2:4].
    # Since n=2, we have one block of size 2 for rows of A and cols of C: [0:2].
    # Since p=7, we have three blocks of size 2 for cols of B and rows of C: [0:2], [2:4], [4:6], and one [6:7].
    
    # Let's try to use Strassen for the 2x2x2 blocks where possible.
    # The "2x2x2" refers to the sub-matrices involved in the multiplication.
    # If we take A block (2x2), B block (2x2), the result is a 2x2 C block.
    # This fits the Strassen algorithm perfectly.
    
    # Partition:
    # A rows: 0,1 (one block of size 2)
    # A cols / B rows: 0,1 (block 1), 2,3 (block 2)
    # B cols / C rows: 0,1 (block 1), 2,3 (block 2), 4,5 (block 3), 6 (block 4)
    
    # We can apply Strassen to:
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # 2. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] ? No, C indices are (k,i).
    #    C[0:2, 0:2] is computed from A[0:2, :] * B[:, 0:2].
    #    So C[0:2, 0:2] gets contributions from:
    #    - A[0:2, 0:2] * B[0:2, 0:2]
    #    - A[0:2, 2:4] * B[2:4, 0:2]
    
    # Let's decompose the total tensor sum.
    # Total rank = Sum of ranks of all blocks.
    
    # We will use Strassen (rank 7) for 2x2x2 blocks and Schoolbook (rank 4) for others if not square or if we prefer.
    # Actually, for 2x2x2, Strassen (7) is worse than Schoolbook (4).
    # Wait. Strassen is for reducing the number of multiplications in the algorithm.
    # In tensor decomposition rank, Schoolbook for 2x2x2 is rank 4. Strassen is rank 7.
    # So Schoolbook is BETTER (lower rank) for 2x2x2?
    # No, the trivial decomposition for 2x2x2 is rank 8.
    # Schoolbook uses 4 multiplications. Rank is 4.
    # Strassen uses 7 multiplications. Rank is 7.
    # So for 2x2x2, Schoolbook (rank 4) is better than Strassen (rank 7).
    
    # Why did I think Strassen was better?
    # Strassen reduces the *arithmetic complexity* (number of scalar multiplications) from 8 to 7.
    # But in tensor rank, the trivial decomposition of 2x2x2 matrix multiplication has rank 7?
    # No, the rank of 2x2x2 matrix multiplication is 7.
    # The trivial decomposition (schoolbook) has rank 8? No, schoolbook has 8 terms if you write out C_ij = sum A_ik B_kj.
    # Wait, C_11 = A_11 B_11 + A_12 B_21. That's 2 terms.
    # There are 4 entries in C. Each is a sum of 2 products. Total 8 products.
    # So trivial rank is 8.
    # Strassen reduces this to 7 products. So rank 7.
    # So Strassen IS better for 2x2x2. Rank 7 < 8.
    
    # For 1x1x1, rank is 1.
    # For 2x2x1 (C is 2x1), trivial rank is 4. Can we do better?
    # C_1 = A_11 B_11 + A_12 B_21
    # C_2 = A_21 B_11 + A_22 B_21
    # This is a 2x2x1 tensor. Rank is likely 4 (no known improvement for non-square or small dims).
    
    # So, we should use Strassen for all 2x2x2 blocks.
    
    # Let's identify the 2x2x2 blocks we can form.
    # A is 2x4. B is 4x7. C is 7x2.
    
    # We can partition the "middle" dimension (size 4) into two blocks of size 2: J1=[0,1], J2=[2,3].
    # We can partition the "output" dimension (size 7) into blocks: K1=[0,1], K2=[2,3], K3=[4,5], K4=[6].
    # The "input" dimension (size 2) is one block: I1=[0,1].
    
    # The total multiplication is sum over J and K.
    # C[I, K] = sum_J A[I, J] B[J, K]
    
    # We can decompose this into blocks of (I_block, J_block, K_block).
    # Since I is always size 2, and we have J blocks of size 2, and K blocks of size 2 (mostly),
    # we can form 2x2x2 blocks.
    
    # Blocks:
    # 1. I=[0:2], J=[0:2], K=[0:2] -> 2x2x2 Strassen (Rank 7)
    # 2. I=[0:2], J=[0:2], K=[2:4] -> 2x2x2 Strassen (Rank 7)
    # 3. I=[0:2], J=[0:2], K=[4:6] -> 2x2x2 Strassen (Rank 7)
    # 4. I=[0:2], J=[0:2], K=[6:7] -> 2x2x1 Schoolbook (Rank 4) -- Wait, K is size 1 here.
    #    A[0:2, 0:2] * B[0:2, 6:7] -> C[6:7, 0:2].
    #    This is a 2x2x1 block. Rank 4.
    
    # 5. I=[0:2], J=[2:4], K=[0:2] -> 2x2x2 Strassen (Rank 7)
    # 6. I=[0:2], J=[2:4], K=[2:4] -> 2x2x2 Strassen (Rank 7)
    # 7. I=[0:2], J=[2:4], K=[4:6] -> 2x2x2 Strassen (Rank 7)
    # 8. I=[0:2], J=[2:4], K=[6:7] -> 2x2x1 Schoolbook (Rank 4)
    
    # Total Rank = 7 + 7 + 7 + 4 + 7 + 7 + 7 + 4 = 50.
    # Trivial Rank = 2 * 4 * 7 = 56.
    # Improvement: 56 -> 50.
    
    # Let's implement this.
    
    # Initialize factors with sufficient size
    max_rank = 56
    U = np.zeros((n * m, max_rank), dtype=float)
    V = np.zeros((m * p, max_rank), dtype=float)
    W = np.zeros((p * n, max_rank), dtype=float)
    
    current_rank = 0
    
    # Helper to add Strassen block
    def add_strassen(i0, j0, k0):
        nonlocal current_rank
        # i0, j0, k0 are start indices. Size is 2x2x2.
        # A: [i0:i0+2, j0:j0+2]
        # B: [j0:j0+2, k0:k0+2]
        # C: [k0:k0+2, i0:i0+2]
        
        # We use the _add_strassen_block logic defined earlier, but inline for clarity and speed.
        # It adds 7 terms.
        
        # Term 1: (A11+A22)(B11+B22) -> C11, C22
        r = current_rank
        U[(i0+0)*m + (j0+0), r] += 1.0
        U[(i0+1)*m + (j0+1), r] += 1.0
        V[(j0+0)*p + (k0+0), r] += 1.0
        V[(j0+1)*p + (k0+1), r] += 1.0
        W[(k0+0)*n + (i0+0), r] += 1.0
        W[(k0+1)*n + (i0+1), r] += 1.0
        current_rank += 1
        
        # Term 2: (A21+A22)B11 -> C21
        r = current_rank
        U[(i0+1)*m + (j0+0), r] += 1.0
        U[(i0+1)*m + (j0+1), r] += 1.0
        V[(j0+0)*p + (k0+0), r] += 1.0
        W[(k0+0)*n + (i0+1), r] += 1.0
        current_rank += 1
        
        # Term 3: A11(B12-B22) -> C12, C22
        r = current_rank
        U[(i0+0)*m + (j0+0), r] += 1.0
        V[(j0+0)*p + (k0+1), r] += 1.0
        V[(j0+1)*p + (k0+1), r] += -1.0
        W[(k0+1)*n + (i0+0), r] += 1.0
        W[(k0+1)*n + (i0+1), r] += 1.0
        current_rank += 1
        
        # Term 4: A22(B21-B11) -> C11, C21
        r = current_rank
        U[(i0+1)*m + (j0+1), r] += 1.0
        V[(j0+1)*p + (k0+0), r] += 1.0
        V[(j0+0)*p + (k0+0), r] += -1.0
        W[(k0+0)*n + (i0+0), r] += 1.0
        W[(k0+0)*n + (i0+1), r] += 1.0
        current_rank += 1
        
        # Term 5: (A11+A12)B22 -> C11, C12
        r = current_rank
        U[(i0+0)*m + (j0+0), r] += 1.0
        U[(i0+0)*m + (j0+1), r] += 1.0
        V[(j0+1)*p + (k0+1), r] += 1.0
        W[(k0+0)*n + (i0+0), r] += -1.0
        W[(k0+1)*n + (i0+0), r] += 1.0
        current_rank += 1
        
        # Term 6: (A21-A11)(B11+B12) -> C22
        r = current_rank
        U[(i0+1)*m + (j0+0), r] += 1.0
        U[(i0+0)*m + (j0+0), r] += -1.0
        V[(j0+0)*p + (k0+0), r] += 1.0
        V[(j0+0)*p + (k0+1), r] += 1.0
        W[(k0+1)*n + (i0+1), r] += 1.0
        current_rank += 1
        
        # Term 7: (A12-A22)(B21+B22) -> C11
        r = current_rank
        U[(i0+0)*m + (j0+1), r] += 1.0
        U[(i0+1)*m + (j0+1), r] += -1.0
        V[(j0+1)*p + (k0+0), r] += 1.0
        V[(j0+1)*p + (k0+1), r] += 1.0
        W[(k0+0)*n + (i0+0), r] += 1.0
        current_rank += 1

    def add_schoolbook_2x2x1(i0, j0, k0):
        """
        Adds 2x2x1 block: A[2x2] * B[2x1] -> C[1x2]
        Rank 4.
        C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        C[0,1] = A[1,0]B[0,0] + A[1,1]B[1,0]
        Wait, C indices are (k, i).
        k0 is the single k index.
        i0, i0+1 are the i indices.
        j0, j0+1 are the j indices.
        
        C[k0, i0] = A[i0, j0]B[j0, k0] + A[i0, j0+1]B[j0+1, k0]
        C[k0, i0+1] = A[i0+1, j0]B[j0, k0] + A[i0+1, j0+1]B[j0+1, k0]
        
        We decompose into 4 terms.
        """
        # Term 1: A[i0, j0] * B[j0, k0] -> C[k0, i0]
        r = current_rank
        U[(i0+0)*m + (j0+0), r] += 1.0
        V[(j0+0)*p + k0, r] += 1.0
        W[k0*n + (i0+0), r] += 1.0
        current_rank += 1
        
        # Term 2: A[i0, j0+1] * B[j0+1, k0] -> C[k0, i0]
        r = current_rank
        U[(i0+0)*m + (j0+1), r] += 1.0
        V[(j0+1)*p + k0, r] += 1.0
        W[k0*n + (i0+0), r] += 1.0
        current_rank += 1
        
        # Term 3: A[i0+1, j0] * B[j0, k0] -> C[k0, i0+1]
        r = current_rank
        U[(i0+1)*m + (j0+0), r] += 1.0
        V[(j0+0)*p + k0, r] += 1.0
        W[k0*n + (i0+1), r] += 1.0
        current_rank += 1
        
        # Term 4: A[i0+1, j0+1] * B[j0+1, k0] -> C[k0, i0+1]
        r = current_rank
        U[(i0+1)*m + (j0+1), r] += 1.0
        V[(j0+1)*p + k0, r] += 1.0
        W[k0*n + (i0+1), r] += 1.0
        current_rank += 1

    # Execute the plan
    # I block is always 0:2 (i0=0)
    i0 = 0
    
    # J blocks: 0:2, 2:4
    # K blocks: 0:2, 2:4, 4:6, 6:7
    
    # J=0:2
    j0 = 0
    # K=0:2
    add_strassen(i0, j0, 0)
    # K=2:4
    add_strassen(i0, j0, 2)
    # K=4:6
    add_strassen(i0, j0, 4)
    # K=6:7
    add_schoolbook_2x2x1(i0, j0, 6)
    
    # J=2:4
    j0 = 2
    # K=0:2
    add_strassen(i0, j0, 0)
    # K=2:4
    add_strassen(i0, j0, 2)
    # K=4:6
    add_strassen(i0, j0, 4)
    # K=6:7
    add_schoolbook_2x2x1(i0, j0, 6)
    
    # Final rank is current_rank
    # Slice the arrays to the actual rank
    U_final = U[:, :current_rank]
    V_final = V[:, :current_rank]
    W_final = W[:, :current_rank]
    
    # Verify
    if _verify_decomposition(U_final, V_final, W_final, n, m, p):
        return {
            "U": U_final.tolist(),
            "V": V_final.tolist(),
            "W": W_final.tolist()
        }
    else:
        # Fallback to trivial if something went wrong (should not happen)
        U_triv, V_triv, W_triv = _build_trivial(n, m, p)
        return {
            "U": U_triv.tolist(),
            "V": V_triv.tolist(),
            "W": W_triv.tolist()
        }

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i*m + j, j*p + k, k*n + i] = 1
    
    if np.allclose(T, Expected, atol=1e-9):
        print(f"Valid decomposition with rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
        diff = np.abs(T - Expected)
        print(f"Max diff: {diff.max()}")
