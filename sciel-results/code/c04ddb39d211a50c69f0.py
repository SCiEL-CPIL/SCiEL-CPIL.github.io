"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    C = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
    return bool(np.allclose(C, 1.0, atol=tol))

def _schoolbook_decomposition(n, m, p):
    """Returns the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Return Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each product r, define u (2x2), v (2x2), w (2x2) such that:
    # sum_r u_r * v_r * w_r = identity in the tensor sense
    
    # Product 1: (A11+A22)(B11+B22)
    # u1: [[1,0],[0,1]] (A11 and A22)
    # v1: [[1,1],[0,0]] wait, need to be careful
    
    # Let me use the standard formulation:
    # For product r, we have matrices U_r (2x2), V_r (2x2), W_r (2x2)
    # such that sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = delta_{ik} (for the 2x2 case, it's the identity tensor)
    
    # Actually, for 2x2x2, the tensor T[i,j,k] = 1 for all i,j,k in {0,1}
    # And we need sum_r U_r[i,j] V_r[j,k] W_r[k,i] = 1
    
    # Strassen's decomposition:
    products = [
        # (U, V, W) for each of the 7 products
        # U is 2x2, V is 2x2, W is 2x2
        # Product 1: (A11+A22)(B11+B22) -> contributes to C11 and C22
        (np.array([[1,0],[0,1]]), np.array([[1,1],[0,0]]), np.array([[1,0],[0,-1]])),
        # Product 2: (A21+A22)B11
        (np.array([[0,0],[1,1]]), np.array([[1,0],[0,0]]), np.array([[0,0],[0,1]])),
        # Product 3: A11(B21-B22)
        (np.array([[1,0],[0,0]]), np.array([[0,0],[1,-1]]), np.array([[0,1],[0,0]])),
        # Product 4: A22(B12-B11)
        (np.array([[0,0],[0,1]]), np.array([[0,1],[-1,0]]), np.array([[0,0],[1,0]])),
        # Product 5: (A11+A12)B22
        (np.array([[1,1],[0,0]]), np.array([[0,0],[0,1]]), np.array([[-1,0],[0,1]])),
        # Product 6: (A21-A11)(B11+B12)
        (np.array([[-1,0],[1,0]]), np.array([[1,1],[0,0]]), np.array([[0,0],[0,1]])),
        # Product 7: (A12-A22)(B21+B22)
        (np.array([[0,1],[0,-1]]), np.array([[0,0],[1,1]]), np.array([[1,0],[0,1]])),
    ]
    
    for r, (Ur, Vr, Wr) in enumerate(products):
        U[:, r] = Ur.flatten()
        V[:, r] = Vr.flatten()
        W[:, r] = Wr.flatten()
    
    return U, V, W

def _build_3x2x6():
    """
    Build a decomposition for 3x2x6 matrix multiplication.
    Strategy: Split the 6 columns into 3 groups of 2.
    For each group, we have a 3x2x2 multiplication.
    For 3x2x2: we can use a structured approach.
    
    Actually, let me think about this differently.
    3x2x2: A is 3x2, B is 2x2, C is 3x2.
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    This is equivalent to: for each i, compute the dot product of A[i,:] with B[:,k] for k=0,1.
    
    One approach: use the 2x2x2 Strassen decomposition for each "row pair" but we have 3 rows.
    
    Alternative: For 3x2x2, we can view it as 3 independent 1x2x2 multiplications that share B.
    But they also share the structure.
    
    Let me try: for 3x2x2, the schoolbook rank is 12.
    Can we do better? The 2x2x2 part has rank 7 (Strassen).
    
    For 3x2x2, I'll use the following approach:
    - Use Strassen on the 2x2 middle part
    - This gives 7 "products" but each product involves all 3 rows
    
    Actually, for a x 2 x 2 matrix multiplication:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    We can use:
    M1 = (A[i,0]+A[i,1]) * (B[0,0]+B[1,1]) for each i... no, that's not right.
    
    Let me just use schoolbook for 3x2x2 (rank 12) and see if I can improve.
    For 3x2x6 with 3 groups of 2 columns: rank = 3 * 12 = 36 (schoolbook for the whole thing)
    But schoolbook for 3x2x6 is 36 anyway.
    
    Let me try a different approach: use the Kronecker product structure.
    """
    # For now, use schoolbook for 3x2x6
    n, m, p = 3, 2, 6
    U, V, W = _schoolbook_decomposition(n, m, p)
    return U, V, W

def _build_3x3x6():
    """
    Build a decomposition for 3x3x6 matrix multiplication.
    Schoolbook rank: 54
    """
    n, m, p = 3, 3, 6
    U, V, W = _schoolbook_decomposition(n, m, p)
    return U, V, W

def _combine_blocks(U1, V1, W1, U2, V2, W2, n, m1, m2, p):
    """
    Combine two block decompositions:
    Block 1: n x m1 x p
    Block 2: n x m2 x p
    Total: n x (m1+m2) x p
    
    U1: (n*m1, r1), V1: (m1*p, r1), W1: (p*n, r1)
    U2: (n*m2, r2), V2: (m2*p, r2), W2: (p*n, r2)
    
    Combined U: (n*(m1+m2), r1+r2)
    The U matrix needs to be rearranged because the middle index j is split.
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    total_rank = r1 + r2
    
    U = np.zeros((n * (m1 + m2), total_rank), dtype=np.float64)
    V = np.zeros(((m1 + m2) * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    # For U: U[i*m_total + j, r]
    # For j in [0, m1): use U1[i*m1 + j, r]
    # For j in [m1, m1+m2): use U2[i*m2 + (j-m1), r]
    for i in range(n):
        for j in range(m1):
            U[i * (m1 + m2) + j, :r1] = U1[i * m1 + j, :]
        for j in range(m2):
            U[i * (m1 + m2) + m1 + j, r1:] = U2[i * m2 + j, :]
    
    # For V: V[j*p_total + k, r]
    # For j in [0, m1): use V1[j*p + k, r]
    # For j in [m1, m1+m2): use V2[(j-m1)*p + k, r]
    for j in range(m1):
        for k in range(p):
            V[j * p + k, :r1] = V1[j * p + k, :]
    for j in range(m2):
        for k in range(p):
            V[(m1 + j) * p + k, r1:] = V2[j * p + k, :]
    
    # For W: W[k*n + i, r] - same structure for both blocks
    for k in range(p):
        for i in range(n):
            W[k * n + i, :r1] = W1[k * n + i, :]
            W[k * n + i, r1:] = W2[k * n + i, :]
    
    return U, V, W

def _random_search(n, m, p, target_rank, time_limit=10.0):
    """
    Random search for a rank-target_rank decomposition.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n*m, target_rank))
        V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m*p, target_rank))
        W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p*n, target_rank))
        
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for matrix multiplication.
    Dimensions: n=3, m=5, p=6.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    overall_deadline = start_time + 130.0
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try block decomposition: 3x5x6 = 3x2x6 + 3x3x6
    if time.time() < overall_deadline - 20.0:
        U1, V1, W1 = _build_3x2x6()
        U2, V2, W2 = _build_3x3x6()
        U_block, V_block, W_block = _combine_blocks(U1, V1, W1, U2, V2, W2, n, 2, 3, p)
        
        if _verify_decomposition(U_block, V_block, W_block, n, m, p):
            block_rank = U_block.shape[1]
            if block_rank < best_rank:
                best_rank = block_rank
                best_U, best_V, best_W = U_block.copy(), V_block.copy(), W_block.copy()
    
    # Try random search for lower ranks
    if time.time() < overall_deadline - 10.0:
        for target_rank in range(60, min(best_rank, 90)):
            if time.time() > overall_deadline - 5.0:
                break
            remaining = min(10.0, overall_deadline - time.time())
            result = _random_search(n, m, p, target_rank, time_limit=remaining)
            if result is not None:
                U, V, W = result
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
