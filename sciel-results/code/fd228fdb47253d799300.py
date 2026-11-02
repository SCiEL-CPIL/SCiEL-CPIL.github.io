import numpy as np
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-7 exact tensor decomposition for 2x2 matrix multiplication (Strassen).
    
    This implements Strassen's algorithm, which is rank-minimal for 2x2
    matrix multiplication over the reals (proven lower bound is 7).
    
    Indexing convention:
      U has shape (n*m, rank) = (4, 7); row index = i*m + j  (A[i,j])
      V has shape (m*p, rank) = (4, 7); row index = j*p + k  (B[j,k])
      W has shape (p*n, rank) = (4, 7); row index = k*n + i  (C[i,k], trace convention)
    
    The tensor identity: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where T[i,j,k] = 1 for all i,j,k in {0,1}^3.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    rank = 7
    
    # Strassen's 7 intermediate products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    
    # Output reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices:
    # U rows (A): 0=A00, 1=A01, 2=A10, 3=A11
    # V rows (B): 0=B00, 1=B01, 2=B10, 3=B11
    # W rows (C, trace convention k*n+i):
    #   C00 (i=0,k=0) -> 0
    #   C10 (i=1,k=0) -> 1
    #   C01 (i=0,k=1) -> 2
    #   C11 (i=1,k=1) -> 3
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Col 0: M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Col 1: M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # Col 2: M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # Col 3: M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # Col 4: M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # Col 5: M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Col 6: M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    # Vectorized Verification
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want T to be all 1s.
    
    # Create index arrays for all i, j, k
    i_vals, j_vals, k_vals = np.meshgrid(range(n), range(m), range(p), indexing='ij')
    
    # Flatten to 1D arrays of indices
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    # Calculate indices for U, V, W
    u_idx = i_flat * m + j_flat
    v_idx = j_flat * p + k_flat
    w_idx = k_flat * n + i_flat
    
    # Extract rows
    U_rows = U[u_idx, :]  # shape (8, 7)
    V_rows = V[v_idx, :]  # shape (8, 7)
    W_rows = W[w_idx, :]  # shape (8, 7)
    
    # Compute sum over r (axis=1)
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # This is element-wise product of the three rows, then sum along the rank dimension
    products = U_rows * V_rows * W_rows  # shape (8, 7)
    T_values = products.sum(axis=1)     # shape (8,)
    
    # Check if all are 1 (exact integer check)
    if not np.all(T_values == 1):
        # Fallback to schoolbook rank 8 if verification fails (should not happen)
        rank = 8
        U = np.zeros((4, 8), dtype=int)
        V = np.zeros((4, 8), dtype=int)
        W = np.zeros((4, 8), dtype=int)
        # C00 = A00B00 + A01B10 -> r=0: A00B00, r=1: A01B10
        # C01 = A00B01 + A01B11 -> r=2: A00B01, r=3: A01B11
        # C10 = A10B00 + A11B10 -> r=4: A10B00, r=5: A11B10
        # C11 = A10B01 + A11B11 -> r=6: A10B01, r=7: A11B11
        
        # r=0
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        # r=1
        U[1,1]=1; V[2,1]=1; W[0,1]=1
        # r=2
        U[0,2]=1; V[1,2]=1; W[2,2]=1
        # r=3
        U[1,3]=1; V[3,3]=1; W[2,3]=1
        # r=4
        U[2,4]=1; V[0,4]=1; W[1,4]=1
        # r=5
        U[3,5]=1; V[2,5]=1; W[1,5]=1
        # r=6
        U[2,6]=1; V[1,6]=1; W[3,6]=1
        # r=7
        U[3,7]=1; V[3,7]=1; W[3,7]=1
        
    # Convert to list of lists of integers
    def convert_to_int_list(mat):
        return mat.tolist()

    U_out = convert_to_int_list(U)
    V_out = convert_to_int_list(V)
    W_out = convert_to_int_list(W)

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
