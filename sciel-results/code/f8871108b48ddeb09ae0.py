"""Matrix multiplication 2x3x4 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-16 decomposition of the 2x3x4 matrix multiplication tensor.
    
    U shape: (n*m, rank) = (6, 16)
    V shape: (m*p, rank) = (12, 16)  
    W shape: (p*n, rank) = (8, 16)
    
    Index conventions:
    U[i*m + j, r] where m=3
    V[j*p + k, r] where p=4
    W[k*n + i, r] where n=2
    """
    n, m, p = 2, 3, 4
    rank = 16
    
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 16
    V = [[0.0] * rank for _ in range(m * p)]  # 12 x 16
    W = [[0.0] * rank for _ in range(p * n)]  # 8 x 16
    
    # We construct the decomposition using the fact that the 2x3x4 MM tensor
    # can be decomposed as the sum of:
    # - Two 2x2x2 MM tensors (for j in {0,1} x k in {0,1} and j in {0,1} x k in {2,3})
    #   Each has tensor rank 7 (Strassen)
    # - One 2x1x4 tensor (for j=2) with tensor rank 2
    # Total: 7 + 7 + 2 = 16
    
    # The 2x2x2 MM tensor T(a,b,c) = 1 for a,b,c in {0,1} has the following
    # rank-7 decomposition (Strassen). We use the standard Strassen coefficients
    # expressed as rank-1 tensors in the (a,b,c) space.
    
    # For a 2x2x2 MM tensor, the 7 rank-1 terms are:
    # term r has U_r[a,b], V_r[b,c], W_r[c,a]
    
    # Using the explicit Strassen tensor decomposition:
    # We need sum_r U_r[a,b]*V_r[b,c]*W_r[c,a] = 1 for all a,b,c in {0,1}
    
    # Here are the 7 terms (U_r, V_r, W_r) as 2x2 matrices:
    strassen_terms = [
        # (U, V, W) where each is a 2x2 matrix (list of 4 entries: [00, 01, 10, 11])
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [1, 0, 0, 0]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]),
        # M6
        ([1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),
    ]
    
    def add_strassen_block(u_row_offset, v_row_offset, w_row_offset, col_offset):
        """Add a Strassen block to the global matrices.
        u_row_offset: maps local (a,b) -> global row, a in {0,1}, b in {0,1}
        """
        for r in range(7):
            U_mat, V_mat, W_mat = strassen_terms[r]
            for a in range(2):
                for b in range(2):
                    u_val = U_mat[a*2 + b]
                    if u_val != 0:
                        # Local (a,b) -> global: a is i, b is j
                        # Global U row = a * m + b = a*3 + b
                        U[a * m + b][col_offset + r] += u_val
            for b in range(2):
                for c in range(2):
                    v_val = V_mat[b*2 + c]
                    if v_val != 0:
                        # Local (b,c) -> global: b is j, c is k
                        # Global V row = b * p + c = b*4 + c
                        V[b * p + c][col_offset + r] += v_val
            for c in range(2):
                for a in range(2):
                    w_val = W_mat[c*2 + a]
                    if w_val != 0:
                        # Local (c,a) -> global: c is k, a is i
                        # Global W row = c * n + a = c*2 + a
                        W[c * n + a][col_offset + r] += w_val
    
    # Block 1: j in {0,1}, k in {0,1} -> cols 0-6
    # Local a=i, b=j (j=0,1), c=k (k=0,1)
    add_strassen_block(0, 0, 0, 0)
    
    # Block 2: j in {0,1}, k in {2,3} -> cols 7-13
    # Local a=i, b=j (j=0,1), c=k_local (c=0,1 maps to k=2,3)
    # We need to shift k by 2
    for r in range(7):
        U_mat, V_mat, W_mat = strassen_terms[r]
        for a in range(2):
            for b in range(2):
                u_val = U_mat[a*2 + b]
                if u_val != 0:
                    U[a * m + b][7 + r] += u_val
        for b in range(2):
            for c_local in range(2):
                v_val = V_mat[b*2 + c_local]
                if v_val != 0:
                    k = 2 + c_local
                    V[b * p + k][7 + r] += v_val
        for c_local in range(2):
            for a in range(2):
                w_val = W_mat[c_local*2 + a]
                if w_val != 0:
                    k = 2 + c_local
                    W[k * n + a][7 + r] += w_val
    
    # Block 3: j=2, k in {0,1,2,3} -> cols 14-15
    # T[i,2,k] = 1 for i in {0,1}, k in {0,1,2,3}
    # Rank-2 decomposition: two rank-1 tensors
    # r=14: U[i,2]=1 for i=0; V[2,k]=1 for all k; W[k,0]=1 for all k
    # r=15: U[i,2]=1 for i=1; V[2,k]=1 for all k; W[k,1]=1 for all k
    
    # r=14
    U[0 * m + 2][14] = 1.0
    for k in range(4):
        V[2 * p + k][14] = 1.0
        W[k * n + 0][14] = 1.0
    
    # r=15
    U[1 * m + 2][15] = 1.0
    for k in range(4):
        V[2 * p + k][15] = 1.0
        W[k * n + 1][15] = 1.0
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

def _verify_decomposition(U, V, W, n=2, m=3, p=4):
    """Verify that the decomposition exactly reconstructs the multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > 1e-10:
                    return False, (i, j, k, val)
    return True, None

if __name__ == "__main__":
    result = construct()
    valid, info = _verify_decomposition(result["U"], result["V"], result["W"])
    if valid:
        print("Valid rank-16 decomposition found.")
    else:
        print(f"Decomposition failed at {info}")
