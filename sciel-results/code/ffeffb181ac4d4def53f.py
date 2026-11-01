"""Matrix multiplication 2x2x16: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-32 decomposition for 2x2 * 2x16 matrix multiplication.
    
    Strategy: The 16-column dimension is treated as 8 independent 2x2 blocks.
    Each 2x2x2 block can be computed with rank 7 (Strassen).
    However, we can do better by exploiting the structure.
    
    Actually, for 2x2 * 2x16, we can view this as 8 separate 2x2 * 2x2 multiplications
    but sharing the input matrix A (2x2) across all 8 blocks.
    
    Let A = [[a00, a01], [a10, a11]] and B = [B_0, B_1, ..., B_7] where each B_k is 2x2.
    Output C_k = A * B_k for k=0,...,7.
    
    Using Strassen for each block would give 7*8 = 56, but we can share the
    Strassen linear combinations of A across blocks.
    
    Strassen's 7 products for A * B:
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)b00
    M3 = a00(b01-b11)
    M4 = a11(b10-b00)
    M5 = (a00+a01)b11
    M6 = (a10-a00)(b00+b10)
    M7 = (a01-a11)(b01+b11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    For 8 blocks, the linear combinations of A are:
    L1 = a00 + a11
    L2 = a10 + a11
    L3 = a00
    L4 = a11
    L5 = a00 + a01
    L6 = a10 - a00
    L7 = a01 - a11
    
    These 7 linear forms of A are shared across all 8 blocks.
    For each block k, we need 7 products of these linear forms with linear forms of B_k.
    
    Total rank = 7 * 8 = 56? No, that's not right.
    
    Wait, let me reconsider. The rank-1 terms are:
    For each block k and each Strassen product j, we have:
    U term: the j-th linear form of A (same for all k)
    V term: the corresponding linear form of B_k
    W term: contribution to C_k
    
    So we have 7*8 = 56 rank-1 terms. That's worse than schoolbook 64? No, 56 < 64.
    
    But can we do better? Let me think about the tensor structure more carefully.
    
    The tensor is T[i,j,k,l] = delta_{i,k} for the (i,j) element of A times (j,l) element of B
    giving (i,l) element of C.
    
    For 2x2 * 2x16:
    A has 4 elements, B has 32 elements, C has 32 elements.
    
    Actually, let me just implement the 56-rank Strassen-based decomposition and verify it.
    """
    n, m, p = DIMENSIONS  # 2, 2, 16
    
    # 8 blocks of 2x2
    num_blocks = p // 2  # 8
    
    # Strassen linear forms for A (2x2 matrix with indices (i,j))
    # A = [[a00, a01], [a10, a11]]
    # Index: row*2 + col
    # a00 -> 0, a01 -> 1, a10 -> 2, a11 -> 3
    
    # 7 linear forms of A
    # L1 = a00 + a11 -> [1, 0, 0, 1]
    # L2 = a10 + a11 -> [0, 0, 1, 1]
    # L3 = a00       -> [1, 0, 0, 0]
    # L4 = a11       -> [0, 0, 0, 1]
    # L5 = a00 + a01 -> [1, 1, 0, 0]
    # L6 = a10 - a00 -> [-1, 0, 1, 0]
    # L7 = a01 - a11 -> [0, 1, 0, -1]
    
    A_forms = [
        [1, 0, 0, 1],   # L1
        [0, 0, 1, 1],   # L2
        [1, 0, 0, 0],   # L3
        [0, 0, 0, 1],   # L4
        [1, 1, 0, 0],   # L5
        [-1, 0, 1, 0],  # L6
        [0, 1, 0, -1],  # L7
    ]
    
    # For each block k, B_k = [[b00, b01], [b10, b11]]
    # Index in B: block_k * 4 + row*2 + col
    # b00 -> k*4+0, b01 -> k*4+1, b10 -> k*4+2, b11 -> k*4+3
    
    # 7 linear forms of B_k
    # M1: (b00+b11) -> [1, 0, 0, 1]
    # M2: b00       -> [1, 0, 0, 0]
    # M3: (b01-b11) -> [0, 1, 0, -1]
    # M4: (b10-b00) -> [-1, 0, 1, 0]
    # M5: b11       -> [0, 0, 0, 1]
    # M6: (b00+b10) -> [1, 0, 1, 0]
    # M7: (b01+b11) -> [0, 1, 0, 1]
    
    B_forms = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 0, 1, 0],   # M6
        [0, 1, 0, 1],   # M7
    ]
    
    # Output reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W coefficients for each output element (row, col) in terms of M1..M7
    # C index: block_k * 4 + row*2 + col
    # c00 -> k*4+0, c01 -> k*4+1, c10 -> k*4+2, c11 -> k*4+3
    
    W_coeffs = {
        0: [1, 0, 0, 1, -1, 0, 1],   # c00 = M1 + M4 - M5 + M7
        1: [0, 0, 1, 0, 1, 0, 0],    # c01 = M3 + M5
        2: [0, 1, 0, 1, 0, 0, 0],    # c10 = M2 + M4
        3: [1, -1, 1, 0, 0, 1, 0],   # c11 = M1 - M2 + M3 + M6
    }
    
    rank = 7 * num_blocks  # 56
    
    # Build U, V, W
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    for k in range(num_blocks):
        for j in range(7):
            col = k * 7 + j
            
            # U: linear form of A
            for i in range(4):
                U[i][col] = A_forms[j][i]
            
            # V: linear form of B_k
            for b_idx in range(4):
                V[k * 4 + b_idx][col] = B_forms[j][b_idx]
            
            # W: contribution to C_k
            for c_idx in range(4):
                W[k * 4 + c_idx][col] = W_coeffs[c_idx][j]
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, max_samples=100):
    """Verify that the decomposition exactly computes matrix multiplication."""
    import random
    random.seed(42)
    
    for _ in range(max_samples):
        # Random integer matrices
        A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
        
        # True product C = A * B
        C_true = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        # Product via decomposition: C[i,k] = sum_c U[i*? + ?, c] * V[? * p + k, c] * W[k*n + i, c]
        # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
        # U[i*m + j, c] corresponds to A[i,j]
        # V[j*p + k, c] corresponds to B[j,k]
        # W[k*n + i, c] corresponds to C[i,k]
        
        C_decomp = [[0] * p for _ in range(n)]
        rank = len(U[0])
        for i in range(n):
            for k in range(p):
                total = 0
                for c in range(rank):
                    # Sum over j: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    # But W[k*n+i, c] is the same for all j, so factor it out
                    w_val = W[k * n + i][c]
                    if w_val != 0:
                        for j in range(m):
                            total += U[i * m + j][c] * V[j * p + k][c] * w_val
                C_decomp[i][k] = total
        
        for i in range(n):
            for k in range(p):
                if C_decomp[i][k] != C_true[i][k]:
                    return False
    return True

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print("Verifying decomposition...")
    valid = _verify_decomposition(result['U'], result['V'], result['W'], 2, 2, 16)
    print(f"Valid: {valid}")
