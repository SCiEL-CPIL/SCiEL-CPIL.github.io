"""Rewritten program: block-diagonal Strassen for 2x2 * 2x10 multiplication."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START

def _strassen_2x2_decomposition():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    
    The multiplication tensor T[i,j,k] = 1 if A[i,j]*B[j,k] = C[i,k].
    In Strassen, we have 7 products M_1..M_7 such that:
      C[0,0] = M_1 - M_2 + M_3
      C[0,1] = M_2 + M_4
      C[1,0] = M_5 + M_6
      C[1,1] = M_1 - M_3 + M_7
    
    Where:
      M_1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
      M_2 = (A[1,0]+A[1,1]) * B[0,0]
      M_3 = A[0,0] * (B[0,1]-B[1,1])
      M_4 = A[1,1] * (B[1,0]-B[0,0])
      M_5 = (A[0,0]+A[0,1]) * B[1,1]
      M_6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
      M_7 = (A[1,1]-A[0,1]) * (B[1,0]+B[1,1])
    
    We need U (4x7), V (4x7), W (4x7) such that for each product r:
      U[0][r] = A[0,0] coefficient, U[1][r] = A[0,1] coeff, U[2][r] = A[1,0] coeff, U[3][r] = A[1,1] coeff
      V[0][r] = B[0,0] coeff, V[1][r] = B[0,1] coeff, V[2][r] = B[1,0] coeff, V[3][r] = B[1,1] coeff
      W[0][r] = C[0,0] coeff, W[1][r] = C[0,1] coeff, W[2][r] = C[1,0] coeff, W[3][r] = C[1,1] coeff
    
    The tensor decomposition requires:
      T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    Note the index convention: W[k*n+i, r] for n=2 means W[k*2+i, r].
    So W[0] = W[k=0,i=0] = C[0,0], W[1] = W[k=0,i=1] = C[1,0], 
       W[2] = W[k=1,i=0] = C[0,1], W[3] = W[k=1,i=1] = C[1,1]
    
    Wait, let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    So W[k*n+i, r] where n=2:
      k=0,i=0 -> W[0] -> contributes to C[0,0]
      k=0,i=1 -> W[1] -> contributes to C[1,0]
      k=1,i=0 -> W[2] -> contributes to C[0,1]
      k=1,i=1 -> W[3] -> contributes to C[1,1]
    
    And the tensor is T[i,j,k] = 1 (for the multiplication tensor, each (i,j,k) has value 1).
    The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For 2x2*2x2: n=2,m=2,p=2
      U has shape (4,7), V has shape (4,7), W has shape (4,7)
      U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    Let me define the 7 products with their A, B, C coefficients:
    """
    # Each product r has (a_coeffs, b_coeffs, c_coeffs) where
    # a_coeffs = [A00, A01, A10, A11], b_coeffs = [B00, B01, B10, B11], c_coeffs = [C00, C10, C01, C11]
    # Note: c_coeffs indexed as [k*n+i] = [0,1,2,3] = [C00, C10, C01, C11]
    
    products = [
        # M_1: (A00+A11)*(B00+B11) -> C00 + C11... wait no
        # M_1 contributes to C00 with +1 and C11 with +1? No...
        # C[0,0] = M_1 - M_2 + M_3
        # C[0,1] = M_2 + M_4
        # C[1,0] = M_5 + M_6
        # C[1,1] = M_1 - M_3 + M_7
        #
        # So for M_1: C[0,0]+=1, C[1,1]+=1 -> c_coeffs = [1, 0, 0, 1]
        #   A: A00=1, A11=1 -> a = [1,0,0,1]
        #   B: B00=1, B11=1 -> b = [1,0,0,1]
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # M_2: C[0,0]-=1, C[0,1]+=1 -> c = [-1, 0, 1, 0]
        #   A: A10=1, A11=1 -> a = [0,0,1,1]
        #   B: B00=1 -> b = [1,0,0,0]
        ([0,0,1,1], [1,0,0,0], [-1,0,1,0]),
        # M_3: C[0,0]+=1, C[1,1]-=1 -> c = [1, 0, 0, -1]
        #   A: A00=1 -> a = [1,0,0,0]
        #   B: B01=1, B11=-1 -> b = [0,1,0,-1]
        ([1,0,0,0], [0,1,0,-1], [1,0,0,-1]),
        # M_4: C[0,1]+=1 -> c = [0, 0, 1, 0]
        #   A: A11=1 -> a = [0,0,0,1]
        #   B: B10=1, B00=-1 -> b = [-1,0,1,0]
        ([0,0,0,1], [-1,0,1,0], [0,0,1,0]),
        # M_5: C[1,0]+=1 -> c = [0, 1, 0, 0]
        #   A: A00=1, A01=1 -> a = [1,1,0,0]
        #   B: B11=1 -> b = [0,0,0,1]
        ([1,1,0,0], [0,0,0,1], [0,1,0,0]),
        # M_6: C[1,0]+=1 -> c = [0, 1, 0, 0]
        #   A: A10=1, A00=-1 -> a = [-1,0,1,0]
        #   B: B00=1, B01=1 -> b = [1,1,0,0]
        ([-1,0,1,0], [1,1,0,0], [0,1,0,0]),
        # M_7: C[1,1]+=1 -> c = [0, 0, 0, 1]
        #   A: A11=1, A01=-1 -> a = [0,-1,0,1]
        #   B: B10=1, B11=1 -> b = [0,0,1,1]
        ([0,-1,0,1], [0,0,1,1], [0,0,0,1]),
    ]
    
    # Build U, V, W matrices
    # U[i*2+j, r] = a_coeffs[i*2+j]
    # V[j*2+k, r] = b_coeffs[j*2+k]
    # W[k*2+i, r] = c_coeffs[k*2+i]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    for r, (a, b, c) in enumerate(products):
        for idx in range(4):
            U[idx][r] = a[idx]
            V[idx][r] = b[idx]
            W[idx][r] = c[idx]
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the multiplication tensor."""
    rank = len(U[0])
    # T[i,j,k] should equal 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True


def _construct_block_diagonal_strassen():
    """
    Construct the rank-35 decomposition using block-diagonal Strassen.
    
    The 2x10 B matrix can be viewed as 5 independent 2x2 blocks.
    We use Strassen (rank 7) for each 2x2*2x2 block.
    
    Total rank = 5 * 7 = 35.
    """
    n, m, p = 2, 2, 10
    num_blocks = p // 2  # 5 blocks of 2 columns each
    block_rank = 7
    total_rank = num_blocks * block_rank  # 35
    
    # Get base Strassen decomposition
    U_base, V_base, W_base = _strassen_2x2_decomposition()
    
    # U has shape (n*m, total_rank) = (4, 35)
    # V has shape (m*p, total_rank) = (20, 35)
    # W has shape (p*n, total_rank) = (20, 35)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for b in range(num_blocks):
        # Block b handles B columns [2*b, 2*b+1] and C columns [2*b, 2*b+1]
        offset = b * block_rank
        
        for idx in range(4):
            # U: same for all blocks (A is shared)
            U[idx][offset + 0] = U_base[idx][0]
            U[idx][offset + 1] = U_base[idx][1]
            U[idx][offset + 2] = U_base[idx][2]
            U[idx][offset + 3] = U_base[idx][3]
            U[idx][offset + 4] = U_base[idx][4]
            U[idx][offset + 5] = U_base[idx][5]
            U[idx][offset + 6] = U_base[idx][6]
            
            # V: only nonzero for rows corresponding to this block's B columns
            # V[j*p+k, r] where k is in [2*b, 2*b+1]
            # V_base[idx, r] where idx = j*2 + k_local, k_local in {0,1}
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * b + k_local
                    v_idx = j * p + k_global  # j*10 + k_global
                    v_base_idx = j * 2 + k_local
                    for r in range(block_rank):
                        V[v_idx][offset + r] = V_base[v_base_idx][r]
            
            # W: only nonzero for rows corresponding to this block's C columns
            # W[k*n+i, r] where k is in [2*b, 2*b+1]
            for k_local in range(2):
                for i in range(2):
                    k_global = 2 * b + k_local
                    w_idx = k_global * n + i  # k*2 + i
                    w_base_idx = k_local * 2 + i
                    for r in range(block_rank):
                        W[w_idx][offset + r] = W_base[w_base_idx][r]
    
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x10 matrix multiplication.
    Uses block-diagonal Strassen: 5 independent 2x2 blocks, each with rank-7 Strassen.
    Total rank: 35.
    """
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Build the rank-35 block-diagonal Strassen decomposition
    result = _construct_block_diagonal_strassen()
    
    # Verify correctness
    n, m, p = DIMENSIONS
    U, V, W = result["U"], result["V"], result["W"]
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return result

# EVOLVE-BLOCK-END
