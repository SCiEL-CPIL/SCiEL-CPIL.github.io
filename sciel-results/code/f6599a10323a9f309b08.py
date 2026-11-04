"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_block_factors():
    """Return (U, V, W) for a 2x2x2 Strassen block.
    U: (2*2, 7), V: (2*2, 7), W: (2*2, 7)
    U rows indexed by (i,j) with i in {0,1}, j in {0,1} -> row = i*2+j
    V rows indexed by (j,k) with j in {0,1}, k in {0,1} -> row = j*2+k
    W rows indexed by (k,i) with k in {0,1}, i in {0,1} -> row = k*2+i
    """
    rank = 7
    U = np.zeros((4, rank), dtype=int)
    V = np.zeros((4, rank), dtype=int)
    W = np.zeros((4, rank), dtype=int)
    
    # Strassen products:
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    #
    # A = [[A11, A12], [A21, A22]] -> A[i,j]
    # B = [[B11, B12], [B21, B22]] -> B[j,k]
    # C = A*B = [[C11, C12], [C21, C22]]
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U: (i,j) -> linear form on A
    # M1: A11 + A22 -> U[0,0]+=1, U[3,0]+=1
    # M2: A21 + A22 -> U[2,1]+=1, U[3,1]+=1
    # M3: A11 -> U[0,2]+=1
    # M4: A22 -> U[3,3]+=1
    # M5: A11 + A12 -> U[0,4]+=1, U[1,4]+=1
    # M6: A21 - A11 -> U[2,5]+=1, U[0,5]-=1
    # M7: A12 - A22 -> U[1,6]+=1, U[3,6]-=1
    
    U[0, 0] += 1; U[3, 0] += 1  # M1
    U[2, 1] += 1; U[3, 1] += 1  # M2
    U[0, 2] += 1                # M3
    U[3, 3] += 1                # M4
    U[0, 4] += 1; U[1, 4] += 1  # M5
    U[2, 5] += 1; U[0, 5] -= 1  # M6
    U[1, 6] += 1; U[3, 6] -= 1  # M7
    
    # V: (j,k) -> linear form on B
    # M1: B11 + B22 -> V[0,0]+=1, V[3,0]+=1
    # M2: B11 -> V[0,1]+=1
    # M3: B12 - B22 -> V[1,2]+=1, V[3,2]-=1
    # M4: B21 - B11 -> V[2,3]+=1, V[0,3]-=1
    # M5: B22 -> V[3,4]+=1
    # M6: B11 + B12 -> V[0,5]+=1, V[1,5]+=1
    # M7: B21 + B22 -> V[2,6]+=1, V[3,6]+=1
    
    V[0, 0] += 1; V[3, 0] += 1  # M1
    V[0, 1] += 1                # M2
    V[1, 2] += 1; V[3, 2] -= 1  # M3
    V[2, 3] += 1; V[0, 3] -= 1  # M4
    V[3, 4] += 1                # M5
    V[0, 5] += 1; V[1, 5] += 1  # M6
    V[2, 6] += 1; V[3, 6] += 1  # M7
    
    # W: (k,i) -> linear form on C output
    # C11 = M1 + M4 - M5 + M7 -> W[0,0]+=1, W[0,3]+=1, W[0,4]-=1, W[0,6]+=1
    # C12 = M3 + M5 -> W[1,2]+=1, W[1,4]+=1
    # C21 = M2 + M4 -> W[2,1]+=1, W[2,3]+=1
    # C22 = M1 - M2 + M3 + M6 -> W[3,0]+=1, W[3,1]-=1, W[3,2]+=1, W[3,5]+=1
    
    W[0, 0] += 1; W[0, 3] += 1; W[0, 4] -= 1; W[0, 6] += 1  # C11
    W[1, 2] += 1; W[1, 4] += 1                               # C12
    W[2, 1] += 1; W[2, 3] += 1                               # C21
    W[3, 0] += 1; W[3, 1] -= 1; W[3, 2] += 1; W[3, 5] += 1   # C22
    
    return U, V, W


def _winograd_2x4x4_factors():
    """Return (U, V, W) for a 2x4x4 Winograd block with rank 16.
    U: (2*4, 16), V: (4*4, 16), W: (4*2, 16)
    U rows indexed by (i,j) with i in {0,1}, j in {0..3} -> row = i*4+j
    V rows indexed by (j,k) with j in {0..3}, k in {0..3} -> row = j*4+k
    W rows indexed by (k,i) with k in {0..3}, i in {0,1} -> row = k*2+i
    """
    rank = 16
    U = np.zeros((8, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((8, rank), dtype=int)
    
    # Winograd 2x4x4: We use the standard Winograd algorithm for 2x4 convolution.
    # The factor matrices can be derived from the Winograd algorithm.
    # For a 2x4x4 tensor, we need 16 multiplications.
    #
    # The Winograd algorithm for 2x4x4 uses the following structure:
    # U is 8x16, V is 16x16, W is 8x16
    #
    # Let me use the known Winograd factors for 2x4x4.
    # The key insight is that Winograd uses a basis change on the input and output.
    #
    # For the 2x4x4 case, the factors are:
    # U: 8x16 matrix where each row corresponds to a linear combination of A elements
    # V: 16x16 matrix where each row corresponds to a linear combination of B elements
    # W: 8x16 matrix where each row corresponds to a linear combination of C elements
    #
    # I'll construct these using the Winograd algorithm structure.
    # The Winograd algorithm for 2x4 convolution uses:
    # - 2x2 blocks of the input
    # - A specific set of 16 multiplications
    #
    # Let me use the standard Winograd 2x4x4 factors from the literature.
    # These factors are well-known and can be verified.
    
    # U factors (A side): 8 rows (i=0,1; j=0,1,2,3) x 16 cols
    # The Winograd algorithm groups the A matrix into 2x2 blocks and applies transforms.
    # For simplicity, I'll use a known valid set of factors.
    
    # Actually, let me use a simpler approach: use 4 Strassen 2x2x2 blocks for the 2x4x4 case.
    # This gives rank 28, which is worse than the current 84 total.
    # So I need to find a better decomposition.
    
    # Let me try a different approach: use the Winograd algorithm for 2x4x4.
    # The Winograd algorithm for 2x4x4 has rank 16.
    # I'll construct the factors based on the Winograd algorithm structure.
    
    # For now, I'll return None to indicate that this function needs to be implemented
    # with actual Winograd factors. The caller will fall back to Strassen.
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Try to use Winograd 2x4x4 for the first 2x4x4 block
    # i in {0,1}, j in {0..3}, k in {0..3}
    # If Winograd factors are available and valid, use them
    # Otherwise, fall back to Strassen 2x2x2 blocks
    
    winograd_factors = _winograd_2x4x4_factors()
    use_winograd = winograd_factors is not None
    
    if use_winograd:
        U_w, V_w, W_w = winograd_factors
        rank_w = 16
    else:
        U_w, V_w, W_w = None, None, None
        rank_w = 0
    
    # Determine the block structure
    # If using Winograd: 2x4x4 block for i in {0,1}, j in {0..3}, k in {0..3}
    # Rest: i=2, j in {0..4}, k in {0..5} (30 terms)
    #        i in {0,1}, j=4, k in {0..5} (12 terms)
    #        i in {0,1}, j in {0..3}, k in {4,5} (16 terms)
    # Total remainder: 58 terms
    
    # If not using Winograd: 6 Strassen 2x2x2 blocks
    # i in {0}, j in {0,1}, k in {0,1,2} -> 6 blocks
    # Each block: i in {2b_i, 2b_i+1}, j in {2b_j, 2b_j+1}, k in {2b_k, 2b_k+1}
    # But n=3, so only b_i=0 is valid (i in {0,1})
    # m=5, so b_j in {0,1} (j in {0,1} or {2,3})
    # p=6, so b_k in {0,1,2} (k in {0,1}, {2,3}, or {4,5})
    # Total: 1 * 2 * 3 = 6 blocks
    # Each block covers 2x2x2 = 8 terms
    # Total covered: 6 * 8 = 48 terms
    # Remainder: 90 - 48 = 42 terms
    
    if use_winograd:
        # Winograd covers 2x4x4 = 32 terms
        # Remainder: 90 - 32 = 58 terms
        rank = rank_w + 58
    else:
        # Strassen covers 6 * 8 = 48 terms
        # Remainder: 90 - 48 = 42 terms
        rank = 6 * 7 + 42
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    if use_winograd:
        # Place Winograd factors for the 2x4x4 block
        # i in {0,1}, j in {0..3}, k in {0..3}
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    u_row = i * m + j
                    v_row = j * p + k
                    w_row = k * n + i
                    for c in range(rank_w):
                        U[u_row, col + c] = U_w[i * 4 + j, c]
                        V[v_row, col + c] = V_w[j * 4 + k, c]
                        W[w_row, col + c] = W_w[k * 2 + i, c]
        col += rank_w
        
        # Handle remainder with schoolbook
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Check if this (i,j,k) is in the Winograd block
                    if i < 2 and j < 4 and k < 4:
                        continue
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    else:
        # Use Strassen 2x2x2 blocks
        U_s, V_s, W_s = _strassen_block_factors()
        rank_s = 7
        
        blocks = []
        for b_i in range(1):
            for b_j in range(2):
                for b_k in range(3):
                    blocks.append((b_i, b_j, b_k))
        
        for b_i, b_j, b_k in blocks:
            i0 = 2 * b_i
            j0 = 2 * b_j
            k0 = 2 * b_k
            
            for c in range(rank_s):
                for u in range(2):
                    for v in range(2):
                        i = i0 + u
                        j = j0 + v
                        k = k0 + v  # Wait, this is wrong. Let me fix.
                        # Actually, for the Strassen block:
                        # i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
                        # The U matrix for this block is placed at rows (i*m+j)
                        # The V matrix for this block is placed at rows (j*p+k)
                        # The W matrix for this block is placed at rows (k*n+i)
                        pass
            
            # Place Strassen factors
            for u in range(2):
                for v in range(2):
                    i = i0 + u
                    j = j0 + v
                    for c in range(rank_s):
                        U[i * m + j, col + c] = U_s[u * 2 + v, c]
            
            for u in range(2):
                for v in range(2):
                    j = j0 + u
                    k = k0 + v
                    for c in range(rank_s):
                        V[j * p + k, col + c] = V_s[u * 2 + v, c]
            
            for u in range(2):
                for v in range(2):
                    k = k0 + u
                    i = i0 + v
                    for c in range(rank_s):
                        W[k * n + i, col + c] = W_s[u * 2 + v, c]
            
            col += rank_s
        
        # Handle remainder with schoolbook
        covered = set()
        for b_i, b_j, b_k in blocks:
            for u in range(2):
                for v in range(2):
                    for w in range(2):
                        i = 2 * b_i + u
                        j = 2 * b_j + v
                        k = 2 * b_k + w
                        covered.add((i, j, k))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
    
    # Convert to lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
