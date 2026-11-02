import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Strategy: Construct a rank-27 decomposition for 2x3x6
    # by using three rank-9 decompositions for 2x3x2.
    # Each 2x3x2 is decomposed into a 2x2x2 (Strassen, rank 7) 
    # and a 2x1x2 (rank 2) part.
    
    # 1. Define Rank 7 Strassen for 2x2x2
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # T[i,j,k] = 1
    # Strassen terms:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # U (i,j) -> index i*2 + j
    # V (j,k) -> index j*2 + k
    # W (k,i) -> index k*2 + i
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # Term 1: (A00+A11)(B00+B11) -> C00 += M1, C01 += M1, C10 += M1, C11 += M1
    # U: A00 (0), A11 (3)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    # V: B00 (0), B11 (3)
    V_s[0, 0] = 1; V_s[3, 0] = 1
    # W: C00 (0), C01 (1), C10 (2), C11 (3)
    W_s[0, 0] = 1; W_s[1, 0] = 1; W_s[2, 0] = 1; W_s[3, 0] = 1
    
    # Term 2: (A10+A11)B00 -> C10 += M2, C11 += M2
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = 1
    
    # Term 3: A00(B10-B11) -> C00 -= M3, C01 -= M3
    U_s[0, 2] = 1
    V_s[2, 2] = 1; V_s[3, 2] = -1
    W_s[0, 2] = -1; W_s[1, 2] = -1
    
    # Term 4: A11(B01-B00) -> C10 += M4, C11 += M4
    U_s[3, 3] = 1
    V_s[1, 3] = 1; V_s[0, 3] = -1
    W_s[2, 3] = 1; W_s[3, 3] = 1
    
    # Term 5: (A00+A01)B11 -> C00 += M5, C01 += M5
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = 1; W_s[1, 4] = 1
    
    # Term 6: (A10-A00)(B00+B01) -> C10 -= M6, C11 -= M6
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[2, 5] = -1; W_s[3, 5] = -1
    
    # Term 7: (A01-A11)(B10+B11) -> C00 += M7, C01 += M7
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1; W_s[1, 6] = 1

    # 2. Define Rank 2 for 2x1x2
    # Indices: i in {0,1}, j=0 (local), k in {0,1}
    # T[i,0,k] = 1
    # This is C = A B^T where A is 2x1, B is 1x2.
    # C_ik = A_i B_k.
    # We need sum_l U_{i,l} V_{0,l} W_{k,l} = 1? No.
    # The tensor entry is 1 for all i,k.
    # Wait, for 2x1x2, the tensor is T_{(i,0),(0,k),(k,i)} = 1.
    # This means C_{ik} = A_{i0} B_{0k}.
    # This is a rank-1 tensor in the (i,k) space?
    # T_{i,k} = 1.
    # We can decompose the all-ones 2x2 matrix into sum of rank-1 matrices?
    # No, the tensor rank is defined by the cyclic indices.
    # T_{(i,0),(0,k),(k,i)} = 1.
    # Let's check if rank 2 is sufficient.
    # We need U_i V_k W_{ki} = 1 for all i,k?
    # If we set l=1: U_0=1, V_0=1, W_{0,0}=1, W_{0,1}=0, W_{1,0}=0, W_{1,1}=1.
    # Then term 1 is 1 for (0,0) and 0 for others.
    # If we set l=2: U_1=1, V_1=1, W_{1,1}=1 (wait, W is fixed).
    # We need to cover (0,0), (0,1), (1,0), (1,1).
    # The tensor is actually the identity? No, T_{i,k}=1 for all i,k.
    # So we need sum_l U_{i,l} V_{k,l} W_{k,i,l} = 1.
    # If we choose W_{k,i,l} = 1 for all k,i,l, then we need sum_l U_{i,l} V_{k,l} = 1.
    # This is decomposing the all-ones 2x2 matrix into sum of rank-1 matrices.
    # Rank of all-ones 2x2 matrix is 1.
    # So we can do it with 1 term?
    # U_0=1, U_1=1. V_0=1, V_1=1. W_{k,i}=1.
    # Then sum is 1*1*1 = 1.
    # So R(2,1,2) = 1?
    # Let's verify.
    # T_{(i,0),(0,k),(k,i)} = 1.
    # Term: U_{i,0,1} V_{0,k,1} W_{k,i,1}.
    # If U_{i,0,1}=1, V_{0,k,1}=1, W_{k,i,1}=1.
    # Then for any i,k, the value is 1.
    # So yes, Rank 1 is sufficient for 2x1x2.
    
    U_r2 = np.zeros((2, 1))
    V_r2 = np.zeros((1, 1))
    W_r2 = np.zeros((2, 1))
    
    U_r2[0, 0] = 1; U_r2[1, 0] = 1
    V_r2[0, 0] = 1
    W_r2[0, 0] = 1; W_r2[1, 0] = 1 # Wait, W is (p*n) = (2*2)=4?
    # W index is k*n + i. k in {0,1}, i in {0,1}.
    # W_{k,i} = 1 for all k,i.
    W_r2_flat = np.ones(4)
    
    # So R(2,3,2) <= 7 + 1 = 8.
    # Total Rank for 2x3x6 = 3 * 8 = 24.
    # Score 36/24 = 1.5.

    # Let's construct the Rank 8 block for 2x3x2.
    # U_block: (n*m, 8) = (6, 8)
    # V_block: (m*p, 8) = (6, 8) -> m=3, p=2.
    # W_block: (p*n, 8) = (4, 8) -> p=2, n=2.
    
    U_block = np.zeros((6, 8))
    V_block = np.zeros((6, 8))
    W_block = np.zeros((4, 8))
    
    # Part 1: Strassen (j=0,1). Indices in U: i*3 + j.
    # U_s is (4,7) for i in {0,1}, j in {0,1}.
    # Map U_s[i*2+j, l] to U_block[i*3+j, l].
    for i in range(2):
        for j in range(2):
            U_block[i*3 + j, 0:7] = U_s[i*2 + j, 0:7]
            
    # V_s is (4,7) for j in {0,1}, k in {0,1}.
    # Map V_s[j*2+k, l] to V_block[j*2+k, l].
    for j in range(2):
        for k in range(2):
            V_block[j*2 + k, 0:7] = V_s[j*2 + k, 0:7]
            
    # W_s is (4,7) for k in {0,1}, i in {0,1}.
    # Map W_s[k*2+i, l] to W_block[k*2+i, l].
    for k in range(2):
        for i in range(2):
            W_block[k*2 + i, 0:7] = W_s[k*2 + i, 0:7]
            
    # Part 2: Rank 1 (j=2).
    # U_r2 is (2,1) for i in {0,1}, j=2.
    # Map U_r2[i, 0] to U_block[i*3 + 2, 7].
    U_block[0*3 + 2, 7] = 1
    U_block[1*3 + 2, 7] = 1
    
    # V_r2 is (1,1) for j=2, k in {0,1}.
    # Map V_r2[0, 0] to V_block[2*2 + k, 7].
    V_block[2*2 + 0, 7] = 1
    V_block[2*2 + 1, 7] = 1
    
    # W_r2 is (4,1) for k in {0,1}, i in {0,1}.
    # Map W_r2[k*2+i, 0] to W_block[k*2+i, 7].
    W_block[:, 7] = 1.0

    # Now assemble the full 2x3x6 decomposition using 3 blocks.
    # Total rank = 3 * 8 = 24.
    U_full = np.zeros((6, 24))
    V_full = np.zeros((18, 24))
    W_full = np.zeros((12, 24))
    
    for b in range(3):
        # Block b corresponds to k in {2b, 2b+1}
        offset = b * 8
        
        # U is independent of k, so it's the same for all blocks?
        # No, U is (n*m, rank). The rank dimension is separate.
        # So U_full[:, offset:offset+8] = U_block.
        U_full[:, offset:offset+8] = U_block
        
        # V is (m*p, rank). m=3, p=6.
        # V index is j*6 + k.
        # In the block, V_block index is j*2 + k_local.
        # k_local in {0,1} maps to k_global = 2b + k_local.
        for j in range(3):
            for k_local in range(2):
                k_global = 2*b + k_local
                V_full[j*6 + k_global, offset:offset+8] = V_block[j*2 + k_local, 0:8]
                
        # W is (p*n, rank). p=6, n=2.
        # W index is k*2 + i.
        # In the block, W_block index is k_local*2 + i.
        # k_local in {0,1} maps to k_global = 2b + k_local.
        for k_local in range(2):
            for i in range(2):
                k_global = 2*b + k_local
                W_full[k_global*2 + i, offset:offset+8] = W_block[k_local*2 + i, 0:8]
                
    # Verification
    # Check if U @ V^T @ W^T reconstructs the tensor?
    # The decomposition is sum_l U_{ij,l} V_{jk,l} W_{ki,l} = 1.
    # We can verify by checking a few entries.
    
    # If time allows, verify fully.
    if time.time() < deadline - 5:
        T_recon = np.zeros((2, 3, 6))
        for l in range(24):
            u = U_full[:, l].reshape(2, 3)
            v = V_full[:, l].reshape(3, 6)
            w = W_full[:, l].reshape(6, 2)
            # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            # This is slow in python, but 24 * 36 is small.
            for i in range(2):
                for j in range(3):
                    for k in range(6):
                        T_recon[i, j, k] += u[i, j] * v[j, k] * w[k, i]
                        
        if not np.allclose(T_recon, 1.0):
            # Fallback to Rank 34 if verification fails
            return _construct_rank_34()

    return {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }

def _construct_rank_34():
    # Fallback to the known valid rank 34 solution
    n, m, p = DIMENSIONS
    
    U233_seed = np.zeros((6, 17))
    V233_seed = np.zeros((9, 17))
    W233_seed = np.zeros((6, 17))
    
    U233_seed[0, 0] = 1; U233_seed[4, 0] = 1
    V233_seed[0, 0] = 1; V233_seed[4, 0] = 1
    W233_seed[0, 0] = 1
    W233_seed[3, 0] = 1
    
    U233_seed[3, 1] = 1; U233_seed[4, 1] = 1
    V233_seed[0, 1] = 1
    W233_seed[1, 1] = 1
    W233_seed[3, 1] = -1
    
    U233_seed[0, 2] = 1
    V233_seed[1, 2] = 1; V233_seed[4, 2] = -1
    W233_seed[2, 2] = 1
    W233_seed[3, 2] = 1
    
    U233_seed[4, 3] = 1
    V233_seed[3, 3] = 1; V233_seed[0, 3] = -1
    W233_seed[0, 3] = 1
    W233_seed[1, 3] = 1
    
    U233_seed[0, 4] = 1; U233_seed[1, 4] = 1
    V233_seed[4, 4] = 1
    W233_seed[0, 4] = -1
    W233_seed[2, 4] = 1
    
    U233_seed[3, 5] = 1; U233_seed[0, 5] = -1
    V233_seed[0, 5] = 1; V233_seed[1, 5] = 1
    W233_seed[3, 5] = 1
    
    U233_seed[1, 6] = 1; U233_seed[4, 6] = -1
    V233_seed[3, 6] = 1; V233_seed[4, 6] = 1
    W233_seed[0, 6] = 1
    
    U233_seed[0, 7] = 1
    V233_seed[2, 7] = 1
    W233_seed[4, 7] = 1
    
    U233_seed[1, 8] = 1
    V233_seed[5, 8] = 1
    W233_seed[4, 8] = 1
    
    U233_seed[3, 9] = 1
    V233_seed[2, 9] = 1
    W233_seed[5, 9] = 1
    
    U233_seed[4, 10] = 1
    V233_seed[5, 10] = 1
    W233_seed[5, 10] = 1
    
    U233_seed[2, 11] = 1
    V233_seed[6, 11] = 1
    W233_seed[0, 11] = 1
    
    U233_seed[2, 12] = 1
    V233_seed[7, 12] = 1
    W233_seed[2, 12] = 1
    
    U233_seed[2, 13] = 1
    V233_seed[8, 13] = 1
    W233_seed[4, 13] = 1
    
    U233_seed[5, 14] = 1
    V233_seed[6, 14] = 1
    W233_seed[1, 14] = 1
    
    U233_seed[5, 15] = 1
    V233_seed[7, 15] = 1
    W233_seed[3, 15] = 1
    
    U233_seed[5, 16] = 1
    V233_seed[8, 16] = 1
    W233_seed[5, 16] = 1
    
    U_comb = np.zeros((6, 34))
    V_comb = np.zeros((18, 34))
    W_comb = np.zeros((12, 34))
    
    U_comb[:, :17] = U233_seed
    U_comb[:, 17:] = U233_seed
    
    for j in range(3):
        for k in range(3):
            V_comb[j * 6 + k, :17] = V233_seed[j * 3 + k]
            V_comb[j * 6 + 3 + k, 17:] = V233_seed[j * 3 + k]
            
    for k in range(3):
        for i in range(2):
            W_comb[k * 2 + i, :17] = W233_seed[k * 2 + i]
            W_comb[(k + 3) * 2 + i, 17:] = W233_seed[k * 2 + i]
            
    return {
        "U": U_comb.tolist(),
        "V": V_comb.tolist(),
        "W": W_comb.tolist()
    }

if __name__ == "__main__":
    construct()
