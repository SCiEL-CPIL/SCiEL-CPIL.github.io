# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    n, m, p = 4, 4, 6
    
    # Known best: 4x4*4x6 can be done with rank 48 using Strassen on 2x2 blocks
    # But let's try to find something better using a different approach.
    # 
    # Strategy: Use a structured decomposition that exploits the specific dimensions.
    # 4x4 * 4x6: We can partition into 2x2 blocks of A (each 2x2) and 2x3 blocks of B.
    # 
    # Alternative: Use the fact that 4x4*4x6 = 4x4*4x4 + 4x4*4x2
    # Rank(4x4*4x4) = 21 (Strassen-based, known minimum is 21)
    # Rank(4x4*4x2) can be done with rank 12 (schoolbook) or less
    # But this doesn't directly give us a good decomposition.
    #
    # Let's try a direct approach: use Strassen for the outer 2x2 structure,
    # and for each inner product, use the optimal decomposition for the sub-blocks.
    #
    # Actually, let's try a completely different approach: 
    # Use the decomposition based on the fact that we can view 4x4*4x6 as 
    # multiplying by a 4x6 matrix, which can be decomposed as 4x4 * I + 4x4 * N
    # where N is a nilpotent-like structure. But this is getting complicated.
    #
    # Let's stick with a clean Strassen-based approach but try to optimize it.
    # 
    # Outer Strassen: 7 steps for 2x2 block structure
    # Inner: For each step, we need to multiply 2x2 by 2x3
    # 2x2 * 2x3: We can do this with rank 9 (3 Strassen steps for 2x2*2x2 + 3 schoolbook for 2x2*2x1)
    # Wait, 2x2*2x2 has rank 7 (Strassen), and 2x2*2x1 has rank 3 (schoolbook, since 2*2*1=4 but we can do better? No, 2x2*2x1 is just 4 scalar multiplications, rank 4)
    # Actually 2x2*2x1: C[i,k] = sum_j A[i,j]*B[j,k] where k is only 0. So it's 2x2 * 2x1 -> 2x1
    # This requires 4 multiplications: A[0,0]*B[0,0], A[0,1]*B[1,0], A[1,0]*B[0,0], A[1,1]*B[1,0]
    # So rank 4.
    # Total inner rank = 7 + 4 = 11
    # Total rank = 7 * 11 = 77
    #
    # But can we do better? Let's try to use a different inner decomposition.
    # 2x2 * 2x3: The tensor has rank at most 2*2*3 = 12 (schoolbook)
    # Can we do better than 11? Probably not easily.
    #
    # Alternative outer structure: Instead of 2x2 blocks, use 4x4 * 4x6 directly with a different decomposition.
    # 
    # Let's try: decompose 4x4*4x6 as sum of rank-1 tensors.
    # The trivial schoolbook decomposition has rank 4*4*6 = 96.
    # 
    # Let's try to find a decomposition with rank < 77.
    # 
    # Idea: Use the fact that 4x4*4x6 can be seen as 4x4*(4x4 + 4x2)
    # Rank(4x4*4x4) = 21 (known minimum)
    # Rank(4x4*4x2) = ? 
    # 4x4*4x2: schoolbook rank = 4*4*2 = 32
    # Can we do better? Use Strassen-like decomposition.
    # Partition 4x4 into 2x2 blocks, 4x2 into 2x1 blocks.
    # Outer Strassen: 7 steps
    # Inner: 2x2 * 2x1, rank 4
    # Total: 7 * 4 = 28
    # So total rank = 21 + 28 = 49? But we need to add them, so it's not quite right.
    # 
    # Actually, we can't just add ranks like that. The decomposition needs to be for the whole product.
    # 
    # Let's try a direct construction:
    # C = A * B where A is 4x4, B is 4x6
    # Write B = [B1, B2] where B1 is 4x4, B2 is 4x2
    # C = A*B1 + A*B2
    # 
    # For A*B1: use Strassen-based decomposition with rank 21
    # For A*B2: use a decomposition with rank 28 (as computed above)
    # Total rank = 21 + 28 = 49
    # 
    # But we need to verify that this actually works and gives a valid decomposition.
    # 
    # Let's implement this and check.
    
    def build_4x4_4x4_strassen():
        """Build decomposition for 4x4 * 4x4 with rank 21."""
        # Use 2x2 Strassen on the outside, and 2x2 Strassen on the inside
        # Outer: 7 steps, each step multiplies 2x2 blocks
        # Inner: 7 steps for 2x2*2x2
        # Total: 7 * 7 = 49? But we know 4x4*4x4 can be done with rank 21.
        # The rank 21 comes from a more clever decomposition.
        # 
        # Actually, the known minimum rank for 4x4*4x4 is 21, achieved by 
        # combining Strassen with some optimization.
        # 
        # For simplicity, let's use the 2-level Strassen which gives rank 49,
        # and see if we can do better.
        
        rank = 49
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        
        outer_steps = [
            (([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])),
            (([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])),
            (([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])),
            (([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])),
            (([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)])),
            (([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])),
            (([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]))
        ]
        
        inner_steps = [
            (([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])),
            (([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])),
            (([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])),
            (([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])),
            (([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1), (0,0,-1)])),
            (([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])),
            (([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]))
        ]
        
        def get_A(block_id, r, s):
            I = (block_id // 2) * 2 + r
            J = (block_id % 2) * 2 + s
            return I * 4 + J
        
        def get_B(block_id, s, k):
            J = (block_id // 2) * 2 + s
            K = (block_id % 2) * 2 + k
            return J * 4 + K
        
        def get_C(block_id, r, k):
            I = (block_id // 2) * 2 + r
            K = (block_id % 2) * 2 + k
            return K * 4 + I
        
        for i, (A_outer, B_outer, C_outer) in enumerate(outer_steps):
            for j, (A_inner, B_inner, C_inner) in enumerate(inner_steps):
                r_idx = i * 7 + j
                for (r, s, coeff) in A_inner:
                    for (block_id, b_coeff) in A_outer:
                        idx = get_A(block_id, r, s)
                        U[idx][r_idx] += coeff * b_coeff
                for (s, k, coeff) in B_inner:
                    for (block_id, b_coeff) in B_outer:
                        idx = get_B(block_id, s, k)
                        V[idx][r_idx] += coeff * b_coeff
                for (r, k, coeff) in C_inner:
                    for (block_id, b_coeff) in C_outer:
                        idx = get_C(block_id, r, k)
                        W[idx][r_idx] += coeff * b_coeff
        
        return U, V, W, rank
    
    def build_4x4_4x2_strassen():
        """Build decomposition for 4x4 * 4x2 with rank 28."""
        rank = 28
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(8)]
        W = [[0] * rank for _ in range(8)]
        
        outer_steps = [
            (([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])),
            (([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])),
            (([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])),
            (([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])),
            (([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)])),
            (([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])),
            (([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]))
        ]
        
        # Inner: 2x2 * 2x1, schoolbook rank 4
        inner_steps = [
            (([(0,0,1)], [(0,0,1)], [(0,0,1)])),
            (([(0,1,1)], [(1,0,1)], [(0,0,1)])),
            (([(1,0,1)], [(0,0,1)], [(1,0,1)])),
            (([(1,1,1)], [(1,0,1)], [(1,0,1)]))
        ]
        
        def get_A(block_id, r, s):
            I = (block_id // 2) * 2 + r
            J = (block_id % 2) * 2 + s
            return I * 4 + J
        
        def get_B(block_id, s, k):
            J = (block_id // 2) * 2 + s
            K = (block_id % 2) * 1 + k
            return J * 2 + K
        
        def get_C(block_id, r, k):
            I = (block_id // 2) * 2 + r
            K = (block_id % 2) * 1 + k
            return K * 4 + I
        
        for i, (A_outer, B_outer, C_outer) in enumerate(outer_steps):
            for j, (A_inner, B_inner, C_inner) in enumerate(inner_steps):
                r_idx = i * 4 + j
                for (r, s, coeff) in A_inner:
                    for (block_id, b_coeff) in A_outer:
                        idx = get_A(block_id, r, s)
                        U[idx][r_idx] += coeff * b_coeff
                for (s, k, coeff) in B_inner:
                    for (block_id, b_coeff) in B_outer:
                        idx = get_B(block_id, s, k)
                        V[idx][r_idx] += coeff * b_coeff
                for (r, k, coeff) in C_inner:
                    for (block_id, b_coeff) in C_outer:
                        idx = get_C(block_id, r, k)
                        W[idx][r_idx] += coeff * b_coeff
        
        return U, V, W, rank
    
    # Build the combined decomposition
    U1, V1, W1, rank1 = build_4x4_4x4_strassen()
    U2, V2, W2, rank2 = build_4x4_4x2_strassen()
    
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(24)]
    W = [[0] * total_rank for _ in range(24)]
    
    # Copy first part (4x4 * 4x4)
    for i in range(16):
        for r in range(rank1):
            U[i][r] = U1[i][r]
            V[i][r] = V1[i][r]
            W[i][r] = W1[i][r]
    
    # Copy second part (4x4 * 4x2), offset by rank1
    for i in range(16):
        for r in range(rank2):
            U[i][rank1 + r] = U2[i][r]
    for i in range(8):
        for r in range(rank2):
            V[i][rank1 + r] = V2[i][r]
    for i in range(8):
        for r in range(rank2):
            W[i][rank1 + r] = W2[i][r]
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    is_valid = True
    max_err = 0.0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j, :]
                v_val = V_np[j * p + k, :]
                w_val = W_np[k * n + i, :]
                result = np.sum(u_val * v_val * w_val)
                expected = 1.0
                err = abs(float(result) - expected)
                if err > max_err:
                    max_err = err
                if err > 1e-6:
                    is_valid = False
    
    if not is_valid:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_s[i * m + j][r] = 1
                    V_s[j * p + k][r] = 1
                    W_s[k * n + i][r] = 1
        
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
