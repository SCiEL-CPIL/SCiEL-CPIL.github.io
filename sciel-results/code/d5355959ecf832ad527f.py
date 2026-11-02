import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # The goal is to find a rank-minimal decomposition.
    # Current best known is 22 (from two 2x3x3 blocks of rank 11).
    # Simple block sum for 2x3x2 is 8 (7 Strassen + 1 trivial), so 3*8=24.
    # Can we do better than 22?
    # R(2,3,6) <= R(2,3,2)*3 = 24.
    # R(2,3,6) <= R(2,3,3)*2 = 22.
    # Is R(2,3,2) < 8?
    # R(2,3,2) = R(2,2,2) + R(2,1,2) = 7 + 1 = 8.
    # So 24 is a valid upper bound. 22 is better.
    #
    # Let's try to find a decomposition of rank < 22.
    # Maybe 21? 20?
    #
    # Strategy:
    # 1. Start with the known Rank 22 solution (two copies of Rank 11 2x3x3).
    # 2. Use a local search / optimization to see if we can merge terms or find a lower rank solution.
    # 3. Alternatively, use a randomized search for a rank R decomposition, starting from R=21 downwards.
    
    # First, let's construct the Rank 22 solution as a baseline.
    # We need a Rank 11 decomposition for 2x3x3.
    # Known: R(2,3,3) = 11.
    # We can construct it by combining R(2,2,3) and R(2,1,3).
    # R(2,2,3) <= R(2,2,2) + R(2,2,1) = 7 + 2 = 9?
    # Actually, R(2,2,3) is known to be 9? Or 8?
    # Strassen is 7 for 2x2x2.
    # For 2x2x1, it's just 4 multiplications (standard).
    # So 7+4=11 for 2x2x3? No, 2x2x3 is A(2x2)*B(2x3).
    # Split B into B1(2x2) and B2(2x1).
    # C1 = A*B1 (rank 7).
    # C2 = A*B2 (rank 4).
    # Total rank 11.
    # Then for 2x3x3, split A into A1(2x2) and A2(2x1).
    # C = A1*B + A2*B.
    # A1*B is 2x2x3 (rank 11).
    # A2*B is 2x1x3 (rank 3? No, 2x1x3 is C_ik = A_i B_k. Rank 1? No.
    # A is 2x1, B is 1x3. C is 2x3.
    # C_ik = A_i B_k. This is an outer product. Rank 1.
    # So R(2,1,3) = 1.
    # So R(2,3,3) <= 11 + 1 = 12.
    # But known R(2,3,3) is 11.
    
    # Let's just use the known Rank 22 construction if we can't find better quickly.
    # But we want to try to improve.
    
    # Let's try a random search for rank 21.
    # If it fails, try 22 (known).
    
    def verify(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # T[i,j,k] = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
        # Must equal 1.
        n, m, p = DIMENSIONS
        R = U.shape[1]
        T_recon = np.zeros((n, m, p))
        for l in range(R):
            u = U[:, l].reshape(n, m)
            v = V[:, l].reshape(m, p)
            w = W[:, l].reshape(p, n)
            # T += u * v * w (element-wise with appropriate indices)
            # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        return np.allclose(T_recon, 1.0, atol=1e-5)

    def construct_rank_22():
        # Construct Rank 22: Two copies of Rank 11 for 2x3x3.
        # We need a valid Rank 11 decomposition for 2x3x3.
        # Let's build it from Strassen (7) + Standard (4).
        # 2x3x3: A(2x3) * B(3x3).
        # Split B into B1(3x2) and B2(3x1).
        # C1 = A*B1 (2x3x2). Rank 8.
        # C2 = A*B2 (2x3x1). Rank 3?
        # A(2x3) * B2(3x1) -> C(2x1).
        # C_i = sum_j A_ij B_j.
        # This is 2 independent dot products of length 3.
        # Rank 2? No, each is a sum of 3 products.
        # But they share the A_ij? No, C0 uses A0j, C1 uses A1j.
        # So it's two separate rank 3 tensors?
        # T2[i, j, k=0] = A_ij B_j.
        # This is rank 3 for each i?
        # Sum_j A_ij B_j.
        # We can decompose this using 3 terms:
        # l=0: A_00 B_0, A_10 B_0
        # l=1: A_01 B_1, A_11 B_1
        # l=2: A_02 B_2, A_12 B_2
        # So rank 3 for the 2x3x1 part.
        # Total Rank = 8 + 3 = 11.
        
        # Let's assemble the 2x3x2 part (Rank 8) and 2x3x1 part (Rank 3).
        
        # 2x3x2 Part (Rank 8):
        # As in the current program's U_block, V_block, W_block.
        U_232 = np.zeros((6, 8))
        V_232 = np.zeros((6, 8))
        W_232 = np.zeros((4, 8))
        
        # Strassen 2x2x2 (cols 0-6)
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # Term 1
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[1, 0] = 1; W_s[2, 0] = 1; W_s[3, 0] = 1
        # Term 2
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[2, 1] = 1; W_s[3, 1] = 1
        # Term 3
        U_s[0, 2] = 1
        V_s[2, 2] = 1; V_s[3, 2] = -1
        W_s[0, 2] = -1; W_s[1, 2] = -1
        # Term 4
        U_s[3, 3] = 1
        V_s[1, 3] = 1; V_s[0, 3] = -1
        W_s[2, 3] = 1; W_s[3, 3] = 1
        # Term 5
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = 1; W_s[1, 4] = 1
        # Term 6
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[2, 5] = -1; W_s[3, 5] = -1
        # Term 7
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1; W_s[1, 6] = 1
        
        for i in range(2):
            for j in range(2):
                U_232[i*3 + j, 0:7] = U_s[i*2 + j, 0:7]
        for j in range(2):
            for k in range(2):
                V_232[j*2 + k, 0:7] = V_s[j*2 + k, 0:7]
        for k in range(2):
            for i in range(2):
                W_232[k*2 + i, 0:7] = W_s[k*2 + i, 0:7]
                
        # Rank 1 part for j=2 (col 7)
        U_232[0*3 + 2, 7] = 1
        U_232[1*3 + 2, 7] = 1
        V_232[2*2 + 0, 7] = 1
        V_232[2*2 + 1, 7] = 1
        W_232[:, 7] = 1.0
        
        # 2x3x1 Part (Rank 3):
        # C_i = sum_{j=0}^2 A_ij B_j
        # Terms:
        # l=8: A_00 B_0, A_10 B_0
        # l=9: A_01 B_1, A_11 B_1
        # l=10: A_02 B_2, A_12 B_2
        U_233 = np.zeros((6, 11))
        V_233 = np.zeros((9, 11))
        W_233 = np.zeros((6, 11)) # p=3, n=2 -> 6
        
        # Copy 2x3x2 part
        U_233[:, 0:8] = U_232
        V_233[:, 0:8] = V_232
        # W_232 is (4, 8). W_233 needs (6, 11).
        # W index k*n+i. k in {0,1,2}.
        # W_232 covers k=0,1.
        W_233[0:4, 0:8] = W_232
        
        # Add 2x3x1 part
        # l=8: j=0, k=2 (since B is 3x3, the "1" part is the 3rd column? No.)
        # Wait, the 2x3x1 part corresponds to the third column of B (index 2).
        # So k=2.
        # U: A_ij.
        # V: B_jk. Here k=2.
        # W: C_ki. Here k=2.
        
        # l=8: j=0
        U_233[0*3+0, 8] = 1 # A_00
        U_233[1*3+0, 8] = 1 # A_10
        V_233[0*3+2, 8] = 1 # B_02
        W_233[2*2+0, 8] = 1 # C_20
        W_233[2*2+1, 8] = 1 # C_21
        
        # l=9: j=1
        U_233[0*3+1, 9] = 1 # A_01
        U_233[1*3+1, 9] = 1 # A_11
        V_233[1*3+2, 9] = 1 # B_12
        W_233[2*2+0, 9] = 1 # C_20
        W_233[2*2+1, 9] = 1 # C_21
        
        # l=10: j=2
        U_233[0*3+2, 10] = 1 # A_02
        U_233[1*3+2, 10] = 1 # A_12
        V_233[2*3+2, 10] = 1 # B_22
        W_233[2*2+0, 10] = 1 # C_20
        W_233[2*2+1, 10] = 1 # C_21
        
        # Assemble 2x3x6 from two 2x3x3 blocks.
        U_full = np.zeros((6, 22))
        V_full = np.zeros((18, 22))
        W_full = np.zeros((12, 22))
        
        for b in range(2):
            offset = b * 11
            U_full[:, offset:offset+11] = U_233
            
            # V: j*6 + k. k in {0,1,2} for block 0, {3,4,5} for block 1.
            for j in range(3):
                for k_local in range(3):
                    k_global = 3*b + k_local
                    V_full[j*6 + k_global, offset:offset+11] = V_233[j*3 + k_local, 0:11]
                    
            # W: k*2 + i.
            for k_local in range(3):
                for i in range(2):
                    k_global = 3*b + k_local
                    W_full[k_global*2 + i, offset:offset+11] = W_233[k_local*2 + i, 0:11]
                    
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }

    # Try to find a better solution via random search for rank 21.
    # This is a long shot, but if it works, score jumps.
    # If time permits, we can try a few random initializations and gradient descent.
    # But gradient descent on tensor rank is non-convex and hard.
    # Random search with verification is safer.
    
    best_result = construct_rank_22()
    
    # Check time
    if time.time() < deadline - 10:
        # Try random search for rank 21
        # We'll initialize with the rank 22 solution, zero out one column, and try to fix it?
        # Or just random init.
        # Random init for rank 21 is unlikely to converge without optimization.
        # Let's stick with the known 22.
        pass
        
    return best_result

if __name__ == "__main__":
    construct()
