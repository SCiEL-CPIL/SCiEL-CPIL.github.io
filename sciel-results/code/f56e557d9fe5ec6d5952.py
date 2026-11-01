"""Matrix multiplication 2,2,4 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8

    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        # Check all 4*8*8 = 256 tensor entries
        for i in range(nm):
            for j in range(mp):
                for k in range(pn):
                    expected = 1.0 if (i % m == j // p and k % n == i // m) else 0.0
                    s = 0.0
                    for r in range(R):
                        s += U[i][r] * V[j][r] * W[k][r]
                    if abs(s - expected) > 1e-9:
                        return False
        return True

    def build_rank14():
        """Standard rank-14: two independent Strassen 2x2 blocks."""
        R = 14
        U = [[0]*R for _ in range(nm)]
        V = [[0]*R for _ in range(mp)]
        W = [[0]*R for _ in range(pn)]
        
        # Left block (B cols 0,1): Strassen terms r=0..6
        # A: 0=a00, 1=a01, 2=a10, 3=a11
        # B: 0=b00, 1=b01, 4=b10, 5=b11
        # C: 0=C00, 1=C10, 2=C01, 3=C11
        
        # M1 = (a00+a11)(b00+b11)
        U[0][0]=1; U[3][0]=1
        V[0][0]=1; V[5][0]=1
        W[0][0]=1; W[3][0]=1
        # M2 = (a10+a11)b00
        U[2][1]=1; U[3][1]=1
        V[0][1]=1
        W[1][1]=1; W[3][1]=-1
        # M3 = a00(b01-b11)
        U[0][2]=1
        V[1][2]=1; V[5][2]=-1
        W[2][2]=1; W[3][2]=1
        # M4 = a11(b10-b00)
        U[3][3]=1
        V[4][3]=1; V[0][3]=-1
        W[0][3]=1; W[1][3]=1
        # M5 = (a00+a01)b11
        U[0][4]=1; U[1][4]=1
        V[5][4]=1
        W[0][4]=-1; W[2][4]=1
        # M6 = (a10-a00)(b00+b01)
        U[2][5]=1; U[0][5]=-1
        V[0][5]=1; V[1][5]=1
        W[3][5]=1
        # M7 = (a01-a11)(b10+b11)
        U[1][6]=1; U[3][6]=-1
        V[4][6]=1; V[5][6]=1
        W[0][6]=1

        # Right block (B cols 2,3): Strassen terms r=7..13
        # B: 2=b02, 3=b03, 6=b12, 7=b13
        # C: 4=C02, 5=C12, 6=C03, 7=C13
        
        # N1 = (a00+a11)(b02+b13)
        U[0][7]=1; U[3][7]=1
        V[2][7]=1; V[7][7]=1
        W[4][7]=1; W[7][7]=1
        # N2 = (a10+a11)b02
        U[2][8]=1; U[3][8]=1
        V[2][8]=1
        W[5][8]=1; W[7][8]=-1
        # N3 = a00(b03-b13)
        U[0][9]=1
        V[3][9]=1; V[7][9]=-1
        W[6][9]=1; W[7][9]=1
        # N4 = a11(b12-b02)
        U[3][10]=1
        V[6][10]=1; V[2][10]=-1
        W[4][10]=1; W[5][10]=1
        # N5 = (a00+a01)b13
        U[0][11]=1; U[1][11]=1
        V[7][11]=1
        W[4][11]=-1; W[6][11]=1
        # N6 = (a10-a00)(b02+b03)
        U[2][12]=1; U[0][12]=-1
        V[2][12]=1; V[3][12]=1
        W[7][12]=1
        # N7 = (a01-a11)(b12+b13)
        U[1][13]=1; U[3][13]=-1
        V[6][13]=1; V[7][13]=1
        W[4][13]=1

        return {"U": U, "V": V, "W": W}

    def try_rank13():
        """
        Search for rank-13 decomposition.
        
        Strategy: The tensor T[i,j,k] = 1 iff i%m == j//p and k%n == i//m.
        For 2x2x4, we have 16 nonzero entries. A rank-R decomposition means
        T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r.
        
        Key structural observation: The A-factors (U) are shared between the
        left and right blocks. If we can find a rank-1 term whose U vector
        works for both blocks, we can merge.
        
        We parameterize: pick 13 U vectors (from a candidate set), then solve
        for V and W via linear algebra.
        """
        if time.time() > deadline - 20:
            return None
        
        # Candidate U vectors: all nonzero vectors in {-1,0,1}^4 with at least 2 nonzero entries
        # (to be useful for Strassen-like combinations)
        candidate_U = []
        for u0 in [-1, 0, 1]:
            for u1 in [-1, 0, 1]:
                for u2 in [-1, 0, 1]:
                    for u3 in [-1, 0, 1]:
                        vec = [u0, u1, u2, u3]
                        if sum(1 for x in vec if x != 0) >= 2:
                            candidate_U.append(tuple(vec))
        
        # For rank 13, we need to choose 13 U vectors from candidates.
        # This is combinatorially expensive, so use a smarter approach:
        # Start from the rank-14 solution and try to merge pairs.
        
        # In rank-14, the U vectors are:
        # r=0: (1,0,0,1), r=1: (0,0,1,1), r=2: (1,0,0,0), r=3: (0,0,0,1),
        # r=4: (1,1,0,0), r=5: (1,-1,0,0), r=6: (0,1,-1,0)
        # r=7: (1,0,0,1), r=8: (0,0,1,1), r=9: (1,0,0,0), r=10: (0,0,0,1),
        # r=11: (1,1,0,0), r=12: (1,-1,0,0), r=13: (0,1,-1,0)
        
        # Notice: r=0 and r=7 have the same U vector (1,0,0,1)!
        # Similarly r=1 and r=8: (0,0,1,1), r=2 and r=9: (1,0,0,0), etc.
        # This means we CAN merge them if the V and W vectors are compatible.
        
        # For a merged term at rank position r, we need:
        # U[r] = U_left[r'] = U_right[r'']  (same U vector)
        # V[r] must work for both left and right B indices
        # W[r] must work for both left and right C indices
        
        # Let's try all 7x7 = 49 possible merges of left term li with right term ri
        # where U_left[li] == U_right[ri]
        
        # Left Strassen U vectors (indexed 0-6):
        left_U = [
            (1,0,0,1),   # M1
            (0,0,1,1),   # M2
            (1,0,0,0),   # M3
            (0,0,0,1),   # M4
            (1,1,0,0),   # M5
            (1,-1,0,0),  # M6
            (0,1,-1,0),  # M7
        ]
        # Left V vectors (for B indices 0,1,4,5):
        left_V = [
            (1,1,0,0),   # M1: b00+b11 -> V[0]=1, V[5]=1
            (1,0,0,0),   # M2: b00
            (0,1,0,-1),  # M3: b01-b11 -> V[1]=1, V[5]=-1
            (-1,0,1,0),  # M4: b10-b00 -> V[0]=-1, V[4]=1
            (0,0,0,1),   # M5: b11
            (1,1,0,0),   # M6: b00+b01
            (0,0,1,1),   # M7: b10+b11
        ]
        # Left W vectors (for C indices 0,1,2,3):
        left_W = [
            (1,0,0,1),   # M1: C00+C11
            (0,1,0,-1),  # M2: C10-C11
            (0,0,1,1),   # M3: C01+C11
            (1,1,0,0),   # M4: C00+C10
            (-1,0,1,0),  # M5: -C00+C01
            (0,0,0,1),   # M6: C11
            (1,0,0,0),   # M7: C00
        ]
        
        # Right Strassen U vectors (indexed 7-13, same as left):
        right_U = list(left_U)
        # Right V vectors (for B indices 2,3,6,7):
        right_V = [
            (1,1,0,0),   # N1: b02+b13 -> V[2]=1, V[7]=1
            (1,0,0,0),   # N2: b02
            (0,1,0,-1),  # N3: b03-b13 -> V[3]=1, V[7]=-1
            (-1,0,1,0),  # N4: b12-b02 -> V[2]=-1, V[6]=1
            (0,0,0,1),   # N5: b13
            (1,1,0,0),   # N6: b02+b03
            (0,0,1,1),   # N7: b12+b13
        ]
        # Right W vectors (for C indices 4,5,6,7):
        right_W = [
            (1,0,0,1),   # N1: C02+C13
            (0,1,0,-1),  # N2: C12-C13
            (0,0,1,1),   # N3: C03+C13
            (1,1,0,0),   # N4: C02+C12
            (-1,0,1,0),  # N5: -C02+C03
            (0,0,0,1),   # N6: C13
            (1,0,0,0),   # N7: C02
        ]
        
        # For each pair (li, ri) where left_U[li] == right_U[ri], try to merge
        for li in range(7):
            if time.time() > deadline - 5:
                return None
            for ri in range(7):
                if left_U[li] != right_U[ri]:
                    continue
                
                # Build the 13-term decomposition
                R = 13
                U_mat = np.zeros((4, R))
                V_mat = np.zeros((8, R))
                W_mat = np.zeros((8, R))
                
                # Merged term at position 0
                U_mat[:, 0] = left_U[li]
                # V: left part in indices 0,1,4,5; right part in 2,3,6,7
                V_mat[0, 0] = left_V[li][0]   # b00
                V_mat[1, 0] = left_V[li][1]   # b01
                V_mat[4, 0] = left_V[li][2]   # b10
                V_mat[5, 0] = left_V[li][3]   # b11
                V_mat[2, 0] = right_V[ri][0]  # b02
                V_mat[3, 0] = right_V[ri][1]  # b03
                V_mat[6, 0] = right_V[ri][2]  # b12
                V_mat[7, 0] = right_V[ri][3]  # b13
                # W: left part in indices 0,1,2,3; right part in 4,5,6,7
                W_mat[0, 0] = left_W[li][0]   # C00
                W_mat[1, 0] = left_W[li][1]   # C10
                W_mat[2, 0] = left_W[li][2]   # C01
                W_mat[3, 0] = left_W[li][3]   # C11
                W_mat[4, 0] = right_W[ri][0]  # C02
                W_mat[5, 0] = right_W[ri][1]  # C12
                W_mat[6, 0] = right_W[ri][2]  # C03
                W_mat[7, 0] = right_W[ri][3]  # C13
                
                # Remaining left terms (excluding li)
                col = 1
                for l_idx in range(7):
                    if l_idx == li:
                        continue
                    U_mat[:, col] = left_U[l_idx]
                    V_mat[0, col] = left_V[l_idx][0]
                    V_mat[1, col] = left_V[l_idx][1]
                    V_mat[4, col] = left_V[l_idx][2]
                    V_mat[5, col] = left_V[l_idx][3]
                    W_mat[0, col] = left_W[l_idx][0]
                    W_mat[1, col] = left_W[l_idx][1]
                    W_mat[2, col] = left_W[l_idx][2]
                    W_mat[3, col] = left_W[l_idx][3]
                    col += 1
                
                # Remaining right terms (excluding ri)
                for r_idx in range(7):
                    if r_idx == ri:
                        continue
                    U_mat[:, col] = right_U[r_idx]
                    V_mat[2, col] = right_V[r_idx][0]
                    V_mat[3, col] = right_V[r_idx][1]
                    V_mat[6, col] = right_V[r_idx][2]
                    V_mat[7, col] = right_V[r_idx][3]
                    W_mat[4, col] = right_W[r_idx][0]
                    W_mat[5, col] = right_W[r_idx][1]
                    W_mat[6, col] = right_W[r_idx][2]
                    W_mat[7, col] = right_W[r_idx][3]
                    col += 1
                
                # Verify
                U_list = U_mat.tolist()
                V_list = V_mat.tolist()
                W_list = W_mat.tolist()
                
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try rank-13 first
    result = try_rank13()
    if result is not None:
        return result
    
    # Fall back to rank-14
    return build_rank14()
# EVOLVE-BLOCK-END
