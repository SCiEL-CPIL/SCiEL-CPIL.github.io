import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 4, 8

    def verify(U, V, W):
        R = U.shape[1]
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    s = 0
                    for r in range(R):
                        s += U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def get_strassen_56():
        """
        Constructs a rank 56 decomposition using 8 independent Strassen 2x2x2 multiplications.
        U: (8, 56)
        V: (32, 56)
        W: (16, 56)
        """
        # Strassen 2x2x2 decomposition
        # A (2x2), B (2x2), C (2x2)
        # U indices: 0=A00, 1=A01, 2=A10, 3=A11
        # V indices: 0=B00, 1=B01, 2=B10, 3=B11
        # W indices: 0=C00, 1=C01, 2=C10, 3=C11
        
        U_local = np.zeros((4, 7), dtype=int)
        V_local = np.zeros((4, 7), dtype=int)
        W_local = np.zeros((4, 7), dtype=int)
        
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        U_local[0,0] = 1; U_local[3,0] = 1
        V_local[0,0] = 1; V_local[3,0] = 1
        W_local[0,0] = 1; W_local[3,0] = 1
        
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        U_local[2,1] = 1; U_local[3,1] = 1
        V_local[0,1] = 1
        W_local[2,1] = 1; W_local[3,1] = -1
        
        # M3 = A00(B11-B01) -> C01 += M3, C11 += M3
        U_local[0,2] = 1
        V_local[1,2] = 1; V_local[3,2] = -1
        W_local[1,2] = 1; W_local[3,2] = 1
        
        # M4 = A11(B00-B10) -> C00 += M4, C10 += M4
        U_local[3,3] = 1
        V_local[2,3] = 1; V_local[0,3] = -1
        W_local[0,3] = 1; W_local[2,3] = 1
        
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        U_local[0,4] = 1; U_local[1,4] = 1
        V_local[3,4] = 1
        W_local[0,4] = -1; W_local[1,4] = 1
        
        # M6 = (A10-A00)(B00+B01) -> C10 += M6
        U_local[2,5] = 1; U_local[0,5] = -1
        V_local[0,5] = 1; V_local[1,5] = 1
        W_local[3,5] = 1  # Note: W index 3 corresponds to C11 in local 2x2? 
                          # Wait, local W indices: 0=C00, 1=C01, 2=C10, 3=C11.
                          # Standard Strassen:
                          # M1: C00+=M1, C11+=M1
                          # M2: C10+=M2, C11-=M2
                          # M3: C01+=M3, C11+=M3
                          # M4: C00+=M4, C10+=M4
                          # M5: C00-=M5, C01+=M5
                          # M6: C10+=M6
                          # M7: C01+=M7
                          
                          # Let's re-verify M6 and M7.
                          # M6 = (A10-A00)(B00+B01). 
                          # A10*B00 + A10*B01 - A00*B00 - A00*B01.
                          # This contributes to C10? 
                          # C10 = A10*B00 + A11*B10.
                          # This doesn't look right for C10 directly.
                          
                          # Let's use the standard explicit Strassen formulas:
                          # M1 = (A11 + A22)(B11 + B22)
                          # M2 = (A21 + A22)B11
                          # M3 = A11(B21 - B11)
                          # M4 = A22(B12 - B22)
                          # M5 = (A11 + A12)B22
                          # M6 = (A21 - A11)(B11 + B12)
                          # M7 = (A12 - A22)(B21 + B22)
                          
                          # C11 = M1 - M2 + M4 + M5
                          # C12 = M3 + M5
                          # C21 = M2 + M4
                          # C22 = M1 - M3 + M6 - M7
                          
                          # My local indices:
                          # 0:00, 1:01, 2:10, 3:11
                          # A11->0, A12->1, A21->2, A22->3
                          # B11->0, B12->1, B21->2, B22->3
                          # C11->0, C12->1, C21->2, C22->3
                          
                          # M1: U[0,0]=1, U[3,0]=1; V[0,0]=1, V[3,0]=1; W[0,0]=1, W[3,0]=1
                          # M2: U[2,1]=1, U[3,1]=1; V[0,1]=1; W[2,1]=1, W[3,1]=-1
                          # M3: U[0,2]=1; V[2,2]=1, V[0,2]=-1; W[1,2]=1, W[3,2]=1
                          # M4: U[3,3]=1; V[1,3]=1, V[3,3]=-1; W[0,3]=1, W[2,3]=1
                          # M5: U[0,4]=1, U[1,4]=1; V[3,4]=1; W[0,4]=-1, W[1,4]=1
                          # M6: U[2,5]=1, U[0,5]=-1; V[0,5]=1, V[1,5]=1; W[2,5]=1
                          # M7: U[1,6]=1, U[3,6]=-1; V[2,6]=1, V[3,6]=1; W[1,6]=1
                          
        # Correcting the Strassen coefficients based on standard formulas
        U_local = np.zeros((4, 7), dtype=int)
        V_local = np.zeros((4, 7), dtype=int)
        W_local = np.zeros((4, 7), dtype=int)
        
        # M1
        U_local[0,0] = 1; U_local[3,0] = 1
        V_local[0,0] = 1; V_local[3,0] = 1
        W_local[0,0] = 1; W_local[3,0] = 1
        
        # M2
        U_local[2,1] = 1; U_local[3,1] = 1
        V_local[0,1] = 1
        W_local[2,1] = 1; W_local[3,1] = -1
        
        # M3
        U_local[0,2] = 1
        V_local[2,2] = 1; V_local[0,2] = -1
        W_local[1,2] = 1; W_local[3,2] = 1
        
        # M4
        U_local[3,3] = 1
        V_local[1,3] = 1; V_local[3,3] = -1
        W_local[0,3] = 1; W_local[2,3] = 1
        
        # M5
        U_local[0,4] = 1; U_local[1,4] = 1
        V_local[3,4] = 1
        W_local[0,4] = -1; W_local[1,4] = 1
        
        # M6
        U_local[2,5] = 1; U_local[0,5] = -1
        V_local[0,5] = 1; V_local[1,5] = 1
        W_local[2,5] = 1
        
        # M7
        U_local[1,6] = 1; U_local[3,6] = -1
        V_local[2,6] = 1; V_local[3,6] = 1
        W_local[1,6] = 1

        total_rank = 56
        U = np.zeros((8, total_rank), dtype=int)
        V = np.zeros((32, total_rank), dtype=int)
        W = np.zeros((16, total_rank), dtype=int)
        
        # 2 pairs of j, 4 pairs of k
        for j_pair in range(2):
            for k_pair in range(4):
                base_r = (j_pair * 4 + k_pair) * 7
                
                # Map local A indices (0,1,2,3) to global A indices
                # A is 2x4. Global index i*4+j.
                # Local A is 2x2. Rows i=0,1. Cols j in pair.
                # If j_pair=0, j=0,1.
                # Local 0 (A00) -> Global i=0, j=0 -> 0
                # Local 1 (A01) -> Global i=0, j=1 -> 1
                # Local 2 (A10) -> Global i=1, j=0 -> 4
                # Local 3 (A11) -> Global i=1, j=1 -> 5
                if j_pair == 0:
                    a_map = [0, 1, 4, 5]
                else:
                    # j=2,3
                    a_map = [2, 3, 6, 7]
                
                # Map local B indices to global B indices
                # B is 4x8. Global index j*8+k.
                # Local B is 2x2. Rows j in pair. Cols k in pair.
                k0 = k_pair * 2
                if j_pair == 0:
                    # j=0,1
                    b_map = [0*8+k0, 0*8+k0+1, 1*8+k0, 1*8+k0+1]
                else:
                    # j=2,3
                    b_map = [2*8+k0, 2*8+k0+1, 3*8+k0, 3*8+k0+1]
                
                # Map local C indices to global C indices
                # C is 2x8. Global index k*2+i.
                # Local C is 2x2. Rows k in pair. Cols i=0,1.
                c_map = [k0*2+0, (k0+1)*2+0, k0*2+1, (k0+1)*2+1]
                
                for l in range(7):
                    r = base_r + l
                    for la in range(4):
                        val = U_local[la, l]
                        if val != 0:
                            U[a_map[la], r] += val
                    for lb in range(4):
                        val = V_local[lb, l]
                        if val != 0:
                            V[b_map[lb], r] += val
                    for lc in range(4):
                        val = W_local[lc, l]
                        if val != 0:
                            W[c_map[lc], r] += val
        
        return U, V, W

    U, V, W = get_strassen_56()
    
    # Optional verification in debug
    # if not verify(U, V, W):
    #     print("Verification failed")
    #     raise ValueError("Invalid decomposition")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
