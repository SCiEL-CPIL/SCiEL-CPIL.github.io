import numpy as np
import random
import time

DIMENSIONS = [2, 4, 5]

def construct():
    # Target: 2x4 * 4x5
    n, m, p = DIMENSIONS
    
    # Strategy: Split A into A_L (2x2) and A_R (2x2)
    # Split B into B_T (2x5) and B_B (2x5)
    # C = A_L B_T + A_R B_B
    # This requires two independent 2x2x5 multiplications.
    # Rank = 2 * R(2,2,5)
    
    # We need a decomposition for 2x2x5.
    # Let's try to find a rank-16 decomposition for 2x2x5.
    # If we find rank 12, we get total rank 24.
    # If we find rank 16, we get total rank 32.
    # Fallback: Rank 18 (2+2+1 split) -> Total 36.
    
    def find_225_decomposition(max_rank=16):
        # Tensor T for 2x2 * 2x5
        # U: 4 x r, V: 10 x r, W: 10 x r
        # T[i,j,k] = delta(i, ?) no.
        # C[i,k] = sum_j A[i,j] B[j,k]
        # i in 0,1; j in 0,1; k in 0..4
        # T is a 4x10x10 tensor? No.
        # U rows: A indices (4)
        # V rows: B indices (10)
        # W rows: C indices (10)
        
        # The tensor entries T[u, v, w] are 1 if u corresponds to A[i,j], v to B[j,k], w to C[i,k].
        # A[i,j] index: i*2+j
        # B[j,k] index: j*5+k
        # C[i,k] index: i*5+k
        
        T = np.zeros((4, 10, 10))
        for i in range(2):
            for j in range(2):
                for k in range(5):
                    u = i * 2 + j
                    v = j * 5 + k
                    w = i * 5 + k
                    T[u, v, w] = 1.0
                    
        # Try to find rank r decomposition
        # Use random initialization and gradient descent or similar?
        # Or use a known construction if available.
        # For now, we will return a known rank-18 construction (2+2+1)
        # and try to improve it if time permits.
        
        # Rank 18 Construction:
        # Split k into k1 (0,1), k2 (2,3), k3 (4)
        # C1 = A B1 (2x2x2), C2 = A B2 (2x2x2), C3 = A B3 (2x2x1)
        # Rank 7 + 7 + 4 = 18.
        
        # We can hardcode the rank-18 decomposition.
        # Or we can try to search for rank 16.
        
        # Given the time constraint and the need for a robust answer, 
        # I will implement the rank-36 solution (2 * 18) first.
        # Then I will add a search for rank-16 (total 32) or rank-12 (total 24).
        
        pass

    # Implementing the Rank-36 solution (2 * Rank-18)
    # Rank-18 for 2x2x5:
    # We need U (4x18), V (10x18), W (10x18)
    
    # Let's build the rank-7 decomposition for 2x2x2 (Strassen)
    # Strassen for 2x2x2:
    # M1 = (a+d)(x+y) -> no, M1 = a x, M2 = a y, M3 = b x, M4 = b y, M5 = c x, M6 = c y, M7 = d x, M8 = d y
    # Strassen:
    # P1 = (a+d)(x+y)
    # P2 = (c+d)x
    # P3 = a(y-z)
    # P4 = d(x-w)
    # P5 = (a+b)w
    # P6 = (c+d)(z-w)
    # P7 = (a-c)(x+w)
    # C11 = P1 + P4 - P5 - P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P6 - P7
    
    # We need to map this to the tensor format.
    # For 2x2x2:
    # U (4x7), V (8x7), W (8x7)
    # A indices: 0(a), 1(b), 2(c), 3(d)
    # B indices: 0(x), 1(y), 2(z), 3(w)  (Row-major: x y / z w? No, B is 2x2. B[0,0]=x, B[0,1]=y, B[1,0]=z, B[1,1]=w)
    # C indices: 0(C11), 1(C12), 2(C21), 3(C22)
    
    # Let's create the Strassen matrices for 2x2x2.
    def get_strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((8, 7))
        W = np.zeros((8, 7))
        
        # P1 = (a+d)(x+y)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[1, 0] = 1
        W[0, 0] = 1; W[3, 0] = -1 # C11 gets +, C22 gets -
        
        # P2 = (c+d)x
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1 # C21 gets +, C22 gets -
        
        # P3 = a(y-z)
        U[0, 2] = 1
        V[1, 2] = 1; V[2, 2] = -1
        W[1, 2] = 1
        
        # P4 = d(x-w)
        U[3, 3] = 1
        V[0, 3] = 1; V[3, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # P5 = (a+b)w
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # P6 = (c+d)(z-w)
        U[2, 5] = 1; U[3, 5] = 1
        V[2, 5] = 1; V[3, 5] = -1
        W[3, 5] = 1
        
        # P7 = (a-c)(x+w)
        U[0, 6] = 1; U[2, 6] = -1
        V[0, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1; W[3, 6] = -1
        
        return U, V, W

    # Now build 2x2x5 using 2x2x2 (for k=0,1), 2x2x2 (for k=2,3), 2x2x1 (for k=4)
    # 2x2x1 rank is 4.
    # U (4x4), V (2x4), W (2x4)
    # C1 = a x + b y
    # C2 = c x + d y
    # M1 = a x, M2 = b y, M3 = c x, M4 = d y
    def get_221():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4)) # B is 2x1, so 2 entries
        W = np.zeros((2, 4)) # C is 2x1, so 2 entries
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1 # a x -> C1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1 # b y -> C1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1 # c x -> C2
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1 # d y -> C2
        return U, V, W

    # Combine
    U1, V1, W1 = get_strassen_222()
    U2, V2, W2 = get_strassen_222()
    U3, V3, W3 = get_221()
    
    U_225 = np.hstack([U1, U2, U3])
    # V indices:
    # V1: B[0,0], B[0,1], B[1,0], B[1,1] correspond to k=0,1
    # V2: B[0,2], B[0,3], B[1,2], B[1,3] correspond to k=2,3
    # V3: B[0,4], B[1,4] correspond to k=4
    
    # We need to map V1, V2, V3 to the global 10x18 V matrix.
    # Global B indices:
    # 0: B[0,0], 1: B[0,1], 2: B[0,2], 3: B[0,3], 4: B[0,4]
    # 5: B[1,0], 6: B[1,1], 7: B[1,2], 8: B[1,3], 9: B[1,4]
    
    V_225 = np.zeros((10, 18))
    # V1 (8x7)
    # V1[0] is B[0,0] (global 0)
    # V1[1] is B[0,1] (global 1)
    # V1[2] is B[1,0] (global 5)
    # V1[3] is B[1,1] (global 6)
    # V1[4] is B[0,0]? No, Strassen V is 8x7?
    # In get_strassen_222, V is 8x7.
    # Indices 0,1,2,3 correspond to x,y,z,w?
    # x=B[0,0], y=B[0,1], z=B[1,0], w=B[1,1].
    # So V1[0] -> Global 0
    # V1[1] -> Global 1
    # V1[2] -> Global 5
    # V1[3] -> Global 6
    # V1[4..7] are zeros?
    # In my get_strassen_222, V is 8x7.
    # I only used rows 0,1,2,3.
    # So V1[0:4, :] are non-zero.
    
    V_225[0, 0:7] = V1[0, :]
    V_225[1, 0:7] = V1[1, :]
    V_225[5, 0:7] = V1[2, :]
    V_225[6, 0:7] = V1[3, :]
    
    # V2 (8x7) for k=2,3
    # x=B[0,2] (Global 2), y=B[0,3] (Global 3), z=B[1,2] (Global 7), w=B[1,3] (Global 8)
    V_225[2, 7:14] = V2[0, :]
    V_225[3, 7:14] = V2[1, :]
    V_225[7, 7:14] = V2[2, :]
    V_225[8, 7:14] = V2[3, :]
    
    # V3 (2x4) for k=4
    # x=B[0,4] (Global 4), y=B[1,4] (Global 9)
    V_225[4, 14:18] = V3[0, :]
    V_225[9, 14:18] = V3[1, :]
    
    # W indices:
    # W1: C[0,0], C[0,1], C[1,0], C[1,1] (Global 0,1,5,6)
    # W2: C[0,2], C[0,3], C[1,2], C[1,3] (Global 2,3,7,8)
    # W3: C[0,4], C[1,4] (Global 4,9)
    
    W_225 = np.zeros((10, 18))
    W_225[0, 0:7] = W1[0, :]
    W_225[1, 0:7] = W1[1, :]
    W_225[5, 0:7] = W1[2, :]
    W_225[6, 0:7] = W1[3, :]
    
    W_225[2, 7:14] = W2[0, :]
    W_225[3, 7:14] = W2[1, :]
    W_225[7, 7:14] = W2[2, :]
    W_225[8, 7:14] = W2[3, :]
    
    W_225[4, 14:18] = W3[0, :]
    W_225[9, 14:18] = W3[1, :]
    
    return U_225, V_225, W_225

    U_225, V_225, W_225 = find_225_decomposition()
    
    # Now combine two 2x2x5 decompositions for the full 2x4x5
    # A = [A_L, A_R]
    # B = [B_T; B_B]
    # C = A_L B_T + A_R B_B
    
    # U_full (8x36)
    # U_L (4x18) for A_L
    # U_R (4x18) for A_R
    U_full = np.zeros((8, 36))
    U_full[0:4, 0:18] = U_225
    U_full[4:8, 18:36] = U_225
    
    # V_full (20x36)
    # V_T (10x18) for B_T
    # V_B (10x18) for B_B
    V_full = np.zeros((20, 36))
    V_full[0:10, 0:18] = V_225
    V_full[10:20, 18:36] = V_225
    
    # W_full (10x36)
    # W is shared?
    # C = A_L B_T + A_R B_B
    # The output C is the same.
    # So W_full should be the sum?
    # No, the decomposition is C = sum_k U_k A * V_k B * W_k.
    # Here we have two sets of terms.
    # C = sum_{k=1}^{18} (U_k A_L) (V_k B_T) W_k + sum_{k=19}^{36} (U_{k-18} A_R) (V_{k-18} B_B) W_{k-18}
    # So W_full is just [W_225, W_225]?
    # No, the W matrix maps the intermediate products to C.
    # The intermediate product for term k is a scalar.
    # So W is 10x36.
    W_full = np.zeros((10, 36))
    W_full[:, 0:18] = W_225
    W_full[:, 18:36] = W_225
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

