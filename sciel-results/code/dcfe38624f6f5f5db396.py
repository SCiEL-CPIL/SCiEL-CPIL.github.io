import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Get 2x2 Strassen factors
    U2, V2, W2 = get_2x2_strassen()
    
    # Build 4x4x4 factors
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Define coefficients for A and B in Strassen products
    # A_l is 2x2 matrix of coefficients for A elements
    # A_l[a,b] is a dict {global_idx: coeff}
    
    # M1 = (A00+A11)(B00+B11)
    A1 = { (0,0): {0:1, 3:1}, (0,1): {1:1, 4:1}, (1,0): {2:1, 5:1}, (1,1): {3:1, 6:1} } # Wait, A is 4x4.
    # A indices:
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # Let's re-verify A indices.
    # A is 4x4.
    # A00 (block 0,0): rows 0,1 cols 0,1. Indices: 0,1,4,5.
    # A01 (block 0,1): rows 0,1 cols 2,3. Indices: 2,3,6,7.
    # A10 (block 1,0): rows 2,3 cols 0,1. Indices: 8,9,12,13.
    # A11 (block 1,1): rows 2,3 cols 2,3. Indices: 10,11,14,15.
    
    # B indices:
    # B00: 0,1,8,9
    # B01: 2,3,10,11
    # B10: 16,17,24,25
    # B11: 18,19,26,27
    
    # M1
    A1_coeffs = { (0,0): {0:1, 10:1}, (0,1): {1:1, 11:1}, (1,0): {4:1, 14:1}, (1,1): {5:1, 15:1} }
    B1_coeffs = { (0,0): {0:1, 18:1}, (0,1): {1:1, 19:1}, (1,0): {8:1, 26:1}, (1,1): {9:1, 27:1} }
    
    # M2 = (A10+A11)B00
    A2_coeffs = { (0,0): {8:1, 10:1}, (0,1): {9:1, 11:1}, (1,0): {12:1, 14:1}, (1,1): {13:1, 15:1} }
    B2_coeffs = { (0,0): {0:1}, (0,1): {1:1}, (1,0): {8:1}, (1,1): {9:1} }
    
    # M3 = A00(B01-B11)
    A3_coeffs = { (0,0): {0:1}, (0,1): {1:1}, (1,0): {4:1}, (1,1): {5:1} }
    B3_coeffs = { (0,0): {2:1, 18:-1}, (0,1): {3:1, 19:-1}, (1,0): {10:1, 26:-1}, (1,1): {11:1, 27:-1} }
    
    # M4 = A11(B10-B00)
    A4_coeffs = { (0,0): {10:1}, (0,1): {11:1}, (1,0): {14:1}, (1,1): {15:1} }
    B4_coeffs = { (0,0): {16:1, 0:-1}, (0,1): {17:1, 1:-1}, (1,0): {24:1, 8:-1}, (1,1): {25:1, 9:-1} }
    
    # M5 = (A00+A01)B11
    A5_coeffs = { (0,0): {0:1, 2:1}, (0,1): {1:1, 3:1}, (1,0): {4:1, 6:1}, (1,1): {5:1, 7:1} }
    B5_coeffs = { (0,0): {18:1}, (0,1): {19:1}, (1,0): {26:1}, (1,1): {27:1} }
    
    # M6 = (A10-A00)(B00+B01)
    A6_coeffs = { (0,0): {8:1, 0:-1}, (0,1): {9:1, 1:-1}, (1,0): {12:1, 4:-1}, (1,1): {13:1, 5:-1} }
    B6_coeffs = { (0,0): {0:1, 2:1}, (0,1): {1:1, 3:1}, (1,0): {8:1, 10:1}, (1,1): {9:1, 11:1} }
    
    # M7 = (A01-A11)(B10+B11)
    A7_coeffs = { (0,0): {2:1, 10:-1}, (0,1): {3:1, 11:-1}, (1,0): {6:1, 14:-1}, (1,1): {7:1, 15:-1} }
    B7_coeffs = { (0,0): {16:1, 18:1}, (0,1): {17:1, 19:1}, (1,0): {24:1, 26:1}, (1,1): {25:1, 27:1} }
    
    A_list = [A1_coeffs, A2_coeffs, A3_coeffs, A4_coeffs, A5_coeffs, A6_coeffs, A7_coeffs]
    B_list = [B1_coeffs, B2_coeffs, B3_coeffs, B4_coeffs, B5_coeffs, B6_coeffs, B7_coeffs]
    
    # C coefficients
    # C_l is 2x2 matrix of coefficients for M_l elements
    # C_l[u,w] is dict {(a,d): coeff}
    
    C1_coeffs = { (0,0): {(0,0):1, (1,1):1}, (0,1): {}, (1,0): {}, (1,1): {} } # M1 in C00, C11
    # Wait, C00 = M1[0,0]... C11 = M1[1,1]...
    # So C1_coeffs[0,0] has (0,0):1. C1_coeffs[1,1] has (1,1):1.
    C1_coeffs = { (0,0): {(0,0):1}, (0,1): {}, (1,0): {}, (1,1): {(1,1):1} }
    
    C2_coeffs = { (0,0): {(0,0):-1}, (0,1): {}, (1,0): {(1,0):1}, (1,1): {} } # M2 in C00(-), C10(+)
    C3_coeffs = { (0,0): {}, (0,1): {(0,1):1}, (1,0): {}, (1,1): {(1,1):-1} } # M3 in C01(+), C11(-)
    C4_coeffs = { (0,0): {(0,0):1}, (0,1): {}, (1,0): {(1,0):1}, (1,1): {} } # M4 in C00(+), C10(+)
    C5_coeffs = { (0,0): {(0,0):-1}, (0,1): {(0,1):1}, (1,0): {}, (1,1): {} } # M5 in C00(-), C01(+)
    C6_coeffs = { (0,0): {}, (0,1): {}, (1,0): {}, (1,1): {(1,1):1} } # M6 in C11(+)
    C7_coeffs = { (0,0): {}, (0,1): {}, (1,0): {}, (1,1): {(1,1):-1} } # M7 in C11(-)
    
    C_list = [C1_coeffs, C2_coeffs, C3_coeffs, C4_coeffs, C5_coeffs, C6_coeffs, C7_coeffs]
    
    for l in range(7):
        for t in range(7):
            c = l * 7 + t
            # U
            for a in range(2):
                for b in range(2):
                    for idx, coeff in A_list[l][(a,b)].items():
                        U4[idx, c] += coeff * U2[a*2+b, t]
                        
            # V
            for c_ in range(2):
                for d_ in range(2):
                    for idx, coeff in B_list[l][(c_,d_)].items():
                        V4[idx, c] += coeff * V2[c_*2+d_, t]
                        
            # W
            for u in range(2):
                for w in range(2):
                    for (a,d), coeff in C_list[l][(u,w)].items():
                        K = (w*2+d)*4 + (u*2+a)
                        W4[K, c] += coeff * W2[a*2+d, t]
                        
    # Build 4x4x8 factors
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # First half
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            J_local = j * 4 + k
            J_global = j * 8 + k
            V[J_global, 0:49] = V4[J_local, 0:49]
            
    for k in range(4):
        for i in range(4):
            K_local = k * 4 + i
            K_global = k * 4 + i
            W[K_global, 0:49] = W4[K_local, 0:49]
            
    # Second half
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            J_local = j * 4 + k
            J_global = j * 8 + (4 + k)
            V[J_global, 49:98] = V4[J_local, 0:49]
            
    for k in range(4):
        for i in range(4):
            K_local = k * 4 + i
            K_global = (4 + k) * 4 + i
            W[K_global, 49:98] = W4[K_local, 0:49]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def get_2x2_strassen():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[0, 1] = -1; W[1, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = -1
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = -1
    
    return U, V, W
# EVOLVE-BLOCK-END
