import numpy as np

def construct():
    n, m, p = 2, 3, 4
    nm, mp, pn = n*m, m*p, p*n  # 6, 12, 8
    
    # Rank-22 decomposition using Strassen on two 2x2 blocks + standard on the rest
    # This is a different construction method than the add_term approach
    
    # We'll build U, V, W as numpy arrays for cleaner construction
    U = np.zeros((nm, 22), dtype=object)
    V = np.zeros((mp, 22), dtype=object)
    W = np.zeros((pn, 22), dtype=object)
    
    def _a(i, j):  # A[i,j] -> U index
        return i * m + j
    def _b(i, j):  # B[i,j] -> V index
        return i * p + j
    def _c(i, j):  # C[i,j] -> W index (W[k*n+i] where k=j, i=row)
        return j * n + i
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2], Strassen rank 7
    # Strassen formulas for 2x2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # c00 = M1+M4+M6-M5
    # c01 = M3+M5
    # c10 = M2+M4
    # c11 = M1+M3-M2+M7
    
    for r in range(7):
        col = r
        if r == 0:
            U[_a(0,0), col] += 1; U[_a(1,1), col] += 1
            V[_b(0,0), col] += 1; V[_b(1,1), col] += 1
            W[_c(0,0), col] += 1; W[_c(1,1), col] += 1
        elif r == 1:
            U[_a(1,0), col] += 1; U[_a(1,1), col] += 1
            V[_b(0,0), col] += 1
            W[_c(1,0), col] += 1; W[_c(1,1), col] += -1
        elif r == 2:
            U[_a(0,0), col] += 1
            V[_b(0,1), col] += 1; V[_b(1,1), col] += -1
            W[_c(0,1), col] += 1; W[_c(1,1), col] += 1
        elif r == 3:
            U[_a(1,1), col] += 1
            V[_b(1,0), col] += 1; V[_b(0,0), col] += -1
            W[_c(0,0), col] += 1; W[_c(1,0), col] += 1
        elif r == 4:
            U[_a(0,0), col] += 1; U[_a(0,1), col] += 1
            V[_b(1,1), col] += 1
            W[_c(0,0), col] += -1; W[_c(0,1), col] += 1
        elif r == 5:
            U[_a(1,0), col] += 1; U[_a(0,0), col] += -1
            V[_b(0,0), col] += 1; V[_b(0,1), col] += 1
            W[_c(1,1), col] += 1
        elif r == 6:
            U[_a(0,1), col] += 1; U[_a(1,1), col] += -1
            V[_b(1,0), col] += 1; V[_b(1,1), col] += 1
            W[_c(0,0), col] += 1
    
    # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4], Strassen rank 7
    # Same A block, different B columns (2,3 instead of 0,1)
    for r in range(7):
        col = 7 + r
        if r == 0:
            U[_a(0,0), col] += 1; U[_a(1,1), col] += 1
            V[_b(0,2), col] += 1; V[_b(1,3), col] += 1
            W[_c(0,2), col] += 1; W[_c(1,3), col] += 1
        elif r == 1:
            U[_a(1,0), col] += 1; U[_a(1,1), col] += 1
            V[_b(0,2), col] += 1
            W[_c(1,2), col] += 1; W[_c(1,3), col] += -1
        elif r == 2:
            U[_a(0,0), col] += 1
            V[_b(0,3), col] += 1; V[_b(1,3), col] += -1
            W[_c(0,3), col] += 1; W[_c(1,3), col] += 1
        elif r == 3:
            U[_a(1,1), col] += 1
            V[_b(1,2), col] += 1; V[_b(0,2), col] += -1
            W[_c(0,2), col] += 1; W[_c(1,2), col] += 1
        elif r == 4:
            U[_a(0,0), col] += 1; U[_a(0,1), col] += 1
            V[_b(1,3), col] += 1
            W[_c(0,2), col] += -1; W[_c(0,3), col] += 1
        elif r == 5:
            U[_a(1,0), col] += 1; U[_a(0,0), col] += -1
            V[_b(0,2), col] += 1; V[_b(0,3), col] += 1
            W[_c(1,3), col] += 1
        elif r == 6:
            U[_a(0,1), col] += 1; U[_a(1,1), col] += -1
            V[_b(1,2), col] += 1; V[_b(1,3), col] += 1
            W[_c(0,2), col] += 1
    
    # Block 3: A[0:2,2:3] * B[2:3,0:4] -> C[0:2,0:4], standard rank 8
    # A[0,2]*B[2,j] -> C[0,j], A[1,2]*B[2,j] -> C[1,j] for j=0,1,2,3
    for j in range(4):
        col = 14 + j
        U[_a(0,2), col] += 1
        V[_b(2,j), col] += 1
        W[_c(0,j), col] += 1
        
    for j in range(4):
        col = 18 + j
        U[_a(1,2), col] += 1
        V[_b(2,j), col] += 1
        W[_c(1,j), col] += 1
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    result = construct()
    rank = len(result['U'][0])
    print(f"Rank: {rank}")
    print(f"U shape: ({len(result['U'])}, {rank})")
    print(f"V shape: ({len(result['V'])}, {rank})")
    print(f"W shape: ({len(result['W'])}, {rank})")
