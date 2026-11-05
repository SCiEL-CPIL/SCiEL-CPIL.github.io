import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    def get_strassen_222():
        # Returns U, V, W for 2x2x2
        # U: 4x7, V: 4x7, W: 4x7
        # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1 # C11, C22?
        # W index: k*n+i. n=2.
        # C11 (k=0,i=0) -> 0*2+0=0.
        # C22 (k=1,i=1) -> 1*2+1=3.
        # M1 contributes to C11 (+1) and C22 (+1).
        W[0][0] = 1
        W[3][0] = 1
        
        # M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1 # C21 (k=1,i=0) -> 1*2+0=2.
        
        # M3
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1 # C12 (k=1,i=0) -> 1*2+0=2? No.
        # C12: i=0, k=1. Index 1*2+0 = 2.
        # Wait, W index is k*n+i.
        # C11: i=0,k=0 -> 0.
        # C12: i=0,k=1 -> 2.
        # C21: i=1,k=0 -> 1.
        # C22: i=1,k=1 -> 3.
        
        # Let's re-map W indices carefully.
        # W[k*n+i]
        # n=2.
        # k=0, i=0 -> 0 (C11)
        # k=0, i=1 -> 1 (C21)
        # k=1, i=0 -> 2 (C12)
        # k=1, i=1 -> 3 (C22)
        
        # M1: (A11+A22)(B11+B22)
        # A11=U[0], A22=U[3]
        # B11=V[0], B22=V[3]
        # C11 += 1, C22 += 1
        W[0][0] = 1
        W[3][0] = 1
        
        # M2: (A21+A22)B11
        # A21=U[2], A22=U[3]
        # B11=V[0]
        # C21 += 1
        W[1][1] = 1
        
        # M3: A11(B21-B22)
        # A11=U[0]
        # B21=V[2], B22=V[3]
        # C12 += 1
        W[2][2] = 1
        
        # M4: A22(B12-B11)
        # A22=U[3]
        # B12=V[1], B11=V[0]
        # C11 += -1, C22 += 1
        W[0][3] = -1
        W[3][3] = 1
        
        # M5: (A11+A12)B22
        # A11=U[0], A12=U[1]
        # B22=V[3]
        # C11 += -1, C12 += 1
        W[0][4] = -1
        W[2][4] = 1
        
        # M6: (A21-A11)(B11+B12)
        # A21=U[2], A11=U[0]
        # B11=V[0], B12=V[1]
        # C22 += 1
        W[3][5] = 1
        
        # M7: (A12-A22)(B21+B22)
        # A12=U[1], A22=U[3]
        # B21=V[2], B22=V[3]
        # C11 += 1, C22 += -1
        W[0][6] = 1
        W[3][6] = -1
        
        return U, V, W

    def get_rank11_223():
        # 2x2x3 decomposition
        # Split p into 2+1
        # Use Strassen for 2x2x2 (rank 7)
        # Use schoolbook for 2x2x1 (rank 4)
        # Total 11
        U1, V1, W1 = get_strassen_222()
        
        # 2x2x1 part
        # A is 2x2, B is 2x1, C is 2x1
        # C1 = A11 B11 + A21 B21
        # C2 = A12 B12 + A22 B22
        # U: 4x4, V: 2x4, W: 2x4
        U2 = [[0]*4 for _ in range(4)]
        V2 = [[0]*4 for _ in range(2)]
        W2 = [[0]*4 for _ in range(2)]
        
        # Term 1: A11 B11 -> C1
        U2[0][0] = 1
        V2[0][0] = 1
        W2[0][0] = 1 # C1 (i=0, k=0? No, C is 2x1. i=0,1. k=0.
        # W index k*n+i. n=2.
        # C1: i=0, k=0 -> 0.
        # C2: i=1, k=0 -> 1.
        
        # Term 2: A21 B21 -> C1
        U2[2][1] = 1
        V2[1][1] = 1
        W2[0][1] = 1
        
        # Term 3: A12 B12 -> C2
        U2[1][2] = 1
        V2[0][2] = 1
        W2[1][2] = 1
        
        # Term 4: A22 B22 -> C2
        U2[3][3] = 1
        V2[1][3] = 1
        W2[1][3] = 1
        
        # Concatenate
        U = [U1[i] + U2[i] for i in range(4)]
        V = [V1[i] + V2[i] for i in range(2)] # V1 is 4x7, V2 is 2x4. Mismatch!
        # V1 rows are j*p+k. p=2.
        # V2 rows are j*p+k. p=1.
        # We need to align V indices.
        # The 2x2x2 part uses B columns 0,1.
        # The 2x2x1 part uses B column 2.
        # So V2 should be 0 in columns 0,1 and non-zero in column 2?
        # No, V is size (m*p) x rank.
        # For 2x2x2, m=2, p=2. V is 4x7.
        # For 2x2x1, m=2, p=1. V is 2x4.
        # In the full 2x2x3, m=2, p=3. V is 6x11.
        # The first 4 rows of V correspond to p=0,1.
        # The last 2 rows of V correspond to p=2.
        
        # So we need to expand V1 to 6x7 (pad rows 4,5 with 0).
        # And expand V2 to 6x4 (pad rows 0,1,2,3 with 0, keep 4,5).
        
        V1_full = [[0]*7 for _ in range(6)]
        for i in range(4):
            V1_full[i] = V1[i]
            
        V2_full = [[0]*4 for _ in range(6)]
        for i in range(2):
            V2_full[4+i] = V2[i]
            
        V_full = [ [V1_full[i][r] + V2_full[i][r] for r in range(11)] for i in range(6) ]
        
        # W: 2x2x2 W is 4x7. Indices k*n+i.
        # 2x2x1 W is 2x4. Indices k*n+i.
        # In full 2x2x3, W is (p*n) x rank = 6x11.
        # W1 rows 0,1,2,3 correspond to k=0,1 and i=0,1.
        # W2 rows 0,1 correspond to k=0 and i=0,1.
        # We need to map W2 to the correct rows in W_full.
        # W2 is for the 3rd column of B (k=2).
        # So W2 rows should map to k=2.
        # W index = k*n+i.
        # k=2, i=0 -> 4.
        # k=2, i=1 -> 5.
        
        W1_full = [[0]*7 for _ in range(6)]
        for i in range(4):
            W1_full[i] = W1[i]
            
        W2_full = [[0]*4 for _ in range(6)]
        W2_full[4] = W2[0]
        W2_full[5] = W2[1]
        
        W_full = [ [W1_full[i][r] + W2_full[i][r] for r in range(11)] for i in range(6) ]
        
        return U, V_full, W_full

    def schoolbook(n, m, p):
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    def combine_decomps(n1, m1, p1, U1, V1, W1, n2, m2, p2, U2, V2, W2):
        # This function is not used directly in the final plan, 
        # but the recursive logic handles the mapping.
        pass

    def decompose(n, m, p):
        if n == 2 and m == 2 and p == 2:
            return get_strassen_222()
        if n == 2 and m == 2 and p == 3:
            return get_rank11_223()
        
        # Try splitting
        best_rank = n*m*p
        best_decomp = None
        
        # Split n
        for n1 in range(1, n):
            n2 = n - n1
            if m > 0 and p > 0:
                U1, V1, W1 = decompose(n1, m, p)
                U2, V2, W2 = decompose(n2, m, p)
                rank = len(U1[0]) + len(U2[0])
                if rank < best_rank:
                    best_rank = rank
                    # Construct full U, V, W
                    U = [[0]*rank for _ in range(n*m)]
                    V = [[0]*rank for _ in range(m*p)]
                    W = [[0]*rank for _ in range(p*n)]
                    
                    # Map U1 (n1*m x r1) to rows 0..n1*m-1
                    for i in range(n1*m):
                        for r in range(len(U1[0])):
                            U[i][r] = U1[i][r]
                    # Map U2 (n2*m x r2) to rows n1*m..n*m-1
                    for i in range(n2*m):
                        for r in range(len(U2[0])):
                            U[n1*m+i][r+len(U1[0])] = U2[i][r]
                            
                    # V is same for both (m*p x r)
                    for i in range(m*p):
                        for r in range(len(V1[0])):
                            V[i][r] = V1[i][r]
                        for r in range(len(V2[0])):
                            V[i][r+len(V1[0])] = V2[i][r]
                            
                    # W1 (p*n1 x r1) to rows 0..p*n1-1
                    for i in range(p*n1):
                        for r in range(len(W1[0])):
                            W[i][r] = W1[i][r]
                    # W2 (p*n2 x r2) to rows p*n1..p*n-1
                    for i in range(p*n2):
                        for r in range(len(W2[0])):
                            W[p*n1+i][r+len(W1[0])] = W2[i][r]
                            
                    best_decomp = (U, V, W)
        
        # Split p
        for p1 in range(1, p):
            p2 = p - p1
            U1, V1, W1 = decompose(n, m, p1)
            U2, V2, W2 = decompose(n, m, p2)
            rank = len(U1[0]) + len(U2[0])
            if rank < best_rank:
                best_rank = rank
                U = [[0]*rank for _ in range(n*m)]
                V = [[0]*rank for _ in range(m*p)]
                W = [[0]*rank for _ in range(p*n)]
                
                for i in range(n*m):
                    for r in range(len(U1[0])):
                        U[i][r] = U1[i][r]
                    for r in range(len(U2[0])):
                        U[i][r+len(U1[0])] = U2[i][r]
                        
                # V1 (m*p1 x r1) to rows 0..m*p1-1
                for i in range(m*p1):
                    for r in range(len(V1[0])):
                        V[i][r] = V1[i][r]
                # V2 (m*p2 x r2) to rows m*p1..m*p-1
                for i in range(m*p2):
                    for r in range(len(V2[0])):
                        V[m*p1+i][r+len(V1[0])] = V2[i][r]
                        
                for i in range(p*n):
                    for r in range(len(W1[0])):
                        W[i][r] = W1[i][r]
                    for r in range(len(W2[0])):
                        W[i][r+len(W1[0])] = W2[i][r]
                        
                best_decomp = (U, V, W)

        if best_decomp is None:
            U, V, W = schoolbook(n, m, p)
            return U, V, W
        return best_decomp

    U, V, W = decompose(n, m, p)
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Rank: {U.shape[1]}")
