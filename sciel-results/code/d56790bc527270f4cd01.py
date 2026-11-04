import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer
    
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for ip in range(n):
                        total = 0.0
                        for c in range(rank):
                            total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + ip][c]
                        expected = 1.0 if i == ip else 0.0
                        if abs(total - expected) > 1e-9:
                            return False
        return True

    # Try to construct a rank 14 solution using Strassen for 2x2x7 blocks
    # A is 2x4, B is 4x7
    # Split A into A1 (cols 0,1) and A2 (cols 2,3)
    # Split B into B1 (rows 0,1) and B2 (rows 2,3)
    # C = A1 B1 + A2 B2
    
    # Strassen decomposition for 2x2 by 2x7
    # Rank 7 per block
    # Total rank 14
    
    def construct_strassen_block(offset_j):
        # offset_j is 0 for A1/B1, 2 for A2/B2
        # Returns U_block (2x2x7), V_block (2x7x7), W_block (7x2x7)
        # U indices: i (0,1), j (0,1 local)
        # V indices: j (0,1 local), k (0..6)
        # W indices: k (0..6), i' (0,1)
        
        U_b = np.zeros((2, 2, 7))
        V_b = np.zeros((2, 7, 7))
        W_b = np.zeros((7, 2, 7))
        
        # Strassen factors for 2x2x2 extended to 2x2x7
        # M1 = (c+d)(r0+r1) -> U[1,0]=1, U[1,1]=1; V[0,k]=1, V[1,k]=1; W[k,0]=1, W[k,1]=1? No.
        # Let's use the standard Strassen matrix for 2x2x2 and extend.
        
        # Standard Strassen for 2x2x2:
        # U (2x2x7):
        # c=0: a -> U[0,0]=1
        # c=1: b -> U[0,1]=1
        # c=2: c -> U[1,0]=1
        # c=3: d -> U[1,1]=1
        # c=4: a+b -> U[0,0]=1, U[0,1]=1
        # c=5: a-b -> U[0,0]=1, U[0,1]=-1
        # c=6: c-a -> U[1,0]=1, U[0,0]=-1
        
        # V (2x2x7) for 2x2x2:
        # c=0: e -> V[0,0]=1
        # c=1: f -> V[0,1]=1
        # c=2: g -> V[1,0]=1
        # c=3: h -> V[1,1]=1
        # c=4: g+h -> V[1,0]=1, V[1,1]=1
        # c=5: e -> V[0,0]=1
        # c=6: e+g -> V[0,0]=1, V[1,0]=1
        
        # W (2x2x7) for 2x2x2:
        # c=0: a -> W[0,0]=1
        # c=1: b -> W[0,1]=1
        # c=2: c -> W[1,0]=1
        # c=3: d -> W[1,1]=1
        # c=4: a+b -> W[0,0]=1, W[0,1]=1
        # c=5: a-b -> W[0,0]=1, W[0,1]=-1
        # c=6: c-a -> W[1,0]=1, W[0,0]=-1
        
        # For 2x2x7, we replicate the V and W structures for each column k.
        # V[j, k, c] depends on k.
        # In 2x2x2, V[0,0]=1 (e), V[0,1]=1 (f).
        # In 2x2x7, for column k, V[0, k, c] should correspond to the "e" or "f" slot.
        # Actually, the Strassen algorithm computes C = AB.
        # C[0, k] = a B[0, k] + b B[1, k].
        # C[1, k] = c B[0, k] + d B[1, k].
        
        # Let's define the 7 factors explicitly for 2x2x7.
        # Factor 0: a * B[0, :]
        U_b[0, 0, 0] = 1
        for k in range(7):
            V_b[0, k, 0] = 1
            W_b[k, 0, 0] = 1
            W_b[k, 1, 0] = 0 # C[1,k] doesn't get 'a'
            
        # Factor 1: b * B[1, :]
        U_b[0, 1, 1] = 1
        for k in range(7):
            V_b[1, k, 1] = 1
            W_b[k, 0, 1] = 0
            W_b[k, 1, 1] = 1
            
        # Factor 2: c * B[0, :]
        U_b[1, 0, 2] = 1
        for k in range(7):
            V_b[0, k, 2] = 1
            W_b[k, 0, 2] = 0
            W_b[k, 1, 2] = 1
            
        # Factor 3: d * B[1, :]
        U_b[1, 1, 3] = 1
        for k in range(7):
            V_b[1, k, 3] = 1
            W_b[k, 0, 3] = 0
            W_b[k, 1, 3] = 1
            
        # Factor 4: (a+b) * (B[0, :] + B[1, :])
        U_b[0, 0, 4] = 1
        U_b[0, 1, 4] = 1
        for k in range(7):
            V_b[0, k, 4] = 1
            V_b[1, k, 4] = 1
            W_b[k, 0, 4] = 1
            W_b[k, 1, 4] = 1
            
        # Factor 5: (a-b) * B[0, :]
        U_b[0, 0, 5] = 1
        U_b[0, 1, 5] = -1
        for k in range(7):
            V_b[0, k, 5] = 1
            W_b[k, 0, 5] = 1
            W_b[k, 1, 5] = 0
            
        # Factor 6: (c-a) * (B[0, :] + B[1, :])? 
        # Standard Strassen M7 = (c-a)(e+g).
        # Here e=B[0,:], g=B[1,:].
        U_b[1, 0, 6] = 1
        U_b[0, 0, 6] = -1
        for k in range(7):
            V_b[0, k, 6] = 1
            V_b[1, k, 6] = 1
            W_b[k, 0, 6] = 1
            W_b[k, 1, 6] = 1

        return U_b, V_b, W_b

    # Assemble Rank 14 solution
    rank = 14
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Block 1: A1 (cols 0,1) x B1 (rows 0,1)
    U1, V1, W1 = construct_strassen_block(0)
    # U1 is 2x2x7. Map to U[i, j, c]
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U[i * m + j, c] = U1[i, j, c]
                
    for j in range(2):
        for k in range(7):
            for c in range(7):
                V[j * p + k, c] = V1[j, k, c]
                
    for k in range(7):
        for i in range(2):
            for c in range(7):
                W[k * n + i, c] = W1[k, i, c]
                
    # Block 2: A2 (cols 2,3) x B2 (rows 2,3)
    U2, V2, W2 = construct_strassen_block(2)
    # Offset c by 7
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U[i * m + (j + 2), 7 + c] = U2[i, j, c]
                
    for j in range(2):
        for k in range(7):
            for c in range(7):
                V[(j + 2) * p + k, 7 + c] = V2[j, k, c]
                
    for k in range(7):
        for i in range(2):
            for c in range(7):
                W[k * n + i, 7 + c] += W2[k, i, c] # Add to existing W? No, separate columns.
                # Wait, W is summed over c.
                # If we use separate columns, W2 should be in columns 7..13.
                W[k * n + i, 7 + c] = W2[k, i, c]

    # Verify
    if verify(U.tolist(), V.tolist(), W.tolist(), rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback
    return get_schoolbook()
# EVOLVE-BLOCK-END
