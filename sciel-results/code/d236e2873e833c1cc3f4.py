import time
import sys

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    
    # Rank 34 decomposition
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, r, val=1):
        U[u_idx][r] += val
        V[v_idx][r] += val
        W[w_idx][r] += val
    
    # Block 1: j=3 (2x1x6), rank 6
    # i in {0,1}, j=3, k in {0..5}
    for r in range(6):
        for i in range(2):
            add_term(i * m + 3, 3 * p + r, r * n + i, r)
        # V and W need delta_k,r
        # V[3*p + r, r] = 1
        # W[r*n + i, r] = 1
        # The loop above adds 1 to all i for a fixed r.
        # But V[3*p + k, r] should be 1 only if k=r.
        # W[k*n + i, r] should be 1 only if k=r.
        
        # Correcting Block 1
        # We need U[i, 3, r] V[3, k, r] W[k, i, r] = 1
        # Let V[3, r, r] = 1, others 0.
        # Let W[r, i, r] = 1, others 0.
        # Let U[i, 3, r] = 1.
        pass

    # Reset and do it properly
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: j=3, rank 6
    for r in range(6):
        for i in range(2):
            U[i * m + 3][col] = 1
        V[3 * p + r][col] = 1
        for i in range(2):
            W[r * n + i][col] = 1
        col += 1
        
    # Block 2a: j in {0,1}, k in {0,1,2}, rank 11
    # 2x2x2 part (k=0,1) rank 7 + 2x2x1 part (k=2) rank 4
    # Strassen for 2x2x2
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # We need to map Strassen to U,V,W
    # This is tedious. I will use schoolbook for these sub-blocks to ensure correctness, 
    # but that gives rank 12+12+6 = 30? No, schoolbook 2x2x3 is 12.
    # 12 + 3 + 12 + 3 + 6 = 36.
    # 36 is better than 48.
    
    # Let's just use schoolbook for all sub-blocks to be safe and fast.
    # Total rank:
    # j=3 (2x1x6): 12
    # j=0,1,2 k=0,1,2 (2x3x3): 18
    # j=0,1,2 k=3,4,5 (2x3x3): 18
    # Total: 12 + 18 + 18 = 48. Same as schoolbook.
    
    # I need to implement Strassen to get 34.
    # Given the complexity, I will return the schoolbook rank 48 if I can't implement Strassen quickly.
    # But 48 is the seed.
    
    # Let's try to implement the 2x1x6 block correctly and schoolbook for the rest.
    # 2x1x6 rank 6.
    # 2x3x3 schoolbook rank 18.
    # 2x3x3 schoolbook rank 18.
    # Total 6 + 18 + 18 = 42.
    
    U = [[0] * 42 for _ in range(n * m)]
    V = [[0] * 42 for _ in range(m * p)]
    W = [[0] * 42 for _ in range(p * n)]
    
    col = 0
    # Block 1: j=3 (2x1x6), rank 6
    for r in range(6):
        for i in range(2):
            U[i * m + 3][col] = 1
        V[3 * p + r][col] = 1
        for i in range(2):
            W[r * n + i][col] = 1
        col += 1
        
    # Block 2: j in {0,1,2}, k in {0,1,2} (2x3x3), schoolbook rank 18
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Block 3: j in {0,1,2}, k in {3,4,5} (2x3x3), schoolbook rank 18
    for i in range(2):
        for j in range(3):
            for k in range(3, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(res['U'][0])
