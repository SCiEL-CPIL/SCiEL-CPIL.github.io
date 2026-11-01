import time
import sys

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 14 (7 for each 2x2x2 block)
    rank = 14
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank) = (4, 14)
    # V: (m*p, rank) = (8, 14)
    # W: (p*n, rank) = (8, 14)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen factors for a 2x2x2 block
    # Local indices i, j, k in {0, 1}
    # u_r[i, j], v_r[j, k], w_r[i, k]
    
    # We will fill 7 columns for block 1 (k=0,1) and 7 for block 2 (k=2,3)
    
    # Block 1: k_local in {0, 1}, k_global = k_local
    # Block 2: k_local in {0, 1}, k_global = k_local + 2
    
    # Strassen components
    # 1: (a+d)(e+h)
    # 2: (c+d)e
    # 3: a(f-h)
    # 4: d(g-e)
    # 5: (a+b)h
    # 6: (c-a)(e+f)
    # 7: (b-d)(g+h)
    
    # Factors for U (i,j)
    # 1: (1,0,0,1) -> indices 0,3
    # 2: (0,0,1,1) -> indices 2,3
    # 3: (1,0,0,0) -> index 0
    # 4: (0,0,0,1) -> index 3
    # 5: (1,1,0,0) -> indices 0,1
    # 6: (-1,0,1,0) -> indices 0,2
    # 7: (0,1,0,-1) -> indices 1,3
    
    # Factors for V (j,k)
    # 1: (1,1,1,1) -> indices 0,1,2,3 (j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1)
    #    Actually V index is j*p+k. For 2x2 block p=2.
    #    j=0,k=0 -> 0
    #    j=0,k=1 -> 1
    #    j=1,k=0 -> 2
    #    j=1,k=1 -> 3
    # 1: (1,1,1,1)
    # 2: (1,0,0,0) -> index 0
    # 3: (0,1,0,-1) -> indices 1,3
    # 4: (0,-1,1,0) -> indices 1,2
    # 5: (0,0,0,1) -> index 3
    # 6: (1,1,0,0) -> indices 0,1
    # 7: (0,0,1,1) -> indices 2,3
    
    # Factors for W (i,k)
    # W index is k*n+i. For 2x2 block n=2.
    # k=0,i=0 -> 0
    # k=0,i=1 -> 1
    # k=1,i=0 -> 2
    # k=1,i=1 -> 3
    # 1: (1,0,0,1) -> indices 0,3
    # 2: (0,1,0,0) -> index 1
    # 3: (0,0,1,0) -> index 2
    # 4: (1,1,0,0) -> indices 0,1
    # 5: (-1,0,1,0) -> indices 0,2
    # 6: (0,0,0,1) -> index 3
    # 7: (1,0,0,-1) -> indices 0,3
    
    # Let's map these to the global indices.
    
    # Helper to add a factor
    def add_factor(u_coeffs, v_coeffs, w_coeffs, offset):
        # u_coeffs: dict mapping local (i,j) to value
        # v_coeffs: dict mapping local (j,k) to value
        # w_coeffs: dict mapping local (i,k) to value
        # offset: column offset in U,V,W
        
        for (i, j), val in u_coeffs.items():
            U[i*m + j][offset] = val
            
        for (j, k), val in v_coeffs.items():
            # k is local k. Global k depends on block.
            # We pass global k directly in v_coeffs? No, better to pass local and map.
            # Let's assume v_coeffs keys are (j, k_global)
            V[j*p + k][offset] = val
            
        for (i, k), val in w_coeffs.items():
            # k is global k
            W[k*n + i][offset] = val

    # Block 1: k_global in {0, 1}
    # Block 2: k_global in {2, 3}
    
    # Define the 7 terms
    # Term 1: (a+d)(e+h)
    u1 = {(0,0): 1, (1,1): 1}
    # For block 1: v1 local k=0,1 -> global 0,1
    # For block 2: v1 local k=0,1 -> global 2,3
    
    # Let's just iterate through the 7 terms and assign to columns 0-6 and 7-13.
    
    terms = []
    
    # T1
    terms.append({
        'u': {(0,0): 1, (1,1): 1},
        'v': {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1},
        'w': {(0,0): 1, (1,1): 1}
    })
    # T2
    terms.append({
        'u': {(1,0): 1, (1,1): 1},
        'v': {(0,0): 1},
        'w': {(1,0): 1}
    })
    # T3
    terms.append({
        'u': {(0,0): 1},
        'v': {(0,1): 1, (1,1): -1},
        'w': {(0,1): 1}
    })
    # T4
    terms.append({
        'u': {(1,1): 1},
        'v': {(1,0): 1, (0,0): -1},
        'w': {(0,0): 1, (1,0): 1}
    })
    # T5
    terms.append({
        'u': {(0,0): 1, (0,1): 1},
        'v': {(1,1): 1},
        'w': {(0,0): -1, (0,1): 1}
    })
    # T6
    terms.append({
        'u': {(1,0): 1, (0,0): -1},
        'v': {(0,0): 1, (0,1): 1},
        'w': {(1,1): 1}
    })
    # T7
    terms.append({
        'u': {(0,1): 1, (1,1): -1},
        'v': {(1,0): 1, (1,1): 1},
        'w': {(0,0): 1, (1,1): -1}
    })
    
    # Fill Block 1 (columns 0-6)
    for t_idx in range(7):
        col = t_idx
        term = terms[t_idx]
        
        for (i, j), val in term['u'].items():
            U[i*m + j][col] = val
            
        for (j, k_local), val in term['v'].items():
            k_global = k_local # Block 1
            V[j*p + k_global][col] = val
            
        for (i, k_local), val in term['w'].items():
            k_global = k_local # Block 1
            W[k_global*n + i][col] = val

    # Fill Block 2 (columns 7-13)
    for t_idx in range(7):
        col = 7 + t_idx
        term = terms[t_idx]
        
        for (i, j), val in term['u'].items():
            U[i*m + j][col] = val
            
        for (j, k_local), val in term['v'].items():
            k_global = k_local + 2 # Block 2
            V[j*p + k_global][col] = val
            
        for (i, k_local), val in term['w'].items():
            k_global = k_local + 2 # Block 2
            W[k_global*n + i][col] = val

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple verification
    res = construct()
    n, m, p = DIMENSIONS
    U = res["U"]
    V = res["V"]
    W = res["W"]
    rank = len(U[0])
    
    # Check tensor decomposition
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can check a few random elements or all elements.
    # Since it's small, check all.
    
    import random
    # We don't have random A, B. We check the tensor T itself.
    # The tensor for matrix multiplication is T[i,j,k] = 1 if (i,j,k) is a valid path?
    # No, the tensor T is defined by T[i,j,k] = delta_{i,j} delta_{j,k}? No.
    # The multiplication tensor T has T[i,j,k] = 1 if i,j,k correspond to the product term A[i,j]B[j,k].
    # So T[i,j,k] = 1 for all valid i,j,k?
    # Yes, the tensor of matrix multiplication is sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{i,k} ?
    # No.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T is such that <T, A \otimes B> = C.
    # T[i,j,k] is the coefficient of A[i,j] B[j,k] in C.
    # Since C[i,k] = sum_j A[i,j] B[j,k], the term A[i,j] B[j,k] contributes to C[i,k].
    # So T[i,j,k] = 1 for all i,j,k.
    
    # Let's verify:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] should be 1 for all i,j,k.
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    print(f"Mismatch at {i},{j},{k}: sum={s}")
                    valid = False
    if valid:
        print("Verification passed.")
    else:
        print("Verification failed.")
