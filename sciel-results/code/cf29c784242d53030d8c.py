import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Rank is 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    # Local indices i, j, k in {0, 1}
    # U_local[i, j], V_local[j, k], W_local[k, i]
    
    # Definitions of the 7 products
    # Each product is a list of (u_indices, v_indices, w_indices, coeff)
    # Actually, it's simpler to define the coefficients for U, V, W directly.
    
    # M1: U: (0,0)=1, (1,1)=1; V: (0,0)=1, (1,1)=1
    # M2: U: (1,0)=1, (1,1)=1; V: (0,0)=1
    # M3: U: (0,0)=1; V: (0,1)=1, (1,1)=-1
    # M4: U: (1,1)=1; V: (1,0)=1, (0,0)=-1
    # M5: U: (0,0)=1, (0,1)=1; V: (1,1)=1
    # M6: U: (1,0)=1, (0,0)=-1; V: (0,0)=1, (0,1)=1
    # M7: U: (0,1)=1, (1,1)=-1; V: (1,0)=1, (1,1)=1
    
    # W coefficients:
    # C00: M1(1), M4(1), M5(-1), M7(1)
    # C01: M3(1), M5(1)
    # C10: M2(1), M4(1)
    # C11: M1(1), M2(-1), M3(1), M6(1)
    
    products = [
        # (u_dict, v_dict, w_dict)
        (
            {(0,0): 1, (1,1): 1}, 
            {(0,0): 1, (1,1): 1}, 
            {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1} # Wait, W indices are (k, i) or (i, k)?
        ),
        # Let's stick to the formula:
        # C_{00} (i=0, k=0) gets M1, M4, -M5, M7
        # C_{01} (i=0, k=1) gets M3, M5
        # C_{10} (i=1, k=0) gets M2, M4
        # C_{11} (i=1, k=1) gets M1, -M2, M3, M6
    ]
    
    # Let's restructure to be cleaner.
    # For each of the 7 products, define U coeffs, V coeffs, W coeffs.
    # U coeffs: dict (i,j) -> val
    # V coeffs: dict (j,k) -> val
    # W coeffs: dict (i,k) -> val (Note: task says W[k*n+i], so index is k, i)
    
    strassen_products = []
    
    # M1
    strassen_products.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1} # C00+=M1, C11+=M1
    })
    
    # M2
    strassen_products.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1, (1,1): -1} # C10+=M2, C11-=M2
    })
    
    # M3
    strassen_products.append({
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): 1} # C01+=M3, C11+=M3
    })
    
    # M4
    strassen_products.append({
        'U': {(1,1): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,0): 1, (1,0): 1} # C00+=M4, C10+=M4
    })
    
    # M5
    strassen_products.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): -1, (0,1): 1} # C00-=M5, C01+=M5
    })
    
    # M6
    strassen_products.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1} # C11+=M6
    })
    
    # M7
    strassen_products.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1} # C00+=M7
    })
    
    # Blocks:
    # 1: A_L B_TL -> i,j in {0,1}, k in {0,1}
    # 2: A_L B_TR -> i,j in {0,1}, k in {2,3}
    # 3: A_R B_BL -> i in {0,1}, j in {2,3}, k in {0,1}
    # 4: A_R B_BR -> i in {0,1}, j in {2,3}, k in {2,3}
    
    col = 0
    for block_id in range(4):
        # Determine offsets for j and k
        # Block 0: j_off=0, k_off=0
        # Block 1: j_off=0, k_off=2
        # Block 2: j_off=2, k_off=0
        # Block 3: j_off=2, k_off=2
        j_off = (block_id // 2) * 2
        k_off = (block_id % 2) * 2
        
        for prod in strassen_products:
            # Fill U
            for (i, j), val in prod['U'].items():
                u_idx = i * m + (j + j_off)
                U[u_idx][col] += val
                
            # Fill V
            for (j, k), val in prod['V'].items():
                v_idx = (j + j_off) * p + k
                V[v_idx][col] += val
                
            # Fill W
            for (i, k), val in prod['W'].items():
                # W index is k*n + i
                w_idx = k * n + i
                W[w_idx][col] += val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}

# Verification (optional, for debugging)
def verify():
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    n, m, p = DIMENSIONS
    # Tensor T = sum U_col * V_col^T * W_col^T?
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Check against schoolbook
    # Schoolbook rank n*m*p
    # T[i,j,k] = 1 for all i,j,k
    
    # Let's check a few entries
    # T[0,0,0] should be 1
    # T[0,0,0] = sum_r U[0, r] V[0, r] W[0, r]
    
    # We can compute the full tensor and compare to 1s
    T = np.zeros((n*m, m*p, p*n))
    for r in range(res['U'].__len__() and res['U'][0].__len__()):
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    T[i,j,k] += U[i,r] * V[j,r] * W[k,r]
    
    # The tensor for matrix multiplication is all 1s?
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Yes, T_{i,j,k} = 1.
    
    # Check if T is all 1s
    # Note: indices in T are flattened i, j, k
    # i in 0..7, j in 0..15, k in 0..7
    # But not all combinations are valid?
    # No, the tensor is defined on the full space.
    # However, the multiplication tensor is sparse?
    # No, the tensor $T$ has $T_{i,j,k} = 1$ for all $i,j,k$ in the domain.
    # Wait, is it?
    # $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # The coefficient of $A_{i,j} B_{j,k}$ in $C_{i,k}$ is 1.
    # So yes, the tensor is all 1s.
    
    if np.allclose(T, 1.0):
        print("Verification Passed: Tensor is all 1s")
    else:
        print("Verification Failed")
        print("Max diff:", np.max(np.abs(T - 1.0)))
        # Print non-1 entries
        diff = np.abs(T - 1.0)
        indices = np.where(diff > 1e-9)
        for idx in zip(*indices):
            print(f"Index {idx}: {T[idx]}")

if __name__ == "__main__":
    verify()
