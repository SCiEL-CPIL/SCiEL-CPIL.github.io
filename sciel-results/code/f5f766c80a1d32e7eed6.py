import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    # Dimensions: A (2x4), B (4x7), C (2x7)
    n, m, p = 2, 4, 7
    
    # Target Rank
    R = 50
    
    # Initialize matrices
    # U: n*m x R
    # V: m*p x R
    # W: p*n x R
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    rank_counter = 0
    
    # Helper to add a Strassen 2x2x2 block
    # i0: start row of A
    # j0: start col of A (row of B)
    # k0: start col of B (row of C)
    def strassen_block(i0, j0, k0):
        nonlocal rank_counter
        if rank_counter >= R:
            return
        
        # A indices
        a00 = i0 * m + j0
        a01 = i0 * m + j0 + 1
        a10 = i0 + 1 * m + j0
        a11 = i0 + 1 * m + j0 + 1
        
        # B indices
        b00 = j0 * p + k0
        b01 = j0 * p + k0 + 1
        b10 = j0 + 1 * p + k0
        b11 = j0 + 1 * p + k0 + 1
        
        # C indices (W uses k*n + i)
        c00 = k0 * n + i0
        c01 = k0 * n + i0 + 1
        c10 = k0 + 1 * n + i0
        c11 = k0 + 1 * n + i0 + 1
        
        # M1 = (a00 + a11)(b00 + b11)
        U[a00, rank_counter] += 1
        U[a11, rank_counter] += 1
        V[b00, rank_counter] += 1
        V[b11, rank_counter] += 1
        W[c00, rank_counter] += 1
        W[c11, rank_counter] += 1
        rank_counter += 1
        
        # M2 = (a10 + a11)b00
        U[a10, rank_counter] += 1
        U[a11, rank_counter] += 1
        V[b00, rank_counter] += 1
        W[c10, rank_counter] += 1
        W[c11, rank_counter] -= 1
        rank_counter += 1
        
        # M3 = a00(b01 - b11)
        U[a00, rank_counter] += 1
        V[b01, rank_counter] += 1
        V[b11, rank_counter] -= 1
        W[c01, rank_counter] += 1
        W[c11, rank_counter] += 1
        rank_counter += 1
        
        # M4 = a11(b10 - b00)
        U[a11, rank_counter] += 1
        V[b10, rank_counter] += 1
        V[b00, rank_counter] -= 1
        W[c00, rank_counter] += 1
        W[c10, rank_counter] += 1
        rank_counter += 1
        
        # M5 = (a00 + a01)b11
        U[a00, rank_counter] += 1
        U[a01, rank_counter] += 1
        V[b11, rank_counter] += 1
        W[c00, rank_counter] -= 1
        W[c01, rank_counter] += 1
        rank_counter += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        U[a10, rank_counter] += 1
        U[a00, rank_counter] -= 1
        V[b00, rank_counter] += 1
        V[b01, rank_counter] += 1
        W[c11, rank_counter] += 1
        rank_counter += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        U[a01, rank_counter] += 1
        U[a11, rank_counter] -= 1
        V[b10, rank_counter] += 1
        V[b11, rank_counter] += 1
        W[c00, rank_counter] += 1
        rank_counter += 1

    # Helper to add a Schoolbook 2x2x1 block
    def schoolbook_block(i0, j0, k0):
        nonlocal rank_counter
        if rank_counter >= R:
            return
        
        # A indices
        a00 = i0 * m + j0
        a01 = i0 * m + j0 + 1
        a10 = i0 + 1 * m + j0
        a11 = i0 + 1 * m + j0 + 1
        
        # B indices
        b00 = j0 * p + k0
        b10 = j0 + 1 * p + k0
        
        # C indices
        c00 = k0 * n + i0
        c10 = k0 * n + i0 + 1
        
        # P1 = a00 b00
        U[a00, rank_counter] += 1
        V[b00, rank_counter] += 1
        W[c00, rank_counter] += 1
        rank_counter += 1
        
        # P2 = a01 b10
        U[a01, rank_counter] += 1
        V[b10, rank_counter] += 1
        W[c00, rank_counter] += 1
        rank_counter += 1
        
        # P3 = a10 b00
        U[a10, rank_counter] += 1
        V[b00, rank_counter] += 1
        W[c10, rank_counter] += 1
        rank_counter += 1
        
        # P4 = a11 b10
        U[a11, rank_counter] += 1
        V[b10, rank_counter] += 1
        W[c10, rank_counter] += 1
        rank_counter += 1

    # Decomposition Strategy:
    # Split M (4) into two 2s: j=0,1 and j=2,3
    # For each j-block, split K (7) into 2, 2, 2, 1
    
    # Block 1: i=0,1, j=0,1
    # k=0,1
    if time.time() < deadline:
        strassen_block(0, 0, 0)
    # k=2,3
    if time.time() < deadline:
        strassen_block(0, 0, 2)
    # k=4,5
    if time.time() < deadline:
        strassen_block(0, 0, 4)
    # k=6
    if time.time() < deadline:
        schoolbook_block(0, 0, 6)
        
    # Block 2: i=0,1, j=2,3
    # k=0,1
    if time.time() < deadline:
        strassen_block(0, 2, 0)
    # k=2,3
    if time.time() < deadline:
        strassen_block(0, 2, 2)
    # k=4,5
    if time.time() < deadline:
        strassen_block(0, 2, 4)
    # k=6
    if time.time() < deadline:
        schoolbook_block(0, 2, 6)

    # Verification
    # Check if U, V, W are valid
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                # Vectorized dot product
                dot = np.dot(U[u_idx, :rank_counter], V[v_idx, :rank_counter])
                val = np.dot(dot, W[w_idx, :rank_counter])
                if abs(val - 1) > 1e-9:
                    max_err = max(max_err, abs(val - 1))
                    
    if max_err > 1e-9:
        # Fallback: Return zero matrices (invalid, but distinct code path)
        return {
            "U": np.zeros((n*m, 1)).tolist(),
            "V": np.zeros((m*p, 1)).tolist(),
            "W": np.zeros((p*n, 1)).tolist()
        }
        
    # Convert to lists as per some strict interpretations, 
    # though numpy arrays are usually accepted if dtype is correct.
    # The prompt says "factor matrices of shapes...". 
    # To be safe, return lists of lists of floats.
    
    return {
        "U": U[:, :rank_counter].tolist(),
        "V": V[:, :rank_counter].tolist(),
        "W": W[:, :rank_counter].tolist()
    }
