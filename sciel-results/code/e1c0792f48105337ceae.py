"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is correct using NumPy for speed."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    # Build the tensor: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    T = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U_arr[:, c].reshape(n, m)
        v_c = V_arr[:, c].reshape(m, p)
        w_c = W_arr[:, c].reshape(p, n)
        # T[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
        # For each i,j: T[i,j,:] += u_c[i,j] * (v_c[j,:] * w_c[:,i])
        for i in range(n):
            for j in range(m):
                T[i, j, :] += u_c[i, j] * (v_c[j, :] * w_c[:, i])
    # Expected: T[i,j,k] = 1 for all i,j,k
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _build_3x3x3_rank23():
    """
    Build a rank-23 decomposition for 3x3x3 matrix multiplication.
    A(3x3) * B(3x3) -> C(3x3)
    
    Index conventions:
    U[i*m+j] where m=3: index = 3i+j
    V[j*p+k] where p=3: index = 3j+k  
    W[k*n+i] where n=3: index = 3k+i
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    """
    terms = []
    
    # Strassen for 2x2x2 (i,j in {0,1}, k in {0,1}): 7 terms
    # U indices: 0:A[0,0], 1:A[0,1], 2:A[0,2], 3:A[1,0], 4:A[1,1], 5:A[1,2], 6:A[2,0], 7:A[2,1], 8:A[2,2]
    # V indices: 0:B[0,0], 1:B[0,1], 2:B[0,2], 3:B[1,0], 4:B[1,1], 5:B[1,2], 6:B[2,0], 7:B[2,1], 8:B[2,2]
    # W indices: 0:C[0,0], 1:C[0,1], 2:C[0,2], 3:C[1,0], 4:C[1,1], 5:C[1,2], 6:C[2,0], 7:C[2,1], 8:C[2,2]
    
    # M0: (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) -> C[0,0]+=1, C[1,1]+=1
    terms.append(([1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0]))
    # M1: (A[1,0]+A[1,1])*B[0,0] -> C[1,0]+=1
    terms.append(([0,0,0, 1,1,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]))
    # M2: A[0,0]*(B[1,0]-B[1,1]) -> C[0,1]+=1
    terms.append(([1,0,0, 0,0,0, 0,0,0], [0,0,0, 1,-1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]))
    # M3: A[1,1]*(B[0,1]-B[0,0]) -> C[0,0]+=-1, C[1,1]+=-1
    terms.append(([0,0,0, 0,1,0, 0,0,0], [-1,1,0, 0,0,0, 0,0,0], [-1,0,0, 0,-1,0, 0,0,0]))
    # M4: (A[0,0]+A[0,1])*B[1,1] -> C[0,1]+=1
    terms.append(([1,1,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]))
    # M5: (A[1,0]-A[0,0])*(B[0,0]+B[0,1]) -> C[1,0]+=1
    terms.append(([-1,0,0, 1,0,0, 0,0,0], [1,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]))
    # M6: (A[0,1]-A[1,1])*(B[1,0]+B[1,1]) -> C[0,0]+=1
    terms.append(([0,1,0, 0,-1,0, 0,0,0], [0,0,0, 1,1,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]))
    
    # 15 cross terms involving index 2:
    # A[2,0]*B[0,0] -> C[2,0]
    terms.append(([0,0,0, 0,0,0, 1,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]))
    # A[2,0]*B[0,1] -> C[2,1]
    terms.append(([0,0,0, 0,0,0, 1,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]))
    # A[2,0]*B[0,2] -> C[2,2]
    terms.append(([0,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]))
    # A[2,1]*B[1,0] -> C[2,0]
    terms.append(([0,0,0, 0,0,0, 0,1,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]))
    # A[2,1]*B[1,1] -> C[2,1]
    terms.append(([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]))
    # A[2,1]*B[1,2] -> C[2,2]
    terms.append(([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1]))
    # A[2,2]*B[2,0] -> C[2,0]
    terms.append(([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 1,0,0]))
    # A[2,2]*B[2,1] -> C[2,1]
    terms.append(([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0]))
    # A[2,2]*B[2,2] -> C[2,2]
    terms.append(([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]))
    # A[0,2]*B[2,0] -> C[0,0]
    terms.append(([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [1,0,0, 0,0,0, 0,0,0]))
    # A[0,2]*B[2,1] -> C[0,1]
    terms.append(([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,1,0, 0,0,0, 0,0,0]))
    # A[0,2]*B[2,2] -> C[0,2]
    terms.append(([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,0]))
    # A[1,2]*B[2,0] -> C[1,0]
    terms.append(([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 1,0,0, 0,0,0]))
    # A[1,2]*B[2,1] -> C[1,1]
    terms.append(([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,1,0, 0,0,0]))
    # A[1,2]*B[2,2] -> C[1,2]
    terms.append(([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0]))
    
    return terms

def _build_3x3x6_from_3x3x3(block_terms, n=3, m=3, p=6, block_p=3):
    """
    Build full 3x3x6 decomposition from 3x3x3 block terms.
    Split B and C into 2 blocks of 3 columns each.
    """
    r_block = len(block_terms)
    total_rank = 2 * r_block
    
    U_full = [[0]*total_rank for _ in range(n*m)]
    V_full = [[0]*total_rank for _ in range(m*p)]
    W_full = [[0]*total_rank for _ in range(p*n)]
    
    for t in range(2):
        for c in range(r_block):
            col = t * r_block + c
            u, v, w = block_terms[c]
            
            # U: same indices (3i+j)
            for idx in range(n*m):
                U_full[idx][col] = u[idx]
            
            # V: local index j*block_p+k -> global index j*p + (t*block_p+k)
            for j in range(m):
                for k in range(block_p):
                    local_v_idx = j * block_p + k
                    global_v_idx = j * p + (t * block_p + k)
                    V_full[global_v_idx][col] = v[local_v_idx]
            
            # W: local index block_p*i+k -> global index (t*block_p+k)*n + i
            for i in range(n):
                for k in range(block_p):
                    local_w_idx = block_p * i + k
                    global_w_idx = (t * block_p + k) * n + i
                    W_full[global_w_idx][col] = w[local_w_idx]
    
    return U_full, V_full, W_full

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Try the 3x3x3 rank-23 based decomposition (rank 46 for 3x3x6)
    block_terms = _build_3x3x3_rank23()
    U_full, V_full, W_full = _build_3x3x6_from_3x3x3(block_terms, n, m, p)
    
    if _verify_decomposition(U_full, V_full, W_full, n, m, p):
        return {"U": U_full, "V": V_full, "W": W_full}
    
    if time.time() > deadline:
        # Fallback: schoolbook rank 54
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 54
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
