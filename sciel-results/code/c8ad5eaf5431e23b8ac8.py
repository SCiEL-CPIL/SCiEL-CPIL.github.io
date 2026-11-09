import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_4x4x4():
    """Return 49 rank-1 terms for 4x4x4 Strassen decomposition."""
    # 2x2x2 Strassen base terms (u, v, w) each of length 4
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    base_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        ([0,0,0,1], [0,0,1,-1], [0,0,0,1]),
        ([1,1,0,0], [0,0,0,1], [0,1,0,1]),
        ([0,0,1,-1], [1,1,0,0], [0,0,1,0]),
        ([0,1,0,-1], [0,0,1,1], [0,0,0,1]),
    ]
    
    # Verify base terms for 2x2x2
    # Build the tensor and check
    tensor = np.zeros((4,4,4))
    for (u, v, w) in base_terms:
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    tensor[i,j,k] += u[i] * v[j] * w[k]
    
    # Expected tensor for 2x2x2 multiplication
    expected = np.zeros((4,4,4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                expected[i*2+j, j*2+k, k*2+i] = 1
    
    if not np.allclose(tensor, expected):
        # Base terms are wrong, return empty to signal failure
        return []
    
    # Generate 49 terms for 4x4x4
    def generate_2x2_terms(a_blocks, b_blocks, c_blocks, c_coeffs):
        terms_out = []
        for (u_base, v_base, w_base) in base_terms:
            u = [0] * 16
            v = [0] * 16
            w = [0] * 16
            for local_idx, val in enumerate(u_base):
                if val == 0:
                    continue
                li, lj = divmod(local_idx, 2)
                block_idx = li
                i_off, j_off = a_blocks[block_idx]
                u[(i_off + li) * 4 + (j_off + lj)] += val
            for local_idx, val in enumerate(v_base):
                if val == 0:
                    continue
                lj, lk = divmod(local_idx, 2)
                block_idx = lj
                j_off, k_off = b_blocks[block_idx]
                v[(j_off + lj) * 4 + (k_off + lk)] += val
            for local_idx, val in enumerate(w_base):
                if val == 0:
                    continue
                lk, li = divmod(local_idx, 2)
                block_idx = lk
                k_off, i_off = c_blocks[block_idx]
                w[(k_off + lk) * 4 + (i_off + li)] += val * c_coeffs[block_idx]
            terms_out.append((u, v, w))
        return terms_out
    
    # 7 Strassen operations for 4x4x4
    # A11:(0,0), A12:(0,2), A21:(2,0), A22:(2,2)
    # B11:(0,0), B12:(0,2), B21:(2,0), B22:(2,2)
    # C11:(0,0), C12:(0,2), C21:(2,0), C22:(2,2)
    
    all_terms = []
    # S1: (A11+A22)(B11+B22) -> C11(+1), C22(+1)
    all_terms += generate_2x2_terms([(0,0),(2,2)], [(0,0),(2,2)], [(0,0),(2,2)], [1,1])
    # S2: (A21+A22)B11 -> C21(+1)
    all_terms += generate_2x2_terms([(2,0),(2,2)], [(0,0),(0,0)], [(2,0),(2,0)], [1,1])
    # S3: A11(B12-B22) -> C12(+1), C22(-1)
    all_terms += generate_2x2_terms([(0,0),(0,0)], [(0,2),(2,2)], [(0,2),(2,2)], [1,-1])
    # S4: A22(B21-B11) -> C11(-1), C21(+1)
    all_terms += generate_2x2_terms([(2,2),(2,2)], [(2,0),(0,0)], [(0,0),(2,0)], [-1,1])
    # S5: (A11+A12)B22 -> C12(+1), C22(+1)
    all_terms += generate_2x2_terms([(0,0),(0,2)], [(2,2),(2,2)], [(0,2),(2,2)], [1,1])
    # S6: (A21-A11)(B11+B12) -> C11(+1), C21(+1)
    all_terms += generate_2x2_terms([(2,0),(0,0)], [(0,0),(0,2)], [(0,0),(2,0)], [1,1])
    # S7: (A12-A22)(B21+B22) -> C11(-1), C22(+1)
    all_terms += generate_2x2_terms([(0,2),(2,2)], [(2,0),(2,2)], [(0,0),(2,2)], [-1,1])
    
    return all_terms

def _schoolbook_rect(n, m, p):
    """Return rank-1 terms for n x m * m x p schoolbook."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                terms.append((u, v, w))
    return terms

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0]) if U else 0
    # Build tensor from decomposition
    tensor = np.zeros((n*m, m*p, p*n))
    for k in range(rank):
        u = np.array([U[i][k] for i in range(n*m)])
        v = np.array([V[j][k] for j in range(m*p)])
        w = np.array([W[i][k] for i in range(p*n)])
        tensor += np.einsum('i,j,k->ijk', u, v, w)
    
    # Expected tensor
    expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i*m+j, j*p+k, k*n+i] = 1
    
    return np.allclose(tensor, expected, atol=1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = 4, 5, 6
    
    # Try Strassen-based decomposition
    strassen_terms = _strassen_4x4x4()
    
    if len(strassen_terms) == 49:
        all_terms = []
        
        # Term 1a: A[:,0:4] * B[0:4,0:4] -> C[0:4,0:4] (4x4x4 Strassen)
        for (u, v, w) in strassen_terms:
            u_global = [0] * (n * m)
            v_global = [0] * (m * p)
            w_global = [0] * (p * n)
            
            for idx, val in enumerate(u):
                if val != 0:
                    i, j = divmod(idx, 4)
                    u_global[i * 5 + j] += val
            
            for idx, val in enumerate(v):
                if val != 0:
                    j, k = divmod(idx, 4)
                    v_global[j * 6 + k] += val
            
            for idx, val in enumerate(w):
                if val != 0:
                    k, i = divmod(idx, 4)
                    w_global[k * 4 + i] += val
            
            all_terms.append((u_global, v_global, w_global))
        
        # Term 1b: A[:,0:4] * B[0:4,4:6] -> C[4:6,0:4] (4x4x2 schoolbook)
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    u[i * 5 + j] = 1
                    v[j * 6 + (k + 4)] = 1
                    w[(k + 4) * 4 + i] = 1
                    all_terms.append((u, v, w))
        
        # Term 2: A[:,4:5] * B[4:5,:] -> C[:,0:6] (4x1x6 schoolbook)
        for i in range(4):
            for k in range(6):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * 5 + 4] = 1
                v[4 * 6 + k] = 1
                w[k * 4 + i] = 1
                all_terms.append((u, v, w))
        
        rank = len(all_terms)
        
        # Build U, V, W
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for col, (u, v, w) in enumerate(all_terms):
            for i in range(n * m):
                if u[i] != 0:
                    U[i][col] = u[i]
            for j in range(m * p):
                if v[j] != 0:
                    V[j][col] = v[j]
            for k in range(p * n):
                if w[k] != 0:
                    W[k][col] = w[k]
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
