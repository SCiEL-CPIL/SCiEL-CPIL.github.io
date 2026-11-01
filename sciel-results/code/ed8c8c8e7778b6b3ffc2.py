import numpy as np

DIMENSIONS = [2, 4, 8]

def _strassen_2x2_decomp():
    """
    Returns U, V, W for 2x2 matrix multiplication with rank 7.
    U: 4x7 (indices: a11=0, a12=1, a21=2, a22=3)
    V: 4x7 (indices: b11=0, b12=1, b21=2, b22=3)
    W: 4x7 (indices: c11=0, c12=1, c21=2, c22=3)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2 = (A21+A22)B11
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] -= 1
    
    # M3 = A11(B12-B22)
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[1][2] += 1; W[3][2] += 1
    
    # M4 = A22(B21-B11)
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[2][3] += 1
    
    # M5 = (A11+A12)B22
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[1][4] += 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Get the base Strassen decomposition
    base_U, base_V, base_W = _strassen_2x2_decomp()
    
    # Iterate over the 8 blocks
    # u: 0, 1 (A block index)
    # v: 0, 1 (B row block index)
    # k: 0, 1, 2, 3 (B col block index / C col block index)
    
    for u in range(2):
        for v in range(2):
            for k in range(4):
                base_col = 7 * (4 * u + 2 * v + k)
                
                # U indices for A_u
                # A is 2x4. A_u is cols 2u, 2u+1.
                # U index = i*m + j
                a11_u = 0 * m + 2 * u
                a12_u = 0 * m + (2 * u + 1)
                a21_u = 1 * m + 2 * u
                a22_u = 1 * m + (2 * u + 1)
                
                # V indices for B_{v,k}
                # B is 4x8. B_{v,k} is rows 2v, 2v+1, cols 2k, 2k+1.
                # V index = j*p + k
                b11_vk = (2 * v) * p + 2 * k
                b12_vk = (2 * v) * p + (2 * k + 1)
                b21_vk = (2 * v + 1) * p + 2 * k
                b22_vk = (2 * v + 1) * p + (2 * k + 1)
                
                # W indices for C_k
                # C is 2x8. C_k is cols 2k, 2k+1.
                # W index = k*n + i
                # Z11 -> C[0, 2k] -> i=0, k_g=2k -> 2k*2 + 0
                # Z12 -> C[0, 2k+1] -> i=0, k_g=2k+1 -> (2k+1)*2 + 0
                # Z21 -> C[1, 2k] -> i=1, k_g=2k -> 2k*2 + 1
                # Z22 -> C[1, 2k+1] -> i=1, k_g=2k+1 -> (2k+1)*2 + 1
                c11_w = 2 * k * n + 0
                c12_w = (2 * k + 1) * n + 0
                c21_w = 2 * k * n + 1
                c22_w = (2 * k + 1) * n + 1
                
                # Map base indices 0,1,2,3 to specific U/V/W indices
                # base_U[0] corresponds to A11
                # base_U[1] corresponds to A12
                # base_U[2] corresponds to A21
                # base_U[3] corresponds to A22
                u_map = [a11_u, a12_u, a21_u, a22_u]
                v_map = [b11_vk, b12_vk, b21_vk, b22_vk]
                w_map = [c11_w, c12_w, c21_w, c22_w]
                
                for s in range(7):
                    for idx in range(4):
                        if base_U[idx][s] != 0:
                            U[u_map[idx]][base_col + s] += base_U[idx][s]
                        if base_V[idx][s] != 0:
                            V[v_map[idx]][base_col + s] += base_V[idx][s]
                        if base_W[idx][s] != 0:
                            W[w_map[idx]][base_col + s] += base_W[idx][s]
                            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
