import time
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

t0 = time.time()
DEADLINE = t0 + 125.0  # Safety margin under 150s kill


def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition (rank 7).
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # r=0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # r=1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # r=2: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # r=3: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1

    # r=4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # r=5: M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # r=6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _kronecker_product(U1, V1, W1, U2, V2, W2):
    """
    Tensor product of two matrix multiplication decompositions.
    If (U1,V1,W1) has rank r1 for (n1,m1,p1) and (U2,V2,W2) has rank r2 for (n2,m2,p2),
    returns (U,V,W) of rank r1*r2 for (n1*n2, m1*m2, p1*p2).
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    total_rank = r1 * r2

    n1, m1, p1 = U1.shape[0] // m1, U1.shape[0] // n1 if False else U1.shape[0] // (U1.shape[0] // (U1.shape[0] // m1)), V1.shape[0] // p1
    # Actually let me derive dimensions properly
    # U1 shape: (n1*m1, r1)
    # V1 shape: (m1*p1, r1)
    # W1 shape: (p1*n1, r1)
    
    n1_m1 = U1.shape[0]
    m1_p1 = V1.shape[0]
    p1_n1 = W1.shape[0]
    
    n2_m2 = U2.shape[0]
    m2_p2 = V2.shape[0]
    p2_n2 = W2.shape[0]
    
    # For simplicity, assume square-ish: n1=m1=p1 for 2x2x2 case
    # General case: extract dimensions from shapes
    # U1: (n1*m1, r1), V1: (m1*p1, r1), W1: (p1*n1, r1)
    # This is tricky without explicit dimensions. Let's hardcode for our use case.
    # For 2x2x2: U1 is (4,7), V1 is (4,7), W1 is (4,7) => n1=m1=p1=2
    # For the product to work, we need to know the individual dimensions.
    # Let's pass them as arguments or infer.
    
    # Since we only use this for 2x2x2 x 2x2x2 -> 4x4x4, let's hardcode the logic
    # or infer. n1*m1=4, m1*p1=4, p1*n1=4 => n1=m1=p1=2
    # n2*m2=4, m2*p2=4, p2*n2=4 => n2=m2=p2=2
    
    # Result dims: n=4, m=4, p=4
    # U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    
    U = np.zeros((n1_m1 * n2_m2, total_rank), dtype=np.float64)
    V = np.zeros((m1_p1 * m2_p2, total_rank), dtype=np.float64)
    W = np.zeros((p1_n1 * p2_n2, total_rank), dtype=np.float64)
    
    # We need to map indices correctly.
    # For 2x2x2:
    # U1[i*2+j, c1], V1[j*2+k, c1], W1[k*2+i, c1]
    # U2[i'*2+j', c2], V2[j'*2+k', c2], W2[k'*2+i', c2]
    
    # Combined index for U: i_total = i1*2 + i2 (if n=4), j_total = j1*2 + j2 (if m=4)
    # U[i_total*4 + j_total, c1*7 + c2] = U1[i1*2+j1, c1] * U2[i2*2+j2, c2]
    
    # This hardcoding works for the specific case. Let's write it generally for the 2x2x2 case.
    # Since we only need this specific product, let's just do it directly in the builder.
    pass


def _build_4x4x4():
    """
    Builds the rank-49 decomposition for 4x4x4 using Kronecker product of Strassen.
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    rank_444 = 49
    U4 = np.zeros((16, rank_444), dtype=np.float64)
    V4 = np.zeros((16, rank_444), dtype=np.float64)
    W4 = np.zeros((16, rank_444), dtype=np.float64)
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                i0, i1 = i // 2, i % 2
                for j in range(4):
                    j0, j1 = j // 2, j % 2
                    U4[i*4+j, c] = U2[i0*2+j0, c0] * U2[i1*2+j1, c1]
            
            for j in range(4):
                j0, j1 = j // 2, j % 2
                for k in range(4):
                    k0, k1 = k // 2, k % 2
                    V4[j*4+k, c] = V2[j0*2+k0, c0] * V2[j1*2+k1, c1]
            
            for k in range(4):
                k0, k1 = k // 2, k % 2
                for i in range(4):
                    i0, i1 = i // 2, i % 2
                    W4[k*4+i, c] = W2[k0*2+i0, c0] * W2[k1*2+i1, c1]
    
    return U4, V4, W4


def _build_rank98():
    """
    Builds a rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    Uses the rank-49 4x4x4 decomposition duplicated for the two 4x4 blocks.
    """
    U4, V4, W4 = _build_4x4x4()
    
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.float64)
    V = np.zeros((32, total_rank), dtype=np.float64)
    W = np.zeros((32, total_rank), dtype=np.float64)
    
    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49
        U[:, col_start:col_end] = U4
        
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j*8 + k_global, col_start:col_end] = V4[j*4 + k_local, :]
        
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global*4 + i, col_start:col_end] = W4[k_local*4 + i, :]
                
    # Verification
    n, m, p = 4, 4, 8
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(total_rank):
        u_col = U[:, r].reshape(n, m)
        v_col = V[:, r].reshape(m, p)
        w_col = W[:, r].reshape(p, n)
        result += np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
    
    if not np.allclose(result, 1.0, atol=1e-9):
        return None
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }


def _verify(U, V, W, n=4, m=4, p=8, atol=1e-6):
    """Verify a decomposition."""
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(U.shape[1]):
        u_col = U[:, r].reshape(n, m)
        v_col = V[:, r].reshape(m, p)
        w_col = W[:, r].reshape(p, n)
        result += np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
    return np.allclose(result, 1.0, atol=atol)


def _try_rank_reduction(U_init, V_init, W_init, target_rank, time_limit):
    """
    Attempts to reduce the rank by optimizing a sub-matrix of the factors.
    Uses a greedy approach: try to zero out columns one by one.
    """
    n, m, p = 4, 4, 8
    current_rank = U_init.shape[1]
    
    if current_rank <= target_rank:
        return None
    
    # We'll try to find a valid decomposition with fewer columns.
    # Strategy: Start with the full decomposition, and iteratively try to
    # remove one column at a time by re-optimizing the remaining columns.
    
    # This is computationally expensive. Let's try a simpler heuristic:
    # Check if any column is linearly dependent on others in the tensor space.
    # This is complex. Instead, let's try a few random perturbations and roundings.
    
    # Actually, the most effective simple improvement is to check if the rank-98
    # solution can be represented with integer coefficients (it already is) and
    # if there's any redundancy.
    
    # For now, let's just try to optimize a smaller subset.
    # This is a placeholder for a more sophisticated search.
    return None


def _schoolbook(n=4, m=4, p=8):
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


def construct():
    # First, build the rank-98 solution
    res = _build_rank98()
    if res is not None:
        return res
    
    # Fallback to schoolbook
    return _schoolbook()


if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
