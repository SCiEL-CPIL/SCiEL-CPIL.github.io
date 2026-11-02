"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time

def _strassen_4x4_4x4_terms():
    """
    Generate the 49-term Strassen decomposition for 4x4 * 4x4.
    Each term is (u_idx, v_idx, w_idx, sign).
    U: 16x49, V: 16x49, W: 16x49
    U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    """
    # Strassen for 4x4: split into 2x2 blocks, use 7 products.
    # Each 2x2*2x2 uses Strassen (7 terms), so total 7*7=49 terms.
    
    # Define the 7 Strassen products at the 4x4 level.
    # A blocks: A11=A[0:2,0:2], A12=A[0:2,2:4], A21=A[2:4,0:2], A22=A[2:4,2:4]
    # B blocks: B11=B[0:2,0:2], B12=B[0:2,2:4], B21=B[2:4,0:2], B22=B[2:4,2:4]
    
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mx is a 2x2 result. For each Mx, we use Strassen (7 terms) for the 2x2*2x2.
    # But the inputs to Mx are themselves sums/differences of 2x2 blocks.
    
    # This is a nested Strassen. Let me just use schoolbook for the inner 2x2*2x2
    # to keep it simpler: 7 * 8 = 56 terms.
    
    # Actually, let me just use schoolbook for the entire 4x4*4x4 to be safe.
    # That gives 64 terms. Not an improvement.
    
    # I'll return None to indicate this function is not used yet.
    return None

def construct():
    n, m, p = DIMENSIONS
    # Use schoolbook: rank = 144
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for col in range(rank):
        k = col % p
        tmp = col // p
        j = tmp % m
        i = tmp // m
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
