"""Program for rank-minimal exact tensor decomposition of 4x5*5x6 matrix multiplication."""
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Uses a block decomposition with Strassen's algorithm for 2x2x2 blocks 
    and schoolbook for 2x1x2 blocks.
    Expected rank: 108 (vs 120 schoolbook).
    """
    n, m, p = DIMENSIONS
    
    # Define block offsets and sizes
    # n=4 split into 2, 2
    # m=5 split into 2, 2, 1
    # p=6 split into 2, 2, 2
    
    i_offsets = [0, 2]
    i_sizes = [2, 2]
    
    j_offsets = [0, 2, 4]
    j_sizes = [2, 2, 1]
    
    k_offsets = [0, 2, 4]
    k_sizes = [2, 2, 2]
    
    # Estimate max rank to allocate memory
    # 12 Strassen blocks * 7 + 6 Schoolbook blocks * 4 = 84 + 24 = 108
    max_rank = 120 # Safe upper bound
    
    U = np.zeros((n * m, max_rank), dtype=int)
    V = np.zeros((m * p, max_rank), dtype=int)
    W = np.zeros((p * n, max_rank), dtype=int)
    
    col = 0
    
    for i_idx, i_0 in enumerate(i_offsets):
        n_b = i_sizes[i_idx]
        for j_idx, j_0 in enumerate(j_offsets):
            m_b = j_sizes[j_idx]
            for k_idx, k_0 in enumerate(k_offsets):
                p_b = k_sizes[k_idx]
                
                # Handle 2x2x2 block with Strassen
                if n_b == 2 and m_b == 2 and p_b == 2:
                    # Strassen terms defined by (u, v, w) local indices
                    # Term 1: M1 = (A00+A11)(B00+B11) -> C00, C11
                    # A: (0,0), (1,1); B: (0,0), (1,1); C: (0,0), (1,1)
                    terms = [
                        # (u, v, w, coeff) for U, V, W
                        # We need to add to U, V, W for each term
                        # Let's define the contributions explicitly
                    ]
                    
                    # Term 1
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    U[(i_0+1)*m + (j_0+1), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    V[(j_0+1)*p + (k_0+1), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    W[(k_0+1)*n + (i_0+1), col] = 1
                    col += 1
                    
                    # Term 2: M2 = (A10+A11)B00 -> C10, C11(-)
                    U[(i_0+1)*m + (j_0+0), col] = 1
                    U[(i_0+1)*m + (j_0+1), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+1), col] = 1
                    W[(k_0+1)*n + (i_0+1), col] = -1
                    col += 1
                    
                    # Term 3: M3 = A00(B01-B11) -> C01, C11
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+1), col] = 1
                    V[(j_0+1)*p + (k_0+1), col] = -1
                    W[(k_0+1)*n + (i_0+0), col] = 1
                    W[(k_0+1)*n + (i_0+1), col] = 1
                    col += 1
                    
                    # Term 4: M4 = A11(B10-B00) -> C00, C10
                    U[(i_0+1)*m + (j_0+1), col] = 1
                    V[(j_0+1)*p + (k_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+0), col] = -1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+1), col] = 1
                    col += 1
                    
                    # Term 5: M5 = (A00+A01)B11 -> C01, C00(-)
                    U[(i_0+0)*m + (j_0+0), col] = 1
                    U[(i_0+0)*m + (j_0+1), col] = 1
                    V[(j_0+1)*p + (k_0+1), col] = 1
                    W[(k_0+1)*n + (i_0+0), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = -1
                    col += 1
                    
                    # Term 6: M6 = (A10-A00)(B00+B01) -> C11
                    U[(i_0+1)*m + (j_0+0), col] = 1
                    U[(i_0+0)*m + (j_0+0), col] = -1
                    V[(j_0+0)*p + (k_0+0), col] = 1
                    V[(j_0+0)*p + (k_0+1), col] = 1
                    W[(k_0+1)*n + (i_0+1), col] = 1
                    col += 1
                    
                    # Term 7: M7 = (A01-A11)(B10+B11) -> C00
                    U[(i_0+0)*m + (j_0+1), col] = 1
                    U[(i_0+1)*m + (j_0+1), col] = -1
                    V[(j_0+1)*p + (k_0+0), col] = 1
                    V[(j_0+1)*p + (k_0+1), col] = 1
                    W[(k_0+0)*n + (i_0+0), col] = 1
                    col += 1
                    
                elif n_b == 2 and m_b == 1 and p_b == 2:
                    # Schoolbook for 2x1x2
                    # C_wu = A_u0 * B_0w
                    for u in range(2):
                        for w in range(2):
                            U[(i_0+u)*m + (j_0+0), col] = 1
                            V[(j_0+0)*p + (k_0+w), col] = 1
                            W[(k_0+w)*n + (i_0+u), col] = 1
                            col += 1
                else:
                    # Fallback: Schoolbook for any other size (should not happen in this specific split)
                    # But for safety, implement generic schoolbook
                    for u in range(n_b):
                        for v in range(m_b):
                            for w in range(p_b):
                                U[(i_0+u)*m + (j_0+v), col] = 1
                                V[(j_0+v)*p + (k_0+w), col] = 1
                                W[(k_0+w)*n + (i_0+u), col] = 1
                                col += 1

    # Trim arrays to actual rank
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
