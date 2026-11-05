"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-26 exact tensor decomposition for 2x4x5 matrix multiplication
    using Strassen's algorithm for 2x2x2 blocks and a custom rank-6 bilinear
    decomposition for 2x2x3 blocks.
    
    Block structure:
    - A[:, 0:2] * B[0:2, 0:2] -> Strassen 2x2x2 (rank 7)
    - A[:, 0:2] * B[0:2, 2:5] -> custom 2x2x3 (rank 6)
    - A[:, 2:4] * B[2:4, 0:2] -> Strassen 2x2x2 (rank 7)
    - A[:, 2:4] * B[2:4, 2:5] -> custom 2x2x3 (rank 6)
    Total: 7 + 6 + 7 + 6 = 26
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = 26
    _t0 = time.time()
    _DEADLINE = 120  # seconds

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_term(idx, u_coeffs, v_coeffs, w_coeffs):
        """
        Add a rank-1 term to U, V, W.
        u_coeffs: dict {(i, j): val}
        v_coeffs: dict {(j, k): val}
        w_coeffs: dict {(i, k): val}  # W indexed as W[k*n+i]
        """
        for (i, j), val in u_coeffs.items():
            U[i*m + j, idx] += val
        for (j, k), val in v_coeffs.items():
            V[j*p + k, idx] += val
        for (i, k), val in w_coeffs.items():
            W[k*n + i, idx] += val

    def strassen_block(a_j_off, b_j_off, b_k_off, idx_start):
        """
        Add Strassen decomposition for a 2x2x2 block.
        a_j_off: offset for A columns
        b_j_off: offset for B rows  
        b_k_off: offset for B columns
        """
        # P1 = (A00+A11)*(B00+B11) -> C00+=1, C11+=1
        add_term(idx_start,
                 {(0, a_j_off): 1, (1, a_j_off+1): 1},
                 {(b_j_off, b_k_off): 1, (b_j_off+1, b_k_off+1): 1},
                 {(0, b_k_off): 1, (1, b_k_off+1): 1})
        
        # P2 = (A10+A11)*B00 -> C10+=1, C11-=1
        add_term(idx_start+1,
                 {(1, a_j_off): 1, (1, a_j_off+1): 1},
                 {(b_j_off, b_k_off): 1},
                 {(1, b_k_off): 1, (1, b_k_off+1): -1})
        
        # P3 = A00*(B01-B11) -> C01+=1, C11+=1
        add_term(idx_start+2,
                 {(0, a_j_off): 1},
                 {(b_j_off, b_k_off+1): 1, (b_j_off+1, b_k_off+1): -1},
                 {(0, b_k_off+1): 1, (1, b_k_off+1): 1})
        
        # P4 = A11*(B10-B00) -> C00+=1, C10+=1
        add_term(idx_start+3,
                 {(1, a_j_off+1): 1},
                 {(b_j_off+1, b_k_off): 1, (b_j_off, b_k_off): -1},
                 {(0, b_k_off): 1, (1, b_k_off): 1})
        
        # P5 = (A00+A01)*B11 -> C00-=1, C01+=1
        add_term(idx_start+4,
                 {(0, a_j_off): 1, (0, a_j_off+1): 1},
                 {(b_j_off+1, b_k_off+1): 1},
                 {(0, b_k_off): -1, (0, b_k_off+1): 1})
        
        # P6 = (A10-A00)*(B00+B01) -> C11+=1
        add_term(idx_start+5,
                 {(1, a_j_off): 1, (0, a_j_off): -1},
                 {(b_j_off, b_k_off): 1, (b_j_off, b_k_off+1): 1},
                 {(1, b_k_off+1): 1})
        
        # P7 = (A01-A11)*(B10+B11) -> C00+=1
        add_term(idx_start+6,
                 {(0, a_j_off+1): 1, (1, a_j_off+1): -1},
                 {(b_j_off+1, b_k_off): 1, (b_j_off+1, b_k_off+1): 1},
                 {(0, b_k_off): 1})

    def bilinear_2x2x3_block(a_j_off, b_j_off, b_k_off, idx_start):
        """
        Add a rank-6 bilinear decomposition for a 2x2x3 block.
        
        A is 2x2, B is 2x3, C is 2x3.
        Products:
        P1 = (A00 + A10) * B00
        P2 = (A00 + A10) * B10
        P3 = A01 * B01
        P4 = A01 * B11
        P5 = (A10 + A11) * B02
        P6 = (A10 + A11) * B12
        
        Output:
        C00 = P1 + P3
        C01 = P4
        C02 = P5
        C10 = P1 + P2 + P3 + P4
        C11 = P4
        C12 = P5 + P6
        """
        # P1 = (A00+A10)*B00 -> C00+=1, C10+=1
        add_term(idx_start,
                 {(0, a_j_off): 1, (1, a_j_off): 1},
                 {(b_j_off, b_k_off): 1},
                 {(0, b_k_off): 1, (1, b_k_off): 1})
        
        # P2 = (A00+A10)*B10 -> C10+=1
        add_term(idx_start+1,
                 {(0, a_j_off): 1, (1, a_j_off): 1},
                 {(b_j_off+1, b_k_off): 1},
                 {(1, b_k_off): 1})
        
        # P3 = A01*B01 -> C00+=1, C10+=1
        add_term(idx_start+2,
                 {(0, a_j_off+1): 1},
                 {(b_j_off, b_k_off+1): 1},
                 {(0, b_k_off): 1, (1, b_k_off): 1})
        
        # P4 = A01*B11 -> C01+=1, C10+=1, C11+=1
        add_term(idx_start+3,
                 {(0, a_j_off+1): 1},
                 {(b_j_off+1, b_k_off+1): 1},
                 {(0, b_k_off+1): 1, (1, b_k_off): 1, (1, b_k_off+1): 1})
        
        # P5 = (A10+A11)*B02 -> C02+=1, C12+=1
        add_term(idx_start+4,
                 {(1, a_j_off): 1, (1, a_j_off+1): 1},
                 {(b_j_off, b_k_off+2): 1},
                 {(0, b_k_off+2): 1, (1, b_k_off+2): 1})
        
        # P6 = (A10+A11)*B12 -> C12+=1
        add_term(idx_start+5,
                 {(1, a_j_off): 1, (1, a_j_off+1): 1},
                 {(b_j_off+1, b_k_off+2): 1},
                 {(1, b_k_off+2): 1})

    # Build the decomposition
    idx = 0
    
    # Block 1: A[:, 0:2] * B[0:2, :]
    strassen_block(0, 0, 0, idx)      # A_L * B_T[0:2, 0:2]
    idx += 7
    bilinear_2x2x3_block(0, 0, 2, idx) # A_L * B_T[0:2, 2:5]
    idx += 6
    
    # Block 2: A[:, 2:4] * B[2:4, :]
    strassen_block(2, 2, 0, idx)      # A_R * B_B[2:4, 0:2]
    idx += 7
    bilinear_2x2x3_block(2, 2, 2, idx) # A_R * B_B[2:4, 2:5]
    idx += 6
    
    if idx != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {idx}")

    # Deadline guard
    if time.time() - _t0 > _DEADLINE - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
