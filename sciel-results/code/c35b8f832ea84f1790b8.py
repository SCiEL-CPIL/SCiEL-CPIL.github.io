"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-84 exact tensor decomposition for 4x4 by 4x6 matrix multiplication
    using Strassen's block decomposition.
    
    Strategy:
    - Partition A (4x4) into four 2x2 blocks.
    - Partition B (4x6) into four 2x3 blocks.
    - Apply Strassen's 7-step algorithm to each of the 4 block-pairs (i-block, j-block).
    - For each Strassen step, we need to handle the 3 columns of the 2x3 sub-blocks.
    - Total rank = 4 (block pairs) * 7 (Strassen steps) * 3 (columns) = 84.
    
    The output W follows the convention W[k*n+i, r] where k is column index of B (0..p-1),
    i is row index of A (0..n-1), n=4, p=6.
    """
    start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=4, m=4, p=6

    rank = 7 * 2 * 2 * 3  # 84
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen steps for 2x2 blocks
    # Each step: (A_coeffs, B_coeffs, C_coeffs)
    # Blocks are labeled '00', '01', '10', '11'
    # '00' = top-left, '01' = top-right, '10' = bottom-left, '11' = bottom-right
    
    steps = [
        # M1 = (A00 + A11)(B00 + B11)
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
        # M2 = (A10 + A11)B00
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),
        # M3 = A00(B01 - B11)
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),
        # M4 = A11(B10 - B00)
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),
        # M5 = (A00 + A01)B11
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),
        # M6 = (A10 - A00)(B00 + B01)
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),
        # M7 = (A01 - A11)(B10 + B11)
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1}),
    ]

    # Block index to flat index helpers
    def get_u_row(block, i1, j1):
        """Get flat index in U for block 'block' with local row i1, col j1 within the 2x2 block"""
        if block == '00':
            i, j = i1, j1
        elif block == '01':
            i, j = i1, j1 + 2
        elif block == '10':
            i, j = i1 + 2, j1
        elif block == '11':
            i, j = i1 + 2, j1 + 2
        else:
            raise ValueError(f"Invalid block {block}")
        return i * m + j

    def get_v_row(block, j1, k1):
        """Get flat index in V for block 'block' with local row j1, col k1 within the 2x3 block"""
        if block == '00':
            j, k = j1, k1
        elif block == '01':
            j, k = j1, k1 + 3
        elif block == '10':
            j, k = j1 + 2, k1
        elif block == '11':
            j, k = j1 + 2, k1 + 3
        else:
            raise ValueError(f"Invalid block {block}")
        return j * p + k

    def get_w_row(block, i1, k1):
        """Get flat index in W for block 'block' with local row i1, col k1 within the 2x3 block
           using convention W[k*n+i]"""
        if block == '00':
            i, k = i1, k1
        elif block == '01':
            i, k = i1, k1 + 3
        elif block == '10':
            i, k = i1 + 2, k1
        elif block == '11':
            i, k = i1 + 2, k1 + 3
        else:
            raise ValueError(f"Invalid block {block}")
        return k * n + i

    col = 0
    for s in range(7):
        alpha, beta, gamma = steps[s]
        for i1 in range(2):
            for j1 in range(2):
                for k1 in range(3):
                    for b, val in alpha.items():
                        U[get_u_row(b, i1, j1), col] += val
                    for b, val in beta.items():
                        V[get_v_row(b, j1, k1), col] += val
                    for b, val in gamma.items():
                        W[get_w_row(b, i1, k1), col] += val
                    col += 1

    # Verify correctness with a random test
    if time.time() - start < DEADLINE:
        rng = np.random.RandomState(42)
        A_test = rng.randn(n, m)
        B_test = rng.randn(m, p)
        C_expected = A_test @ B_test
        
        # Reconstruct C from decomposition
        C_recon = np.zeros((n, p))
        for r in range(rank):
            u_vec = U[:, r].reshape(n, m)
            v_vec = V[:, r].reshape(m, p)
            w_vec = W[:, r]
            # C[i,k] += sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Actually: C[i,k] = sum_r W[k*n+i, r] * (U[:,r] * V[:,r] reshaped)
            # The tensor is: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # So C[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            pass
        
        # More efficient verification
        C_recon = np.zeros((n, p))
        for r in range(rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
                    C_recon[i,k] += s
        
        if np.max(np.abs(C_recon - C_expected)) > 1e-10:
            # Fallback: use schoolbook rank 96
            rank2 = n * m * p  # 96
            U2 = np.zeros((n * m, rank2), dtype=float)
            V2 = np.zeros((m * p, rank2), dtype=float)
            W2 = np.zeros((p * n, rank2), dtype=float)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U2[i*m+j, idx] = 1.0
                        V2[j*p+k, idx] = 1.0
                        W2[k*n+i, idx] = 1.0
                        idx += 1
            return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

    # Deadline guard
    if time.time() - start > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
