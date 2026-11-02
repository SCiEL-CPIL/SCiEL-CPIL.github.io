"""Bootstrap SEARCH for exact tensor decomposition of 3x4x7 matrix multiplication."""
import time
import itertools
import numpy as np
import scipy.sparse.linalg as spla
import scipy.linalg as sla

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Return the best exact tensor decomposition found within a time budget.
    The interface is construct() -> {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start = time.time()
    deadline = start + 140.0  # safety margin below 150s kill

    n, m, p = DIMENSIONS

    def _schoolbook():
        """Return the classical rank-nmp decomposition."""
        R = n * m * p
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _strassen_embedded(num_blocks=6):
        """
        Construct a decomposition by replacing multiple 2x2x2 schoolbook blocks
        with Strassen's rank-7 decomposition. Each 2x2x2 block saves 1 rank.
        
        We place non-overlapping 2x2x2 blocks to maximize rank reduction.
        """
        # Define Strassen 2x2x2 decomposition
        # For a block with i_off, j_off, k_off offsets:
        # i indices: i_off, i_off+1
        # j indices: j_off, j_off+1
        # k indices: k_off, k_off+1
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # Strassen terms (relative to block-local indices 0,1):
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        # M2 = (A10+A11)B00       -> C10 += M2, C11 += M2
        # M3 = A00(B10-B11)       -> C01 += M3, C11 -= M3
        # M4 = A11(B01-B00)       -> C00 -= M4, C01 += M4
        # M5 = (A00+A10)B11       -> C00 += M5, C10 += M5
        # M6 = (A01-A00)(B00+B10) -> C11 -= M6
        # M7 = (A10-A11)(B01+B11) -> C00 += M7
        
        def strassen_block(i_off, j_off, k_off, U, V, W, col):
            """Add Strassen decomposition for block at given offsets. Returns next col."""
            i0, i1 = i_off, i_off + 1
            j0, j1 = j_off, j_off + 1
            k0, k1 = k_off, k_off + 1
            
            # Term 1: M1 = (A00+A11)(B00+B11)
            U[i0*m + j0][col] = 1
            U[i1*m + j1][col] = 1
            V[j0*p + k0][col] = 1
            V[j1*p + k1][col] = 1
            W[k0*n + i0][col] = 1
            W[k1*n + i1][col] = 1
            col += 1
            
            # Term 2: M2 = (A10+A11)B00
            U[i1*m + j0][col] = 1
            U[i1*m + j1][col] = 1
            V[j0*p + k0][col] = 1
            W[k0*n + i1][col] = 1
            W[k1*n + i1][col] = -1
            col += 1
            
            # Term 3: M3 = A00(B10-B11)
            U[i0*m + j0][col] = 1
            V[j1*p + k0][col] = 1
            V[j1*p + k1][col] = -1
            W[k1*n + i0][col] = 1
            col += 1
            
            # Term 4: M4 = A11(B01-B00)
            U[i1*m + j1][col] = 1
            V[j0*p + k1][col] = 1
            V[j0*p + k0][col] = -1
            W[k0*n + i0][col] = -1
            W[k1*n + i0][col] = 1
            col += 1
            
            # Term 5: M5 = (A00+A10)B11
            U[i0*m + j0][col] = 1
            U[i1*m + j0][col] = 1
            V[j1*p + k1][col] = 1
            W[k0*n + i0][col] = 1
            W[k0*n + i1][col] = 1
            col += 1
            
            # Term 6: M6 = (A01-A00)(B00+B10)
            U[i0*m + j1][col] = 1
            U[i0*m + j0][col] = -1
            V[j0*p + k0][col] = 1
            V[j1*p + k0][col] = 1
            W[k1*n + i1][col] = -1
            col += 1
            
            # Term 7: M7 = (A10-A11)(B01+B11)
            U[i1*m + j0][col] = 1
            U[i1*m + j1][col] = -1
            V[j0*p + k1][col] = 1
            V[j1*p + k1][col] = 1
            W[k0*n + i0][col] = 1
            W[k1*n + i1][col] = 1
            col += 1
            
            return col
        
        # Define the 6 non-overlapping Strassen blocks
        # All use i in {0,1} to avoid overlap in i-dimension
        # j blocks: {0,1} and {2,3}
        # k blocks: {0,1}, {2,3}, {4,5}
        blocks = [
            (0, 0, 0),  # i:{0,1}, j:{0,1}, k:{0,1}
            (0, 2, 0),  # i:{0,1}, j:{2,3}, k:{0,1}
            (0, 0, 2),  # i:{0,1}, j:{0,1}, k:{2,3}
            (0, 2, 2),  # i:{0,1}, j:{2,3}, k:{2,3}
            (0, 0, 4),  # i:{0,1}, j:{0,1}, k:{4,5}
            (0, 2, 4),  # i:{0,1}, j:{2,3}, k:{4,5}
        ]
        
        num_blocks = len(blocks)
        # Each block uses 7 terms, rest uses schoolbook
        # Total (i,j,k) triples: n*m*p = 84
        # Each Strassen block covers 8 triples
        # Remaining triples: 84 - 6*8 = 36
        # Total rank: 6*7 + 36 = 78
        R = num_blocks * 7 + (n * m * p - num_blocks * 8)
        
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        col = 0
        
        # Add Strassen blocks
        for i_off, j_off, k_off in blocks:
            col = strassen_block(i_off, j_off, k_off, U, V, W, col)
        
        # Mark which (i,j,k) triples are covered by Strassen blocks
        covered = set()
        for i_off, j_off, k_off in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((i_off + di, j_off + dj, k_off + dk))
        
        # Fill remaining with schoolbook terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in covered:
                        continue
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        # Verify col matches R
        if col != R:
            return _schoolbook()
            
        return {"U": U, "V": V, "W": W}

    def _verify(U, V, W):
        """Verify that the decomposition exactly reconstructs the matrix multiplication."""
        R = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rng = np.random.RandomState(42)
        for trial in range(5):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_true = A @ B
            C_approx = np.zeros((n, p))
            A_flat = A.reshape(-1)
            B_flat = B.reshape(-1)
            for r in range(R):
                ua = Ua[:, r]
                va = Va[:, r]
                wa = Wa[:, r]
                u_dot_a = ua @ A_flat
                v_dot_b = va @ B_flat
                C_approx += (wa * (u_dot_a * v_dot_b)).reshape(p, n).T
            if not np.allclose(C_true, C_approx, atol=1e-10):
                return False
        return True

    # Try the Strassen-embedded decomposition with 6 blocks
    best = _strassen_embedded(6)
    if _verify(best["U"], best["V"], best["W"]):
        return best
    
    # Fallback: try 3 Strassen blocks
    best3 = _strassen_embedded(3)
    if _verify(best3["U"], best3["V"], best3["W"]):
        return best3
    
    # Final fallback: schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
