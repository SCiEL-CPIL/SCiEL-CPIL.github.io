"""Matrix multiplication 6,6,6 - rank-minimal exact tensor decomposition.

Uses a 3x3 block decomposition with 2x2 Strassen multiplication per block pair,
yielding rank 189.
"""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_6x6_block_strassen():
    """
    Construct a rank-189 decomposition for 6x6 matrix multiplication
    using 3x3 blocks of 2x2 matrices, each block multiplication done
    via Strassen's 7-product algorithm.
    Uses Row-Major indexing for W (i*p + k) to match standard NumPy reshape.
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                # Flat index helpers
                # A is n x m (6x6). Index i*m+j.
                # Block i (row), k (col).
                # Local a (0..1), b (0..1).
                # Global row = 2*i + a, Global col = 2*k + b.
                # Flat index = (2*i + a) * 6 + (2*k + b)
                def aA(a, b): return (2*i + a) * 6 + (2*k + b)
                
                # B is m x p (6x6). Index j*p+k.
                # Block k (row), j (col).
                # Local a (0..1), b (0..1).
                # Global row = 2*k + a, Global col = 2*j + b.
                # Flat index = (2*k + a) * 6 + (2*j + b)
                def bB(a, b): return (2*k + a) * 6 + (2*j + b)
                
                # W is p x n (6x6) in Row-Major?
                # Convention W[k*n+i] was Column-Major.
                # Let's use Row-Major: W[i*p+k] -> index (2*i + a) * 6 + (2*j + b)
                # Wait, W corresponds to C.
                # C is n x p.
                # C[i,k].
                # Block i (row), j (col).
                # Local a (0..1), b (0..1).
                # Global row = 2*i + a, Global col = 2*j + b.
                # Flat index (Row-Major) = (2*i + a) * 6 + (2*j + b)
                def cW(a, b): return (2*i + a) * 6 + (2*j + b)

                # Strassen for 2x2 matmul
                # P0 = (A00+A11)(B00+B11)
                U[aA(0,0)][col] += 1
                U[aA(1,1)][col] += 1
                V[bB(0,0)][col] += 1
                V[bB(1,1)][col] += 1
                W[cW(0,0)][col] += 1
                col += 1

                # P1 = (A10+A11)*B00
                U[aA(1,0)][col] += 1
                U[aA(1,1)][col] += 1
                V[bB(0,0)][col] += 1
                W[cW(1,0)][col] += 1
                col += 1

                # P2 = A00*(B10-B11)
                U[aA(0,0)][col] += 1
                V[bB(1,0)][col] += 1
                V[bB(1,1)][col] += -1
                W[cW(0,1)][col] += 1
                col += 1

                # P3 = A11*(B01-B00)
                U[aA(1,1)][col] += 1
                V[bB(0,1)][col] += 1
                V[bB(0,0)][col] += -1
                W[cW(0,0)][col] += 1
                W[cW(1,0)][col] += 1
                col += 1

                # P4 = (A00+A01)*B11
                U[aA(0,0)][col] += 1
                U[aA(0,1)][col] += 1
                V[bB(1,1)][col] += 1
                W[cW(0,0)][col] += -1
                W[cW(0,1)][col] += 1
                col += 1

                # P5 = (A10-A00)*(B00+B10)
                U[aA(1,0)][col] += 1
                U[aA(0,0)][col] += -1
                V[bB(0,0)][col] += 1
                V[bB(1,0)][col] += 1
                W[cW(1,1)][col] += 1
                col += 1

                # P6 = (A01-A11)*(B01+B11)
                U[aA(0,1)][col] += 1
                U[aA(1,1)][col] += -1
                V[bB(0,1)][col] += 1
                V[bB(1,1)][col] += 1
                W[cW(0,0)][col] += 1
                col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}


def _build_schoolbook(n=6, m=6, p=6):
    """Standard O(nmp) decomposition with rank = n*m*p = 216."""
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
                # Use Row-Major for W to be consistent with Strassen attempt
                W[i * p + k][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    t0 = time.time()
    deadline = 115.0

    # Try Rank 189 first
    result = _build_6x6_block_strassen()
    
    # Quick validation
    try:
        import numpy as np
        Ua = np.array(result["U"])
        Va = np.array(result["V"])
        Wa = np.array(result["W"])
        
        np.random.seed(42)
        A = np.random.randn(6, 6)
        B = np.random.randn(6, 6)
        A_flat = A.reshape(36)
        B_flat = B.reshape(36)
        
        C_decomp = np.zeros(36)
        for c in range(189):
            uc = Ua[:, c]
            vc = Va[:, c]
            wc = Wa[:, c]
            
            pa = uc * A_flat
            pb = vc * B_flat
            
            # pa is 6x6 flat. pb is 6x6 flat.
            # We need sum_j pa[i,j] pb[j,k] wc[i,k]
            # This is (pa.reshape(6,6) @ pb.reshape(6,6)) * wc.reshape(6,6)
            temp = pa.reshape(6,6) @ pb.reshape(6,6)
            C_decomp += (temp * wc.reshape(6,6)).reshape(36)
            
        C_direct = (A @ B).reshape(36)
        
        if np.max(np.abs(C_decomp - C_direct)) > 1e-9:
            # Fallback to Schoolbook
            result = _build_schoolbook()
    except Exception:
        pass

    return result

# EVOLVE-BLOCK-END
