"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify_exact(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the multiplication tensor.
    U: (n*m, r), V: (m*p, r), W: (p*n, r). Entries should be int or half-int.
    Returns True if exact."""
    r = U.shape[1]
    # Build the target tensor T implicitly: T[A,B,C]=1 if A=(i,j),B=(j,k),C=(k,i)
    # For each (i,j,k): A=i*m+j, B=j*p+k, C=k*n+i, need sum_c U[A,c]V[B,c]W[C,c] == 1
    # For all other (A,B,C): sum == 0.
    # We'll check all active triples and a sample of inactive, but to be safe check all.
    # Number of (A,B,C) = (n*m)*(m*p)*(p*n) = 15*30*18 = 8100. r up to 90. 8100*90 = 729k ops, fine.
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    # Active triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i*m+j
                B = j*p+k
                C = k*n+i
                s = 0
                for c in range(r):
                    s += Uf[A][c] * Vf[B][c] * Wf[C][c]
                if s != 1:
                    return False
    # Inactive triples: check all
    active = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                active.add((i*m+j, j*p+k, k*n+i))
    for A in range(n*m):
        for B in range(m*p):
            for C in range(p*n):
                if (A,B,C) in active:
                    continue
                s = 0
                for c in range(r):
                    s += Uf[A][c] * Vf[B][c] * Wf[C][c]
                if s != 0:
                    return False
    return True

def _to_halfint(x):
    # Round to nearest half-integer
    return round(2*x)/2

def _als_decompose(T, n, m, p, r, iters=20, seed=0):
    """ALS tensor decomposition. T is (n*m, m*p, p*n) array (0/1). Returns U,V,W."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n*m, r) * 0.1
    V = rng.randn(m*p, r) * 0.1
    W = rng.randn(p*n, r) * 0.1
    # T as array
    for it in range(iters):
        # Update U: for each A, solve (V^T W_c ...) -- actually ALS for CP: 
        # T[A,B,C] = sum_c U[A,c] V[B,c] W[C,c]
        # Fix V,W: for each A, the vector over (B,C) is sum_c U[A,c] (V[:,c] outer W[:,c])
        # So T[A] (flattened B,C) = U[A,:] @ (V outer W flattened per c)
        # Let M_c = V[:,c] outer W[:,c], flattened. Then T[A, :] = sum_c U[A,c] M_c
        # So [M_0 ... M_{r-1}] (size (m*p*p*n) x r) @ U[A,:] = T[A, :] flattened
        # Build design matrix D: (m*p * p*n) x r
        D = np.zeros((m*p * p*n, r))
        for c in range(r):
            D[:, c] = np.outer(V[:, c], W[:, c]).flatten()
        # Solve least squares: D @ X = T_flat where X is (n*m, r), T_flat is (n*m, m*p*p*n)
        Tflat = T.reshape(n*m, m*p*p*n)
        # X = (D^T D)^-1 D^T Tflat^T  -> use lstsq
        try:
            U, *_ = np.linalg.lstsq(D, Tflat.T, rcond=None)
            U = U.T  # (n*m, r)
        except Exception:
            pass
        # Update V
        D = np.zeros((n*m * p*n, r))
        for c in range(r):
            D[:, c] = np.outer(U[:, c], W[:, c]).flatten()
        Tflat = T.reshape(m*p, n*m*p*n)
        # Need T[B, A, C] for each B. Reshape: T is (A,B,C) -> (B,A,C)
        T_BAC = T.transpose(1,0,2).reshape(m*p, n*m*p*n)
        try:
            V, *_ = np.linalg.lstsq(D, T_BAC.T, rcond=None)
            V = V.T
        except Exception:
            pass
        # Update W
        D = np.zeros((n*m * m*p, r))
        for c in range(r):
            D[:, c] = np.outer(U[:, c], V[:, c]).flatten()
        T_CAB = T.transpose(2,0,1).reshape(p*n, n*m*m*p)
        try:
            W, *_ = np.linalg.lstsq(D, T_CAB.T, rcond=None)
            W = W.T
        except Exception:
            pass
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # safety margin
    # Build T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    # Schoolbook fallback
    rank = n*m*p
    U0 = np.zeros((n*m, rank))
    V0 = np.zeros((m*p, rank))
    W0 = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    best = (rank, U0.tolist(), V0.tolist(), W0.tolist())
    
    # Try lower ranks
    for r in range(40, rank):
        if time.time() > deadline - 30:
            break
        for seed in range(20):
            if time.time() > deadline - 30:
                break
            try:
                U, V, W = _als_decompose(T, n, m, p, r, iters=15, seed=seed)
            except Exception:
                continue
            # Round to half-integers
            Ur = np.round(2*U)/2
            Vr = np.round(2*V)/2
            Wr = np.round(2*W)/2
            # Verify
            if _verify_exact(Ur, Vr, Wr, n, m, p):
                best = (r, Ur.tolist(), Vr.tolist(), Wr.tolist())
                return {"U": best[1], "V": best[2], "W": best[3]}
    # Return best (schoolbook)
    return {"U": best[1], "V": best[2], "W": best[3]}
# EVOLVE-BLOCK-END
