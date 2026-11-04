"""Rewritten program for matrix multiplication tensor rank search (3x4 by 4x4)."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _build_target_tensor(n, m, p):
    """Return the 3D tensor T[i,j,k] = 1."""
    return np.ones((n, m, p), dtype=np.float64)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if U,V,W exactly decompose the tensor (within tol)."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _construct_structured_decomposition():
    """
    Construct a rank-44 decomposition using Strassen for 2x2x2 blocks
    and schoolbook for 1x2x2 blocks.
    """
    n, m, p = 3, 4, 4
    # Total rank estimate: 4 * 7 (for 2x2x2) + 4 * 4 (for 1x2x2) = 28 + 16 = 44
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen(rA, cA, rB, cB):
        nonlocal col
        # Strassen 7 terms
        # X is A[rA:rA+2, cA:cA+2]
        # Y is B[rB:rB+2, cB:cB+2]
        # Z is C[rA:rA+2, cB:cB+2]
        
        # M1 = (X11+X22)(Y11+Y22)
        # U: X11, X22 ; V: Y11, Y22 ; W: Z11, Z22, Z12, Z21, Z22?
        # We need to map to global indices.
        
        # Indices in sub-matrix:
        # X: (0,0), (0,1), (1,0), (1,1)
        # Y: (0,0), (0,1), (1,0), (1,1)
        # Z: (0,0), (0,1), (1,0), (1,1)
        
        # Global U index: (rA + i, cA + j) -> (rA+i)*m + (cA+j)
        # Global V index: (rB + j, cB + k) -> (rB+j)*p + (cB+k)
        # Global W index: (cB + k, rA + i) -> (cB+k)*n + (rA+i)
        
        # Term 1: M1 = (X11+X22)(Y11+Y22)
        # U: X11=1, X22=1
        # V: Y11=1, Y22=1
        # W: Z11=1, Z22=1, Z12=1, Z21=1 (Wait, M1 contributes to Z11, Z22, Z12? No)
        # Z11 = M1 + M4 - M5 + M7
        # Z12 = M3 + M5
        # Z21 = M2 + M4
        # Z22 = M1 - M2 + M3 + M6
        
        # So M1 contributes to Z11 (+1) and Z22 (+1).
        
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1 # Z11
        W[(cB+1)*n + (rA+1), col] = 1 # Z22
        col += 1
        
        # Term 2: M2 = (X21+X22)Y11
        # U: X21=1, X22=1
        # V: Y11=1
        # W: Z21=1, Z22=-1
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1 # Z21
        W[(cB+1)*n + (rA+1), col] = -1 # Z22
        col += 1
        
        # Term 3: M3 = X11(Y12-Y22)
        # U: X11=1
        # V: Y12=1, Y22=-1
        # W: Z12=1, Z22=1
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = -1
        W[(cB+1)*n + (rA+0), col] = 1 # Z12
        W[(cB+1)*n + (rA+1), col] = 1 # Z22
        col += 1
        
        # Term 4: M4 = X22(Y21-Y11)
        # U: X22=1
        # V: Y21=1, Y11=-1
        # W: Z11=1, Z21=1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = -1
        W[(cB+0)*n + (rA+0), col] = 1 # Z11
        W[(cB+0)*n + (rA+1), col] = 1 # Z21
        col += 1
        
        # Term 5: M5 = (X11+X12)Y22
        # U: X11=1, X12=1
        # V: Y22=1
        # W: Z12=1, Z11=-1
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1 # Z12
        W[(cB+0)*n + (rA+0), col] = -1 # Z11
        col += 1
        
        # Term 6: M6 = (X21-X11)(Y11+Y12)
        # U: X21=1, X11=-1
        # V: Y11=1, Y12=1
        # W: Z22=1
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+0), col] = -1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1 # Z22
        col += 1
        
        # Term 7: M7 = (X12-X22)(Y21+Y22)
        # U: X12=1, X22=-1
        # V: Y21=1, Y22=1
        # W: Z11=1
        U[(rA+0)*m + (cA+1), col] = 1
        U[(rA+1)*m + (cA+1), col] = -1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1 # Z11
        col += 1

    # Helper to add schoolbook 1x2x2
    def add_schoolbook_1x2x2(rA, cA, rB, cB):
        nonlocal col
        # A is 1x2, B is 2x2, C is 1x2
        # C0 = A0 B00 + A1 B10
        # C1 = A0 B01 + A1 B11
        
        # Term 1: A0 B00 -> C0
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1
        
        # Term 2: A1 B10 -> C0
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1
        
        # Term 3: A0 B01 -> C1
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        col += 1
        
        # Term 4: A1 B11 -> C1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        col += 1

    # Apply blocks
    # Top-Left (2x2 x 2x2) split into two 2x2x2
    add_strassen(0, 0, 0, 0)
    add_strassen(0, 2, 2, 0)
    
    # Top-Right (2x2 x 2x2) split into two 2x2x2
    add_strassen(0, 0, 0, 2)
    add_strassen(0, 2, 2, 2)
    
    # Bottom-Left (1x2 x 2x2) split into two 1x2x2
    add_schoolbook_1x2x2(2, 0, 0, 0)
    add_schoolbook_1x2x2(2, 2, 2, 0)
    
    # Bottom-Right (1x2 x 2x2) split into two 1x2x2
    add_schoolbook_1x2x2(2, 0, 0, 2)
    add_schoolbook_1x2x2(2, 2, 2, 2)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    deadline = time.time() + 110
    result = _construct_structured_decomposition()
    
    # Verify
    n, m, p = DIMENSIONS
    U, V, W = result["U"], result["V"], result["W"]
    # Convert to numpy for verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    if _verify_decomposition(U_np, V_np, W_np, n, m, p, tol=1e-6):
        return result
    else:
        # Fallback to schoolbook if verification fails (should not happen)
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
