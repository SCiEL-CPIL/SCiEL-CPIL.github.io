import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145.0
    
    # Strassen terms for 2x2 * 2x2
    # Inputs: X (2x2), Y (2x2)
    # Indices in X: 00, 01, 10, 11 -> local 0, 1, 2, 3
    # Indices in Y: 00, 01, 10, 11 -> local 0, 1, 2, 3
    # Indices in Z: 00, 01, 10, 11 -> local 0, 1, 2, 3
    
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y10-y11)
    # M4 = x11(y01-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    strassen_defs = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # Local indices: 0=x00/y00/z00, 1=x01/y01/z01, 2=x10/y10/z10, 3=x11/y11/z11
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 2: 1, 3: -1}), # M1 -> z00(1), z11(1) ... wait.
        # Let's stick to the explicit equations.
        
        # M1 contributes to Z00 (+1) and Z11 (+1)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        
        # M2 contributes to Z10 (+1) and Z11 (-1)
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        
        # M3 contributes to Z01 (+1) and Z11 (+1)
        ({0: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}),
        
        # M4 contributes to Z00 (+1) and Z10 (+1)
        ({3: 1}, {1: 1, 0: -1}, {0: 1, 2: 1}),
        
        # M5 contributes to Z00 (-1) and Z01 (+1)
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        
        # M6 contributes to Z11 (+1)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        
        # M7 contributes to Z00 (+1)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def build_strassen_28():
        U = np.zeros((8, 28), dtype=float)
        V = np.zeros((16, 28), dtype=float)
        W = np.zeros((8, 28), dtype=float)
        
        # Mapping local 2x2 indices to global indices
        # A_L: 0,1,4,5 (x00, x01, x10, x11)
        # A_R: 2,3,6,7 (x00, x01, x10, x11)
        
        # B_TL: 0,1,4,5 (y00, y01, y10, y11)
        # B_TR: 2,3,6,7 (y00, y01, y10, y11)
        # B_BL: 8,9,12,13 (y00, y01, y10, y11)
        # B_BR: 10,11,14,15 (y00, y01, y10, y11)
        
        # C_L: W indices 0,2,1,3 (z00, z01, z10, z11)
        # C_R: W indices 4,6,5,7 (z00, z01, z10, z11)
        
        u_maps = {
            'L': [0, 1, 4, 5],
            'R': [2, 3, 6, 7]
        }
        
        v_maps = {
            'TL': [0, 1, 4, 5],
            'TR': [2, 3, 6, 7],
            'BL': [8, 9, 12, 13],
            'BR': [10, 11, 14, 15]
        }
        
        # W local indices 0,1,2,3 correspond to z00, z01, z10, z11
        # Global W index = k*n + i
        # C_L (k=0,1):
        # z00 (i=0,k=0) -> 0
        # z01 (i=0,k=1) -> 2
        # z10 (i=1,k=0) -> 1
        # z11 (i=1,k=1) -> 3
        # C_R (k=2,3):
        # z00 (i=0,k=2) -> 4
        # z01 (i=0,k=3) -> 6
        # z10 (i=1,k=2) -> 5
        # z11 (i=1,k=3) -> 7
        
        w_map_L = [0, 2, 1, 3]
        w_map_R = [4, 6, 5, 7]
        
        col = 0
        # Block 1: A_L * B_TL -> C_L
        for u_c, v_c, w_c in strassen_defs:
            for loc_idx, val in u_c.items():
                U[u_maps['L'][loc_idx], col] += val
            for loc_idx, val in v_c.items():
                V[v_maps['TL'][loc_idx], col] += val
            for loc_idx, val in w_c.items():
                W[w_map_L[loc_idx], col] += val
            col += 1
            
        # Block 2: A_R * B_BL -> C_L
        for u_c, v_c, w_c in strassen_defs:
            for loc_idx, val in u_c.items():
                U[u_maps['R'][loc_idx], col] += val
            for loc_idx, val in v_c.items():
                V[v_maps['BL'][loc_idx], col] += val
            for loc_idx, val in w_c.items():
                W[w_map_L[loc_idx], col] += val
            col += 1
                
        # Block 3: A_L * B_TR -> C_R
        for u_c, v_c, w_c in strassen_defs:
            for loc_idx, val in u_c.items():
                U[u_maps['L'][loc_idx], col] += val
            for loc_idx, val in v_c.items():
                V[v_maps['TR'][loc_idx], col] += val
            for loc_idx, val in w_c.items():
                W[w_map_R[loc_idx], col] += val
            col += 1
                
        # Block 4: A_R * B_BR -> C_R
        for u_c, v_c, w_c in strassen_defs:
            for loc_idx, val in u_c.items():
                U[u_maps['R'][loc_idx], col] += val
            for loc_idx, val in v_c.items():
                V[v_maps['BR'][loc_idx], col] += val
            for loc_idx, val in w_c.items():
                W[w_map_R[loc_idx], col] += val
            col += 1
        
        return U, V, W

    U28, V28, W28 = build_strassen_28()
    
    def verify(U, V, W, tol=1e-9):
        n, m, p = 2, 4, 4
        A = np.arange(n * m).reshape(n, m) + 1.0
        B = np.arange(m * p).reshape(m, p) + 2.0
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(U.shape[1]):
                    for jj in range(m):
                        acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
                Cf[ii, kk] = acc
        return np.allclose(C, Cf, atol=tol)

    if not verify(U28, V28, W28):
        raise AssertionError("Rank-28 solution failed verification")

    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
