import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Rank 28 decomposition using Strassen on 2x2 blocks
    # A is 2x4, B is 4x4, C is 2x4
    # C = A * B
    # Partition A into [A1 A2] (2x2 each)
    # Partition B into [[B11 B12], [B21 B22]] (2x2 each)
    # C[:, 0:2] = A1*B11 + A2*B21
    # C[:, 2:4] = A1*B12 + A2*B22
    
    # Strassen coefficients for X*Y (2x2)
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    sx = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    sy = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    sz = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1}
    ]
    
    # Sub-products: (a_row0, a_col0, b_row0, b_col0, c_col0)
    # A is 2x4, so a_row0 is always 0.
    # B is 4x4.
    # C is 2x4.
    sub_products = [
        (0, 0, 0, 0, 0),   # A1 * B11 -> C[:, 0:2]
        (0, 2, 2, 0, 0),   # A2 * B21 -> C[:, 0:2]
        (0, 0, 0, 2, 2),   # A1 * B12 -> C[:, 2:4]
        (0, 2, 2, 2, 2)    # A2 * B22 -> C[:, 2:4]
    ]
    
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for sp_idx, (ar0, ac0, br0, bc0, cc0) in enumerate(sub_products):
        for s in range(7):
            col = sp_idx * 7 + s
            
            for x_idx, val in sx[s].items():
                xi = x_idx // 2
                xj = x_idx % 2
                a_idx = (ar0 + xi) * m + (ac0 + xj)
                U[a_idx, col] += val
            
            for y_idx, val in sy[s].items():
                yi = y_idx // 2
                yj = y_idx % 2
                b_idx = (br0 + yi) * p + (bc0 + yj)
                V[b_idx, col] += val
            
            for z_idx, val in sz[s].items():
                zi = z_idx // 2
                zj = z_idx % 2
                c_idx = (cc0 + zj) * n + zi
                W[c_idx, col] += val
                
            if time.time() - start > 130:
                break
        if time.time() - start > 130:
            break
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"U shape: {len(result['U'])}x{len(result['U'][0])}")
    print(f"V shape: {len(result['V'])}x{len(result['V'][0])}")
    print(f"W shape: {len(result['W'])}x{len(result['W'][0])}")
