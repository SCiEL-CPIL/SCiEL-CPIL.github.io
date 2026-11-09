"""Matrix multiplication 2x4x8 rank-minimal exact tensor decomposition.

Uses a block-level Strassen decomposition: treat the 2x4 * 4x8 multiplication
as a 2x2 multiplication of sub-blocks, where each sub-block is 1x2 (from A)
and 2x4 (from B), producing 2x4 sub-blocks in C.

A is 2x4: split into 2x2 blocks of size (1,2):
  A = [[A00, A01],
       [A10, A11]]
  where Aij is 1x2.

B is 4x8: split into 2x2 blocks of size (2,4):
  B = [[B00, B01],
       [B10, B11]]
  where Bij is 2x4.

C is 2x8: split into 2x2 blocks of size (1,4):
  C = [[C00, C01],
       [C10, C11]]
  where Cij is 1x4.

The product C = A*B means:
  C00 = A00*B00 + A01*B10   (1x2 * 2x4 + 1x2 * 2x4 = 1x4)
  C01 = A00*B01 + A01*B11
  C10 = A10*B00 + A11*B10
  C11 = A10*B01 + A11*B11

Strassen's 7 products for the 2x2 block structure:
  M1 = (A00+A11)*(B00+B11)
  M2 = (A10+A11)*B00
  M3 = A00*(B01-B11)
  M4 = A11*(B10-B00)
  M5 = (A00+A01)*B11
  M6 = (A10-A00)*(B00+B01)
  M7 = (A01-A11)*(B10+B11)

Then:
  C00 = M1 + M4 - M5 + M7
  C01 = M3 + M5
  C10 = M2 + M4
  C11 = M1 - M2 + M3 + M6

Each Mi is a (1x2)*(2x4) product. Each such product has schoolbook rank 1*2*4 = 8.
Total rank = 7 * 8 = 56.
"""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]
DEADLINE = 140.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (32, rank)  
    # W shape: (p*n, rank) = (16, rank)
    
    # Define the 7 Strassen products explicitly.
    products = []
    
    # M1 = (A00+A11)*(B00+B11)
    products.append({
        'left': [
            [(0, 0, 1), (1, 2, 1)],
            [(0, 1, 1), (1, 3, 1)]
        ],
        'right': [
            [(0, 0, 1), (2, 4, 1)], [(0, 1, 1), (2, 5, 1)], [(0, 2, 1), (2, 6, 1)], [(0, 3, 1), (2, 7, 1)],
            [(1, 0, 1), (3, 4, 1)], [(1, 1, 1), (3, 5, 1)], [(1, 2, 1), (3, 6, 1)], [(1, 3, 1), (3, 7, 1)]
        ],
        'c_contrib': [
            [(0, 0, 1), (1, 4, 1)],
            [(0, 1, 1), (1, 5, 1)],
            [(0, 2, 1), (1, 6, 1)],
            [(0, 3, 1), (1, 7, 1)]
        ]
    })
    
    # M2 = (A10+A11)*B00
    products.append({
        'left': [
            [(1, 0, 1), (1, 2, 1)],
            [(1, 1, 1), (1, 3, 1)]
        ],
        'right': [
            [(0, 0, 1)], [(0, 1, 1)], [(0, 2, 1)], [(0, 3, 1)],
            [(1, 0, 1)], [(1, 1, 1)], [(1, 2, 1)], [(1, 3, 1)]
        ],
        'c_contrib': [
            [(1, 0, 1), (1, 4, -1)],
            [(1, 1, 1), (1, 5, -1)],
            [(1, 2, 1), (1, 6, -1)],
            [(1, 3, 1), (1, 7, -1)]
        ]
    })
    
    # M3 = A00*(B01-B11)
    products.append({
        'left': [
            [(0, 0, 1)],
            [(0, 1, 1)]
        ],
        'right': [
            [(0, 4, 1), (2, 4, -1)], [(0, 5, 1), (2, 5, -1)], [(0, 6, 1), (2, 6, -1)], [(0, 7, 1), (2, 7, -1)],
            [(1, 4, 1), (3, 4, -1)], [(1, 5, 1), (3, 5, -1)], [(1, 6, 1), (3, 6, -1)], [(1, 7, 1), (3, 7, -1)]
        ],
        'c_contrib': [
            [(0, 4, 1), (1, 4, 1)],
            [(0, 5, 1), (1, 5, 1)],
            [(0, 6, 1), (1, 6, 1)],
            [(0, 7, 1), (1, 7, 1)]
        ]
    })
    
    # M4 = A11*(B10-B00)
    products.append({
        'left': [
            [(1, 2, 1)],
            [(1, 3, 1)]
        ],
        'right': [
            [(2, 0, 1), (0, 0, -1)], [(2, 1, 1), (0, 1, -1)], [(2, 2, 1), (0, 2, -1)], [(2, 3, 1), (0, 3, -1)],
            [(3, 0, 1), (1, 0, -1)], [(3, 1, 1), (1, 1, -1)], [(3, 2, 1), (1, 2, -1)], [(3, 3, 1), (1, 3, -1)]
        ],
        'c_contrib': [
            [(0, 0, 1), (1, 0, 1)],
            [(0, 1, 1), (1, 1, 1)],
            [(0, 2, 1), (1, 2, 1)],
            [(0, 3, 1), (1, 3, 1)]
        ]
    })
    
    # M5 = (A00+A01)*B11
    products.append({
        'left': [
            [(0, 0, 1), (0, 2, 1)],
            [(0, 1, 1), (0, 3, 1)]
        ],
        'right': [
            [(2, 4, 1)], [(2, 5, 1)], [(2, 6, 1)], [(2, 7, 1)],
            [(3, 4, 1)], [(3, 5, 1)], [(3, 6, 1)], [(3, 7, 1)]
        ],
        'c_contrib': [
            [(0, 0, -1), (0, 4, 1)],
            [(0, 1, -1), (0, 5, 1)],
            [(0, 2, -1), (0, 6, 1)],
            [(0, 3, -1), (0, 7, 1)]
        ]
    })
    
    # M6 = (A10-A00)*(B00+B01)
    products.append({
        'left': [
            [(1, 0, 1), (0, 0, -1)],
            [(1, 1, 1), (0, 1, -1)]
        ],
        'right': [
            [(0, 0, 1), (0, 4, 1)], [(0, 1, 1), (0, 5, 1)], [(0, 2, 1), (0, 6, 1)], [(0, 3, 1), (0, 7, 1)],
            [(1, 0, 1), (1, 4, 1)], [(1, 1, 1), (1, 5, 1)], [(1, 2, 1), (1, 6, 1)], [(1, 3, 1), (1, 7, 1)]
        ],
        'c_contrib': [
            [(1, 4, 1)],
            [(1, 5, 1)],
            [(1, 6, 1)],
            [(1, 7, 1)]
        ]
    })
    
    # M7 = (A01-A11)*(B10+B11)
    products.append({
        'left': [
            [(0, 2, 1), (1, 2, -1)],
            [(0, 3, 1), (1, 3, -1)]
        ],
        'right': [
            [(2, 0, 1), (2, 4, 1)], [(2, 1, 1), (2, 5, 1)], [(2, 2, 1), (2, 6, 1)], [(2, 3, 1), (2, 7, 1)],
            [(3, 0, 1), (3, 4, 1)], [(3, 1, 1), (3, 5, 1)], [(3, 2, 1), (3, 6, 1)], [(3, 3, 1), (3, 7, 1)]
        ],
        'c_contrib': [
            [(0, 0, 1)],
            [(0, 1, 1)],
            [(0, 2, 1)],
            [(0, 3, 1)]
        ]
    })
    
    # Build U, V, W
    rank = 7 * 8  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for prod in products:
        for j in range(2):
            for k in range(4):
                for (a_row, a_col, coeff) in prod['left'][j]:
                    U[a_row * m + a_col][col] += coeff
                for (b_row, b_col, coeff) in prod['right'][j * 4 + k]:
                    V[b_row * p + b_col][col] += coeff
                for (c_row, c_col, coeff) in prod['c_contrib'][k]:
                    W[c_col * n + c_row][col] += coeff
                col += 1
    
    if time.time() - start_time > DEADLINE:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    n, m, p = DIMENSIONS
    
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0
            for j in range(m):
                for r in range(U.shape[1]):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_decomp[i,k] = total
    
    error = np.max(np.abs(C_true - C_decomp))
    print(f"Max error: {error}")
    print(f"Rank: {U.shape[1]}")
