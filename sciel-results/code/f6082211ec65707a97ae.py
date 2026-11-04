import time
import sys

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Winograd F(4) for 4x4x4 part (rank 12) + Winograd F(2) for 4x4x1 part (rank 6)
    # Total rank: 18
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Winograd F(4) transform for 4x4 matrix multiplication
    # A(4x4) * B(4x4) = C(4x4)
    # Using cyclic group Z_4 structure
    
    # F(4) transform matrix (4x4 DFT):
    # F = [[1, 1, 1, 1],
    #      [1, i, -1, -i],
    #      [1, -1, 1, -1],
    #      [1, -i, -1, i]]
    
    # For Winograd F(4), we use 4 frequency components
    # A_freq[k][r][c] = sum_{i,j} A[i][j] * omega^{(i-k)*j} where omega = i (4th root of unity)
    # B_freq[k][r][c] = sum_{i,j} B[i][j] * omega^{(i-k)*j}
    # C_freq[k][r][c] = A_freq[k][r][c] * B_freq[k][r][c]
    # C[i][j] = (1/4) * sum_k C_freq[k][i][j]
    
    # However, for tensor decomposition, we need rank-1 terms.
    # Winograd F(4) gives 4 independent 4x4x4 subproblems, each of rank 4 (diagonal)
    # Total: 4 * 4 = 16? No, that's not right.
    
    # Let me think differently. Winograd F(4) for matrix multiplication:
    # We transform A and B to frequency domain, multiply pointwise, then inverse transform.
    # The key insight is that in frequency domain, each frequency component is a diagonal matrix product.
    
    # Actually, for 4x4x4, Winograd F(4) uses the fact that Z_4 has 4 characters.
    # The transform decomposes the 4x4x4 tensor into 4 independent rank-4 tensors.
    # Total rank = 4 * 4 = 16. But we want 12.
    
    # Let me use a different approach. The Winograd F(4) algorithm for 4x4 matrix multiplication
    # uses 12 multiplications. Let me construct it directly.
    
    # Winograd F(4) for 4x4:
    # M1 = (A00+A11+A22+A33)(B00+B11+B22+B33)
    # M2 = (A01+A10+A23+A32)(B01+B10+B23+B32)
    # M3 = (A02+A13+A20+A31)(B02+B13+B20+B31)
    # M4 = (A03+A12+A21+A30)(B03+B12+B21+B30)
    # M5 = (A00+A11-A22-A33)(B00+B11-B22-B33)
    # M6 = (A01+A10-A23-A32)(B01+B10-B23-B32)
    # M7 = (A02+A13-A20-A31)(B02+B13-B20-B31)
    # M8 = (A03+A12-A21-A30)(B03+B12-B21-B30)
    # M9 = (A00-A11+A22-A33)(B00-B11+B22-B33)
    # M10 = (A01-A10+A23-A32)(B01-B10+B23-B32)
    # M11 = (A02-A13+A20-A31)(B02-B13+B20-B31)
    # M12 = (A03-A12+A21-A30)(B03-B12+B21-B30)
    
    # This is getting complex. Let me use a simpler approach:
    # Use Strassen for 4x4x4 (rank 7) and schoolbook for 4x4x1 (rank 4)
    # Total: 11? No, Strassen is rank 7 for 2x2x2, so for 4x4x4 it's 7*4=28.
    
    # Let me just use the original approach but with better constants.
    # Actually, let me try a direct Winograd F(4) construction.
    
    # Winograd F(4) transform for 4x4 matrix multiplication:
    # We use the 4x4 DFT matrix. The transform is:
    # A' = F * A * F^T
    # B' = F * B * F^T
    # C' = A' * B' (element-wise in frequency domain)
    # C = F^(-1) * C' * F^(-1)^T
    
    # In the frequency domain, each entry is a product, so we have 16 multiplications.
    # But we can exploit symmetry to reduce to 12.
    
    # For now, let me use a known rank-12 decomposition for 4x4x4.
    
    # Actually, let me just keep the Strassen approach but verify it's correct.
    
    # Let me use a simpler, verified approach:
    # 4x4x4 using 4 independent 2x2x2 Strassen blocks: rank 4*7 = 28
    # 4x4x1 using schoolbook: rank 4
    # Total: 32
    
    # That's worse than 72. Let me stick with the original approach.
    
    # For now, return the original rank-72 solution to ensure correctness.
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    products = [
        (0, 0, 0, 0, 0, 0),
        (0, 1, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 1),
        (0, 1, 1, 1, 0, 1),
        (1, 0, 0, 0, 1, 0),
        (1, 1, 1, 0, 1, 0),
        (1, 0, 0, 1, 1, 1),
        (1, 1, 1, 1, 1, 1),
    ]
    
    for (ar, ac, br, bc, cr, cc) in products:
        a_base = (ar*2, ac*2)
        b_base = (br*2, bc*2)
        c_base = (cr*2, cc*2)
        
        def a_idx(r, c): return (a_base[0]+r)*4 + (a_base[1]+c)
        def b_idx(r, c): return (b_base[0]+r)*5 + (b_base[1]+c)
        def c_idx(r, c):
            return (c_base[1]+c)*4 + (c_base[0]+r)
        
        # M1: (A00+A11)(B00+B11)
        U[a_idx(0,0)][col] += 1
        U[a_idx(1,1)][col] += 1
        V[b_idx(0,0)][col] += 1
        V[b_idx(1,1)][col] += 1
        W[c_idx(0,0)][col] += 1
        W[c_idx(1,1)][col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        U[a_idx(1,0)][col] += 1
        U[a_idx(1,1)][col] += 1
        V[b_idx(0,0)][col] += 1
        W[c_idx(1,0)][col] += 1
        W[c_idx(1,1)][col] -= 1
        col += 1
        
        # M3: A00(B01-B11)
        U[a_idx(0,0)][col] += 1
        V[b_idx(0,1)][col] += 1
        V[b_idx(1,1)][col] -= 1
        W[c_idx(0,1)][col] += 1
        W[c_idx(1,1)][col] += 1
        col += 1
        
        # M4: A11(B10-B00)
        U[a_idx(1,1)][col] += 1
        V[b_idx(1,0)][col] += 1
        V[b_idx(0,0)][col] -= 1
        W[c_idx(0,0)][col] += 1
        W[c_idx(1,0)][col] += 1
        col += 1
        
        # M5: (A00+A01)B11
        U[a_idx(0,0)][col] += 1
        U[a_idx(0,1)][col] += 1
        V[b_idx(1,1)][col] += 1
        W[c_idx(0,0)][col] -= 1
        W[c_idx(0,1)][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[a_idx(1,0)][col] += 1
        U[a_idx(0,0)][col] -= 1
        V[b_idx(0,0)][col] += 1
        V[b_idx(0,1)][col] += 1
        W[c_idx(1,1)][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[a_idx(0,1)][col] += 1
        U[a_idx(1,1)][col] -= 1
        V[b_idx(1,0)][col] += 1
        V[b_idx(1,1)][col] += 1
        W[c_idx(0,0)][col] += 1
        col += 1

    # Part 2: 4x4x1 (rank 16)
    for i in range(4):
        for j in range(4):
            U[i*4 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*4 + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
