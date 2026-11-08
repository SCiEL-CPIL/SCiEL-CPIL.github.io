import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 4, 4, 7
    
    # For 4x4 * 4x7 matrix multiplication, we need rank-r decomposition.
    # The tensor is T_{i,j,k} = delta_{i,j} * delta_{j,k} (standard matmul tensor)
    # but here A is 4x4, B is 4x7, C is 4x7.
    # T_{i, (j,k)} where i in 0..3 (row of C), j in 0..3 (col of A / row of B), k in 0..6 (col of C)
    # Actually the tensor is: C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # So the tensor T has indices (i, j, k) with T[i,j,k] = 1 if the multiplication uses A[i,j]*B[j,k]
    # The tensor rank is what we're minimizing.
    
    # The trivial (schoolbook) rank is 4*4*7 = 112.
    # Using Strassen-like approaches, we can do better.
    
    # Strategy: Use known optimal or near-optimal decompositions.
    # For 4x4 * 4x4, the best known is 49 (Strassen recursive) or potentially lower.
    # Actually for 4x4*4x4, Strassen gives 49. There are known results for smaller ranks.
    
    # Let me think about this differently. The matrix multiplication tensor for m x n * n x p
    # has rank at least max(m*p, m*n, n*p) and at most m*n*p.
    
    # For 4x4 * 4x7:
    # Lower bound: max(4*7, 4*4, 4*7) = max(28, 16, 28) = 28
    # Schoolbook: 4*4*7 = 112
    
    # A good approach: decompose the 4x7 into 4x4 + 4x2 + 4x1
    # 4x4*4x4: use Strassen rank 49
    # 4x4*4x2: use block Strassen, rank 4*7 = 28 (4 blocks of 2x2*2x2)
    # 4x4*4x1: schoolbook rank 16
    # Total: 49 + 28 + 16 = 93
    
    # But we can do better by being smarter.
    
    # Alternative: Use the fact that for multiplying by a matrix with 7 columns,
    # we can group columns and use Strassen-like tricks.
    
    # Let me try a direct approach: build the tensor and try to find a low-rank decomposition
    # using integer/half-integer entries.
    
    # Actually, let me use a known result: for 4x4*4x4, the Strassen decomposition has rank 49.
    # For the remaining 3 columns (4x4*4x3), we need additional terms.
    
    # Better approach: Use the Laderman decomposition or other known decompositions.
    
    # Let me just use the recursive Strassen approach for the 4x4*4x4 part (rank 49)
    # and schoolbook for the 4x4*4x3 part (rank 4*4*3 = 48)
    # Total: 49 + 48 = 97
    
    # Even better: for 4x4*4x2, use 4 blocks of 2x2*2x2 Strassen = 4*7 = 28
    # For 4x4*4x1, schoolbook = 16
    # Total: 49 + 28 + 16 = 93
    
    # Let me try to do even better. What if I use a 2x4*4x4 block decomposition?
    # Or use the fact that we can share some Strassen terms across column groups.
    
    # Actually, let me try a completely different approach: use the known optimal rank
    # for specific matrix multiplication sizes.
    
    # For now, let me implement the 93-rank solution and see if I can improve.
    
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))
    
    def strassen_2x2_block(i_off, j_off, k_off, a_scale=1, b_scale=1, c_scale=1):
        """
        Adds 7 Strassen terms for a 2x2 * 2x2 block multiplication.
        """
        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)
        
        # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = c_scale
        w[(k_off+1) * n + (i_off+1)] = -c_scale
        add_term(u, v, w)
        
        # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = b_scale
        v[(j_off+1) * p + (k_off+1)] = -b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)
        
        # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[j_off * p + k_off] = -b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[k_off * n + (i_off+1)] = c_scale
        add_term(u, v, w)
        
        # M5 = (A11+A12)*B22 -> C12 += M5, C11 -= M5
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[i_off * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[k_off * n + i_off] = -c_scale
        add_term(u, v, w)
        
        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[i_off * m + j_off] = -a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[j_off * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)
        
        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = a_scale
        u[(i_off+1) * m + (j_off+1)] = -a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        add_term(u, v, w)
    
    def strassen_4x4_head():
        """
        Generates 49 terms for 4x4 * 4x4 using recursive Strassen.
        """
        outer_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        inner_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        def get_block_indices(block_idx, local_idx, dim_type):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            
            row = br * 2 + lr
            col = bc * 2 + lc
            
            if dim_type == 'A':
                return row * m + col
            elif dim_type == 'B':
                return row * p + col
            elif dim_type == 'C':
                return col * n + row
        
        for outer_a, outer_b, outer_c in outer_terms:
            for inner_x, inner_y, inner_z in inner_terms:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                for x_local, x_coeff in inner_x:
                    for a_block, a_coeff in outer_a:
                        idx = get_block_indices(a_block, x_local, 'A')
                        u[idx] += a_coeff * x_coeff
                        
                for y_local, y_coeff in inner_y:
                    for b_block, b_coeff in outer_b:
                        idx = get_block_indices(b_block, y_local, 'B')
                        v[idx] += b_coeff * y_coeff
                        
                for z_local, z_coeff in inner_z:
                    for c_block, c_coeff in outer_c:
                        idx = get_block_indices(c_block, z_local, 'C')
                        w[idx] += c_coeff * z_coeff
                        
                add_term(u, v, w)
    
    def strassen_4x4x2_tail():
        """
        Generates 28 terms for 4x4 * 4x2 (cols 4,5) using block Strassen.
        """
        for i_b in range(2):
            for j_b in range(2):
                i_off = i_b * 2
                j_off = j_b * 2
                k_off = 4
                strassen_2x2_block(i_off, j_off, k_off)
    
    def schoolbook_4x4x1_tail():
        """
        Generates 16 terms for 4x4 * 4x1 (col 6) using schoolbook.
        """
        for i in range(n):
            for j in range(m):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + 6] = 1
                w = [0] * (p * n)
                w[6 * n + i] = 1
                add_term(u, v, w)
    
    # Execute generation
    strassen_4x4_head()
    strassen_4x4x2_tail()
    schoolbook_4x4x1_tail()
    
    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
