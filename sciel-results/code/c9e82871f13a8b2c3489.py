"""Matrix multiplication 2x5x6 using Strassen block decomposition."""
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total_rank = 54

    # Preallocate
    U = [[0.0] * total_rank for _ in range(n * m)]  # 10 x 54
    V = [[0.0] * total_rank for _ in range(m * p)]  # 30 x 54
    W = [[0.0] * total_rank for _ in range(p * n)]  # 12 x 54

    col = 0

    def add_term(i0, j0, k0, a, b, c):
        nonlocal col
        U[i0][col] += a
        V[j0][col] += b
        W[k0][col] += c
        col += 1

    # Strassen 2x2x2 decomposition
    # A is 2x2, B is 2x2, C is 2x2
    # U index: i*2 + j (i in {0,1}, j in {0,1})
    # V index: j*2 + k (j in {0,1}, k in {0,1})
    # W index: k*2 + i (k in {0,1}, i in {0,1})
    
    strassen_terms = [
        # P1: (A[0][0]+A[1][1]) * (B[0][0]+B[1][1])
        # A: (0,0,1),(1,1,1); B: (0,0,1),(1,1,1); C: (0,0,1),(1,1,-1)
        {'A': [(0,0,1),(1,1,1)], 'B': [(0,0,1),(1,1,1)], 'C': [(0,0,1),(1,1,-1)]},
        # P2: (A[0][0]+A[1][1]) * B[1][1]
        # A: (0,0,1),(1,1,1); B: (1,1,1); C: (0,0,-1)
        {'A': [(0,0,1),(1,1,1)], 'B': [(1,1,1)], 'C': [(0,0,-1)]},
        # P3: A[0][0] * (B[0][0]-B[0][1])
        # A: (0,0,1); B: (0,0,1),(0,1,-1); C: (0,0,1),(1,0,1)
        {'A': [(0,0,1)], 'B': [(0,0,1),(0,1,-1)], 'C': [(0,0,1),(1,0,1)]},
        # P4: A[1][1] * (B[1][1]-B[1][0])
        # A: (1,1,1); B: (1,1,1),(1,0,-1); C: (0,1,1),(1,1,1)
        {'A': [(1,1,1)], 'B': [(1,1,1),(1,0,-1)], 'C': [(0,1,1),(1,1,1)]},
        # P5: (A[0][0]+A[0][1]) * B[1][1]
        # A: (0,0,1),(0,1,1); B: (1,1,1); C: (0,1,1),(1,1,-1)
        {'A': [(0,0,1),(0,1,1)], 'B': [(1,1,1)], 'C': [(0,1,1),(1,1,-1)]},
        # P6: (A[1][0]-A[0][0]) * (B[0][0]+B[0][1])
        # A: (1,0,1),(0,0,-1); B: (0,0,1),(0,1,1); C: (0,0,1),(1,1,1)
        {'A': [(1,0,1),(0,0,-1)], 'B': [(0,0,1),(0,1,1)], 'C': [(0,0,1),(1,1,1)]},
        # P7: (A[0][1]-A[1][1]) * (B[1][0]+B[1][1])
        # A: (0,1,1),(1,1,-1); B: (1,0,1),(1,1,1); C: (1,0,1)
        {'A': [(0,1,1),(1,1,-1)], 'B': [(1,0,1),(1,1,1)], 'C': [(1,0,1)]},
    ]

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        for term in strassen_terms:
            # For each term, we need to add rank-1 contributions
            # A factors: list of (i, j, coeff) -> U index = (i0+i)*2 + (j0+j)
            # B factors: list of (j, k, coeff) -> V index = (j0+j)*2 + (k0+k)
            # C factors: list of (i, k, coeff) -> W index = (k0+k)*2 + (i0+i)
            
            # This is a product of two linear forms, so we need to expand it
            # Actually, in the tensor decomposition, each Strassen "multiplication" 
            # corresponds to a single rank-1 term in the decomposition? No.
            # The rank of the 2x2x2 tensor is 7, and the 7 terms in the decomposition
            # are rank-1 tensors. Each term is of the form:
            #   sum_a sum_b sum_c A_a B_b C_c * (e_a e_b e_c)
            # But that's not right either. Let me think again.
            
            # A rank-1 tensor for matrix multiplication is:
            #   T = (sum_{i,j} u_{ij} e_i^A e_j^B) * (sum_{j,k} v_{jk} e_j^B e_k^C) * (sum_{i,k} w_{ik} e_i^C e_k^A)
            # Wait, that's not how it works. A rank-1 tensor in the multiplication tensor space is:
            #   T_{ijk} = a_i * b_j * c_k
            # where a_i is a vector in R^m (for A), b_j in R^n (for B), c_k in R^p (for C)
            # Actually, for matrix multiplication C = AB, the tensor is:
            #   T = sum_{i,j,k} e_i^A e_j^B e_k^C * delta_{jk}
            # A rank-1 term is:
            #   u * v * w where u is in R^{m x n}, v in R^{n x p}, w in R^{p x m}
            # and the contribution to T is u_{ij} v_{jk} w_{ki} for each (i,j,k)
            
            # So for a Strassen term, we have:
            #   A_part = sum (coeff * e_{i,j})
            #   B_part = sum (coeff * e_{j,k})
            #   C_part = sum (coeff * e_{i,k})
            # The rank-1 tensor is A_part * B_part * C_part
            # which contributes to T_{i,j,k} the product of the coefficients.
            
            # So for each Strassen term, we add ONE rank-1 tensor to the decomposition.
            # The U, V, W entries for this rank-1 tensor are the coefficients.
            
            for (i, j, a_coeff) in term['A']:
                U[(i0 + i) * m + (j0 + j)][col] += a_coeff
            for (j, k, b_coeff) in term['B']:
                V[(j0 + j) * p + (k0 + k)][col] += b_coeff
            for (i, k, c_coeff) in term['C']:
                W[(k0 + k) * n + (i0 + i)][col] += c_coeff
            col += 1

    # Schoolbook 2x1x2 decomposition
    # A is 2x1, B is 1x2, C is 2x2
    # U index: i*1 + 0 = i
    # V index: 0*2 + k = k
    # W index: k*2 + i
    
    def add_schoolbook_2x1x2(i0, j0, k0):
        nonlocal col
        # Term 1: A[0][0] * B[0][0] -> C[0][0]
        U[(i0 + 0) * m + (j0 + 0)][col] = 1
        V[(j0 + 0) * p + (k0 + 0)][col] = 1
        W[(k0 + 0) * n + (i0 + 0)][col] = 1
        col += 1
        # Term 2: A[0][0] * B[0][1] -> C[0][1]
        U[(i0 + 0) * m + (j0 + 0)][col] = 1
        V[(j0 + 0) * p + (k0 + 1)][col] = 1
        W[(k0 + 1) * n + (i0 + 0)][col] = 1
        col += 1
        # Term 3: A[1][0] * B[0][0] -> C[1][0]
        U[(i0 + 1) * m + (j0 + 0)][col] = 1
        V[(j0 + 0) * p + (k0 + 0)][col] = 1
        W[(k0 + 0) * n + (i0 + 1)][col] = 1
        col += 1
        # Term 4: A[1][0] * B[0][1] -> C[1][1]
        U[(i0 + 1) * m + (j0 + 0)][col] = 1
        V[(j0 + 0) * p + (k0 + 1)][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = 1
        col += 1

    # Partition the 5x6 middle/output dimensions
    # m=5, p=6
    # Partition m into [2, 2, 1] and p into [2, 2, 2]
    # This gives 3x3 = 9 blocks
    # Block (bi, bj) where bi in {0,1,2}, bj in {0,1,2}
    # bi=0: m rows 0-1, bi=1: m rows 2-3, bi=2: m row 4
    # bj=0: p cols 0-1, bj=1: p cols 2-3, bj=2: p cols 4-5
    
    m_offsets = [0, 2, 4]  # start indices for m blocks
    m_sizes = [2, 2, 1]    # sizes of m blocks
    p_offsets = [0, 2, 4]  # start indices for p blocks
    p_sizes = [2, 2, 2]    # sizes of p blocks
    
    for bi in range(3):
        for bj in range(3):
            m_start = m_offsets[bi]
            m_size = m_sizes[bi]
            p_start = p_offsets[bj]
            p_size = p_sizes[bj]
            
            if m_size == 2 and p_size == 2:
                # 2x2x2 block -> use Strassen (7 terms)
                add_strassen_block(0, m_start, p_start)
            elif m_size == 2 and p_size == 2:
                pass
            elif m_size == 1 and p_size == 2:
                # 1x2x2 block -> schoolbook (2 terms)
                # A is 1x2, B is 2x2, C is 1x2
                # Actually, let me handle this more carefully
                # For m_size=1, p_size=2:
                # A is 1x2 (rows: 1, cols: 2)
                # B is 2x2 (rows: 2, cols: 2)
                # C is 1x2 (rows: 1, cols: 2)
                # Schoolbook: 1*2*2 = 4 terms
                for i in range(1):
                    for k in range(2):
                        for j in range(2):
                            U[(i0 + i) * m + (m_start + j)][col] = 1
                            V[(m_start + j) * p + (p_start + k)][col] = 1
                            W[(p_start + k) * n + (i0 + i)][col] = 1
                            col += 1
            elif m_size == 2 and p_size == 2:
                pass
            elif m_size == 2 and p_size == 2:
                pass
            else:
                # Fallback: schoolbook
                for i in range(2):
                    for k in range(p_size):
                        for j in range(m_size):
                            U[(i0 + i) * m + (m_start + j)][col] = 1
                            V[(m_start + j) * p + (p_start + k)][col] = 1
                            W[(p_start + k) * n + (i0 + i)][col] = 1
                            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
