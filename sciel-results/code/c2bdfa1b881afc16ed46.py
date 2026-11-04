import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Strassen 2x2x2 steps
    # Each step is (alpha, beta, gamma) where each is a 2x2 matrix of coefficients
    S7 = [
        (np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])),
        (np.array([[0, 1], [1, 1]]), np.array([[1, 0], [0, 0]]), np.array([[1, 0], [0, -1]])),
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, -1]]), np.array([[0, 1], [1, 1]])),
        (np.array([[0, 0], [1, 1]]), np.array([[1, 0], [-1, 0]]), np.array([[1, 1], [0, 0]])),
        (np.array([[1, 1], [0, 0]]), np.array([[0, 0], [1, 1]]), np.array([[-1, 1], [0, 0]])),
        (np.array([[-1, 0], [1, 0]]), np.array([[1, 1], [0, 0]]), np.array([[0, 0], [1, 1]])),
        (np.array([[0, 1], [-1, 1]]), np.array([[0, 1], [1, 1]]), np.array([[1, 0], [0, 0]]))
    ]
    
    rank = 77
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: 4x4 * 4x4 (first 4 columns of B, first 4 columns of C)
    # Two-level Strassen: 7 outer steps x 7 inner steps = 49 columns
    for outer in range(7):
        alpha_o, beta_o, gamma_o = S7[outer]
        for inner in range(7):
            alpha_i, beta_i, gamma_i = S7[inner]
            col = outer * 7 + inner
            
            # U: A is 4x4, index (row, col) -> row * m + col
            # alpha_o is 2x2 (block rows, block cols)
            # alpha_i is 2x2 (inner rows, inner cols)
            # r = (bo * 2 + bi) * m + (co * 2 + ci)
            for bo in range(2):
                for co in range(2):
                    vo = alpha_o[bo, co]
                    if vo != 0:
                        for bi in range(2):
                            for ci in range(2):
                                vi = alpha_i[bi, ci]
                                if vi != 0:
                                    r = (bo * 2 + bi) * m + (co * 2 + ci)
                                    U[r, col] += vo * vi
            
            # V: B is 4x4 (first 4 cols), index (row, col) -> row * p + col
            for bo in range(2):
                for co in range(2):
                    vo = beta_o[bo, co]
                    if vo != 0:
                        for bi in range(2):
                            for ci in range(2):
                                vi = beta_i[bi, ci]
                                if vi != 0:
                                    r = (bo * 2 + bi) * p + (co * 2 + ci)
                                    V[r, col] += vo * vi
            
            # W: C is 4x4 (first 4 cols), index (col, row) -> col * n + row
            for bo in range(2):
                for co in range(2):
                    vo = gamma_o[bo, co]
                    if vo != 0:
                        for bi in range(2):
                            for ci in range(2):
                                vi = gamma_i[bi, ci]
                                if vi != 0:
                                    r = (co * 2 + ci) * n + (bo * 2 + bi)
                                    W[r, col] += vo * vi
    
    # Part 2: 4x4 * 4x2 (last 2 columns of B, last 2 columns of C)
    # A is 4x4, B is 4x2, C is 4x2
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # 4 block multiplications: A00*B0, A01*B1, A10*B0, A11*B1
    # Each is 2x2 * 2x2 -> 2x2, rank 7
    # Total: 4 * 7 = 28 columns
    
    col_offset = 49
    mults = [
        (0, 0, 0, 0),  # A00 * B0 -> C0
        (0, 1, 1, 0),  # A01 * B1 -> C0
        (1, 0, 0, 1),  # A10 * B0 -> C1
        (1, 1, 1, 1),  # A11 * B1 -> C1
    ]
    
    for m_idx, (ar, ac, br, cr) in enumerate(mults):
        for inner in range(7):
            alpha_i, beta_i, gamma_i = S7[inner]
            col = col_offset + m_idx * 7 + inner
            
            # U: A block at (ar, ac), 2x2
            for bi in range(2):
                for ci in range(2):
                    vi = alpha_i[bi, ci]
                    if vi != 0:
                        r = (ar * 2 + bi) * m + (ac * 2 + ci)
                        U[r, col] += vi
            
            # V: B block at row br, 2x2, cols 4-5
            for bi in range(2):
                for ci in range(2):
                    vi = beta_i[bi, ci]
                    if vi != 0:
                        r = (br * 2 + bi) * p + (4 + ci)
                        V[r, col] += vi
            
            # W: C block at row cr, 2x2, cols 4-5
            for bi in range(2):
                for ci in range(2):
                    vi = gamma_i[bi, ci]
                    if vi != 0:
                        r = (4 + ci) * n + (cr * 2 + bi)
                        W[r, col] += vi
                        
    # Verification and Rank Reduction Check (Structural Change)
    # Check if the 77 rank-1 tensors are linearly independent.
    # If not, we might be able to reduce rank.
    # We form the matrix M where each column is the flattened tensor u_i v_i w_i.
    # If rank(M) < 77, we have dependencies.
    
    if time.time() < deadline:
        try:
            # Flatten U, V, W into a matrix of shape (77, n*m*p)
            # This is expensive but n*m*p = 96, 77*96 is small.
            # Actually, we can just check if the columns of U, V, W are independent?
            # No, the tensor is the sum of outer products.
            # We check if the set {u_i \otimes v_i \otimes w_i} is linearly independent.
            
            # Construct the tensor matrix
            T_flat = np.zeros((rank, n * m * p))
            for k in range(rank):
                T_flat[k, :] = np.outer(np.outer(U[:, k], V[:, k]).flatten(), W[:, k])
                # Wait, outer(outer(U, V), W) is not quite right for flattening.
                # u (nm x 1), v (mp x 1), w (pn x 1)
                # Tensor T_ijk = u_i v_j w_k
                # Flatten: index = (i * mp + j) * pn + k? No.
                # Standard flattening: i * (m*p) + j? No.
                # Let's just use np.einsum or broadcasting.
                T_flat[k, :] = np.multiply.outer(np.multiply.outer(U[:, k], V[:, k]), W[:, k]).flatten()
            
            # Compute rank of T_flat
            r = np.linalg.matrix_rank(T_flat)
            if r < rank:
                # Dependencies exist. We could try to reduce, but for now, 
                # we just note it. To actually reduce, we'd need to find the null space.
                # Given the strict time limit and exactness requirement, we stick to 77.
                pass
        except Exception:
            pass
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
