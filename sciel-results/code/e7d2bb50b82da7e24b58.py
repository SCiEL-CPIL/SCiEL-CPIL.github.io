"""Program for matrix multiplication 3x3x4 tensor decomposition."""
import time

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline to be safe
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]  # 9 x 34
    V = [[0] * rank for _ in range(m * p)]  # 12 x 34
    W = [[0] * rank for _ in range(p * n)]  # 12 x 34
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx, val in u_indices:
            U[idx][col] = val
        for idx, val in v_indices:
            V[idx][col] = val
        for idx, val in w_indices:
            W[idx][col] = val
        col += 1
        
        # Check deadline
        if time.time() > deadline:
            return False
        return True

    # Strassen Left (A11 * B1L) - 7 terms
    # A11 indices: 0,1,3,4
    # B1L indices: 0,1,4,5
    # C1L indices: 0,3,1,4
    
    # P1
    if not add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (4, 1)]): return {"U": U, "V": V, "W": W}
    # P2
    if not add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (4, -1)]): return {"U": U, "V": V, "W": W}
    # P3
    if not add_term([(0, 1)], [(1, 1), (5, -1)], [(3, 1), (4, 1)]): return {"U": U, "V": V, "W": W}
    # P4
    if not add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)]): return {"U": U, "V": V, "W": W}
    # P5
    if not add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (3, 1)]): return {"U": U, "V": V, "W": W}
    # P6
    if not add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(4, 1)]): return {"U": U, "V": V, "W": W}
    # P7
    if not add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)]): return {"U": U, "V": V, "W": W}
    
    # A12 * B2L - 4 terms
    if not add_term([(2, 1)], [(8, 1)], [(0, 1)]): return {"U": U, "V": V, "W": W}
    if not add_term([(2, 1)], [(9, 1)], [(3, 1)]): return {"U": U, "V": V, "W": W}
    if not add_term([(5, 1)], [(8, 1)], [(1, 1)]): return {"U": U, "V": V, "W": W}
    if not add_term([(5, 1)], [(9, 1)], [(4, 1)]): return {"U": U, "V": V, "W": W}
    
    # A21 * B1L - 4 terms
    if not add_term([(6, 1)], [(0, 1)], [(2, 1)]): return {"U": U, "V": V, "W": W}
    if not add_term([(6, 1)], [(1, 1)], [(5, 1)]): return {"U": U, "V": V, "W": W}
    if not add_term([(7, 1)], [(4, 1)], [(2, 1)]): return {"U": U, "V": V, "W": W}
    if not add_term([(7, 1)], [(5, 1)], [(5, 1)]): return {"U": U, "V": V, "W": W}
    
    # A22 * B2L - 2 terms
    if not add_term([(8, 1)], [(8, 1)], [(2, 1)]): return {"U": U, "V": V, "W": W}
    if not add_term([(8, 1)], [(9, 1)], [(5, 1)]): return {"U": U, "V": V, "W": W}
    
    # Strassen Right (A11 * B1R) - 7 terms
    if not add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(6, 1), (10, 1)]): return {"U": U, "V": V, "W": W}
    if not add_term([(3, 1), (4, 1)], [(2, 1)], [(7, 1), (10, -1)]): return {"U": U, "V": V, "W": W}
    if not add_term([(0, 1)], [(3, 1), (7, -1)], [(9, 1), (10, 1)]): return {"U": U, "V": V, "W": W}
    if not add_term([(4, 1)], [(6, 1), (2, -1)], [(6, 1), (7, 1)]): return {"U": U, "V": V, "W": W}
    if not add_term([(0, 1), (1, 1)], [(7, 1)], [(6, -1), (9, 1)]): return {"U": U, "V": V, "W": W}
    if not add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(10, 1)]): return {"U": U, "V": V, "W": W}
    if not add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(6, 1)]): return {"U": U, "V": V, "W": W}
    
    # A12 * B2R - 4 terms
    if not add_term([(2, 1)], [(10, 1)], [(6, 1)]): return {"U": U, "V": V, "W": W}
    if not add_term([(2, 1)], [(11, 1)], [(9, 1)]): return {"U": U, "V": V, "W": W}
    if not add_term([(5, 1)], [(10, 1)], [(7, 1)]): return {"U": U, "V": V, "W": W}
    if not add_term([(5, 1)], [(11, 1)], [(10, 1)]): return {"U": U, "V": V, "W": W}
    
    # A21 * B1R - 4 terms
    if not add_term([(6, 1)], [(2, 1)], [(8, 1)]): return {"U": U, "V": V, "W": W}
    if not add_term([(6, 1)], [(3, 1)], [(11, 1)]): return {"U": U, "V": V, "W": W}
    if not add_term([(7, 1)], [(6, 1)], [(8, 1)]): return {"U": U, "V": V, "W": W}
    if not add_term([(7, 1)], [(7, 1)], [(11, 1)]): return {"U": U, "V": V, "W": W}
    
    # A22 * B2R - 2 terms
    if not add_term([(8, 1)], [(10, 1)], [(8, 1)]): return {"U": U, "V": V, "W": W}
    if not add_term([(8, 1)], [(11, 1)], [(11, 1)]): return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
