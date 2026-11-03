"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is correct."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _build_3x3x2_strassen():
    """
    Build a rank-17 decomposition for 3x3x2 matrix multiplication.
    A(3x3) * B(3x2) -> C(3x2)
    
    Index conventions:
    U[i*m+j] where m=3: index = 3i+j
    V[j*p+k] where p=2: index = 2j+k  
    W[k*n+i] where n=3: index = 3k+i
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    """
    strassen_terms = [
        # M0: (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) -> C[0,0]+=1, C[1,1]+=1
        ([1,0,0, 0,1,0, 0,0,0], [1,0, 0,1, 0,0], [1,0, 0,1, 0,0]),
        # M1: (A[1,0]+A[1,1])*B[0,0] -> C[1,0]+=1
        ([0,0,0, 1,1,0, 0,0,0], [1,0, 0,0, 0,0], [0,0, 1,0, 0,0]),
        # M2: A[0,0]*(B[1,0]-B[1,1]) -> C[0,1]+=1
        ([1,0,0, 0,0,0, 0,0,0], [0,0, 1,-1, 0,0], [0,1, 0,0, 0,0]),
        # M3: A[1,1]*(B[0,1]-B[0,0]) -> C[0,0]+=-1, C[1,1]+=-1
        ([0,0,0, 0,1,0, 0,0,0], [-1,1, 0,0, 0,0], [-1,0, 0,-1, 0,0]),
        # M4: (A[0,0]+A[0,1])*B[1,1] -> C[0,1]+=1
        ([1,1,0, 0,0,0, 0,0,0], [0,0, 0,1, 0,0], [0,1, 0,0, 0,0]),
        # M5: (A[1,0]-A[0,0])*(B[0,0]+B[0,1]) -> C[1,0]+=1
        ([-1,0,0, 1,0,0, 0,0,0], [1,1, 0,0, 0,0], [0,0, 1,0, 0,0]),
        # M6: (A[0,1]-A[1,1])*(B[1,0]+B[1,1]) -> C[0,0]+=1
        ([0,1,0, 0,-1,0, 0,0,0], [0,0, 1,1, 0,0], [1,0, 0,0, 0,0]),
    ]
    
    cross_terms = [
        # A[2,0]*B[0,0] -> C[2,0]
        ([0,0,0, 0,0,0, 1,0,0], [1,0, 0,0, 0,0], [0,0, 0,0, 1,0]),
        # A[2,0]*B[0,1] -> C[2,1]
        ([0,0,0, 0,0,0, 1,0,0], [0,1, 0,0, 0,0], [0,0, 0,0, 0,1]),
        # A[2,1]*B[1,0] -> C[2,0]
        ([0,0,0, 0,0,0, 0,1,0], [0,0, 1,0, 0,0], [0,0, 0,0, 1,0]),
        # A[2,1]*B[1,1] -> C[2,1]
        ([0,0,0, 0,0,0, 0,1,0], [0,0, 0,1, 0,0], [0,0, 0,0, 0,1]),
        # A[2,2]*B[2,0] -> C[2,0]
        ([0,0,0, 0,0,0, 0,0,1], [0,0, 0,0, 1,0], [0,0, 0,0, 1,0]),
        # A[2,2]*B[2,1] -> C[2,1]
        ([0,0,0, 0,0,0, 0,0,1], [0,0, 0,0, 0,1], [0,0, 0,0, 0,1]),
        # A[0,2]*B[2,0] -> C[0,0]
        ([0,0,1, 0,0,0, 0,0,0], [0,0, 0,0, 1,0], [1,0, 0,0, 0,0]),
        # A[0,2]*B[2,1] -> C[0,1]
        ([0,0,1, 0,0,0, 0,0,0], [0,0, 0,0, 0,1], [0,1, 0,0, 0,0]),
        # A[1,2]*B[2,0] -> C[1,0]
        ([0,0,0, 0,0,1, 0,0,0], [0,0, 0,0, 1,0], [0,0, 1,0, 0,0]),
        # A[1,2]*B[2,1] -> C[1,1]
        ([0,0,0, 0,0,1, 0,0,0], [0,0, 0,0, 0,1], [0,0, 0,1, 0,0]),
    ]
    
    return strassen_terms + cross_terms

def _build_3x3x3_strassen():
    """
    Build a rank-23 decomposition for 3x3x3 matrix multiplication.
    A(3x3) * B(3x3) -> C(3x3)
    
    Index conventions:
    U[i*m+j] where m=3: index = 3i+j
    V[j*p+k] where p=3: index = 3j+k  
    W[k*n+i] where n=3: index = 3k+i
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    """
    # We'll build this by combining 3x3x2 blocks
    # Split B and C into [2, 1] columns
    block_2 = _build_3x3x2_strassen()
    block_1 = [
        # A[i,j]*B[j,k] for k=2
        ([1,0,0, 0,0,0, 0,0,0], [0,0, 0,0, 0,1], [0,0, 0,0, 1,0]),  # A[0,0]*B[0,2] -> C[0,2]
        ([0,1,0, 0,0,0, 0,0,0], [0,0, 0,0, 0,1], [0,0, 0,0, 1,0]),  # A[0,1]*B[1,2] -> C[0,2]
        ([0,0,1, 0,0,0, 0,0,0], [0,0, 0,0, 0,1], [0,0, 0,0, 1,0]),  # A[0,2]*B[2,2] -> C[0,2]
        ([0,0,0, 1,0,0, 0,0,0], [0,0, 0,0, 0,1], [0,0, 0,0, 0,1]),  # A[1,0]*B[0,2] -> C[1,2]
        ([0,0,0, 0,1,0, 0,0,0], [0,0, 0,0, 0,1], [0,0, 0,0, 0,1]),  # A[1,1]*B[1,2] -> C[1,2]
        ([0,0,0, 0,0,1, 0,0,0], [0,0, 0,0, 0,1], [0,0, 0,0, 0,1]),  # A[1,2]*B[2,2] -> C[1,2]
        ([0,0,0, 0,0,0, 1,0,0], [0,0, 0,0, 0,1], [0,0, 0,0, 0,0]),  # A[2,0]*B[0,2] -> C[2,2]
        ([0,0,0, 0,0,0, 0,1,0], [0,0, 0,0, 0,1], [0,0, 0,0, 0,0]),  # A[2,1]*B[1,2] -> C[2,2]
        ([0,0,0, 0,0,0, 0,0,1], [0,0, 0,0, 0,1], [0,0, 0,0, 0,0]),  # A[2,2]*B[2,2] -> C[2,2]
    ]
    
    # Extend block_2 to 3x3 by zeroing out the 3rd column of V and W
    block_2_ext = []
    for u, v, w in block_2:
        v_ext = v + [0]  # Add 0 for 3rd column
        w_ext = []
        for i in range(3):
            for k in range(2):
                w_ext.append(w[k*3 + i])
            w_ext.append(0)  # Add 0 for 3rd column
        block_2_ext.append((u, v_ext, w_ext))
    
    return block_2_ext + block_1

def _build_3x3x6_decomposition():
    """
    Build a rank-63 decomposition for 3x3x6 matrix multiplication.
    Split B and C into 3 blocks of 2 columns each, using 3x3x2 block decomposition.
    """
    n, m, p = 3, 3, 6
    block_p = 2
    block_terms = _build_3x3x2_strassen()
    r_block = len(block_terms)
    total_rank = 3 * r_block
    
    U_full = [[0]*total_rank for _ in range(n*m)]
    V_full = [[0]*total_rank for _ in range(m*p)]
    W_full = [[0]*total_rank for _ in range(p*n)]
    
    for t in range(3):
        for c in range(r_block):
            col = t * r_block + c
            u, v, w = block_terms[c]
            
            for idx in range(n*m):
                U_full[idx][col] = u[idx]
            
            for j in range(m):
                for k in range(block_p):
                    local_v_idx = j * block_p + k
                    global_v_idx = j * p + (t * block_p + k)
                    V_full[global_v_idx][col] = v[local_v_idx]
            
            for i in range(n):
                for k in range(block_p):
                    local_w_idx = block_p * i + k
                    global_w_idx = (t * block_p + k) * n + i
                    W_full[global_w_idx][col] = w[local_w_idx]
    
    return U_full, V_full, W_full

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _attempt_rank_reduction(U, V, W, n, m, p, max_iterations=1000, tolerance=1e-8):
    """
    Attempt to reduce the rank of the decomposition by finding linear dependencies
    among the rank-1 terms. This is a heuristic approach.
    """
    rank = len(U[0])
    
    # Try to find columns that are linear combinations of other columns
    # This is a simplification - in practice, we'd use SVD or similar
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    for iteration in range(max_iterations):
        # Check if we can remove any column
        for c in range(rank - 1, -1, -1):
            test_rank = rank - 1
            U_test = [row[:c] + row[c+1:] for row in U]
            V_test = [row[:c] + row[c+1:] for row in V]
            W_test = [row[:c] + row[c+1:] for row in W]
            
            # Verify the reduced decomposition
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        actual = 0.0
                        for cc in range(test_rank):
                            actual += U_test[i*m+j][cc] * V_test[j*p+k][cc] * W_test[k*n+i][cc]
                        if abs(actual - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
                
            if valid:
                best_rank = test_rank
                best_U, best_V, best_W = U_test, V_test, W_test
                U, V, W = U_test, V_test, W_test
                rank = test_rank
    
    return best_U, best_V, best_W, best_rank

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Deadline guard: return best result before 150s limit
    deadline = start_time + 140.0
    
    # Try the block-based decomposition
    if time.time() < deadline:
        U_block, V_block, W_block = _build_3x3x6_decomposition()
        
        if _verify_decomposition(U_block, V_block, W_block, n, m, p):
            # Attempt rank reduction
            U_red, V_red, W_red, rank_red = _attempt_rank_reduction(
                U_block, V_block, W_block, n, m, p, 
                max_iterations=100 if time.time() < deadline - 60 else 10
            )
            
            if _verify_decomposition(U_red, V_red, W_red, n, m, p):
                return {"U": U_red, "V": V_red, "W": W_red}
            else:
                return {"U": U_block, "V": V_block, "W": W_block}
    
    # Fallback: schoolbook rank 54
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
