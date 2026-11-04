"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try to find a lower rank decomposition using rank minimization algorithms
    # For M_{3,3,8}, we can try to exploit structure
    
    # First, let's establish the schoolbook rank as baseline
    schoolbook_rank = n * m * p  # 72
    
    # For matrix multiplication M_{n,m,p}, there are known decompositions:
    # - Schoolbook: n*m*p
    # - For square matrices, Strassen-like reductions exist
    # - For non-square, we need to check known results
    
    # Key insight: M_{3,3,8} can be related to M_{3,3,3} via block structure
    # A(3x3) * B(3x8) = C(3x8)
    # We can write B as [B1 | B2] where B1, B2 are 3x4
    # Then C = [A*B1 | A*B2]
    # This doesn't directly help with rank reduction
    
    # Let's try a different approach: use the fact that we can pad M_{3,3,3} to M_{3,3,8}
    # But that would increase rank, not decrease
    
    # Actually, for M_{n,m,p} where one dimension is larger, we can use:
    # rank(M_{n,m,p}) <= m * rank(M_{n,1,p})
    # And rank(M_{n,1,p}) = n (schoolbook for vector-matrix mult)
    # So rank <= m * n = 9 * 3 = 27? No, that's not right.
    
    # Let me think more carefully. The tensor T_{i,j,k} = delta_{i,j,k} for matrix mult.
    # For M_{3,3,8}: T_{i,j,k} where i in [0,3), j in [0,3), k in [0,8)
    # T_{i,j,k} = 1 if the product term A_{i,j} * B_{j,k} appears in C_{i,k}
    # So T_{i,j,k} = 1 for all valid i,j,k (it's the standard matrix mult tensor)
    
    # Known results:
    # M_{3,3,3} has rank 23 (Bini et al.) or can be done with rank 23
    # Actually, the best known for M_{3,3,3} is 23 over some fields, but with integer constraints it's harder
    
    # For M_{3,3,8}, we can decompose as:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # For k in [0,8), this is 8 independent 3x3 matrix multiplications in a sense
    
    # Actually no - it's one matrix multiplication of 3x3 by 3x8
    
    # Let me try to use a rank-24 decomposition for M_{3,3,3} and extend to M_{3,3,8}
    # But M_{3,3,8} is different from 8 copies of M_{3,3,3}
    
    # Better approach: use the structure that B is 3x8, so we can group columns
    # Let B = [B_0 | B_1 | B_2 | B_3] where each B_t is 3x2
    # Then C = [C_0 | C_1 | C_2 | C_3] where C_t = A * B_t
    # Each C_t is 3x2, computed from A (3x3) and B_t (3x2)
    # This is 4 matrix multiplications of 3x3 * 3x2
    
    # rank(M_{3,3,2}) <= 6 (schoolbook) or possibly lower
    # If we can do M_{3,3,2} with rank 5, then M_{3,3,8} could be 4*5 = 20
    
    # Let's try to find a rank-5 decomposition for M_{3,3,2}
    # T_{i,j,k} for i in [0,3), j in [0,3), k in [0,2)
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    # For 3x3 * 3x2, schoolbook is 18. Can we do better?
    # Using Strassen-like ideas for non-square matrices is tricky
    
    # Alternative: use the fact that we only need integer/half-integer entries
    # Let's try to construct a valid decomposition with rank < 72
    
    # Actually, let me try a direct approach using bilinear complexity bounds
    # For M_{m,n,p}, we have various bounds. For M_{3,3,8}:
    # Lower bound: max(3*3, 3*8, 3*8) = 24 (trivial lower bound is max(mn, mp, np) but that's not tight)
    # Actually the trivial lower bound is mn if we fix one index, but the tensor rank is at least max(mn, np, mp) / something
    
    # Let me just try to use a known decomposition for M_{3,3,8}
    # One approach: use the "block" method
    # Write A as 1x3 blocks of rows, B as 3x1 blocks of columns
    # This gives the schoolbook 72
    
    # Let me try to use a rank-24 decomposition if I can find one that works
    # For now, let me return a valid decomposition and see if I can improve
    
    # Strategy: try to find a decomposition using numpy optimization within time budget
    
    # First, let's create the tensor T for M_{3,3,8}
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i,j,k] corresponds to U[i,j], V[j,k], W[k,i]
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try to find a low-rank decomposition
    # This is a hard problem, so let's use a heuristic approach
    
    # Check if we have time to do optimization
    elapsed = time.time() - start_time
    if elapsed > 120 * 0.5:
        # Not enough time, return schoolbook
        rank = schoolbook_rank
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a lower rank decomposition using random initialization and optimization
    # This is a research-level problem, so let's try a few approaches
    
    # Approach 1: Try rank 24 (if we can find a valid decomposition)
    # For M_{3,3,3}, there's a rank-23 decomposition, but for M_{3,3,8} it's different
    
    # Let me try to use the fact that M_{3,3,8} = M_{3,3,2} repeated 4 times with shifted indices
    # Actually no, it's a single matrix multiplication
    
    # Let me try a different tactic: use the "laser method" or "cut-and-join"
    # For now, let's try to find any valid decomposition with rank < 72
    
    # Try rank 48 (half of schoolbook) using a simple reduction
    # If we can pair up some of the schoolbook terms
    
    # Actually, let me just try to use a rank-24 decomposition if possible
    # For matrix multiplication, the "Strassen" algorithm gives rank 7 for 2x2x2
    # For 3x3x3, the best known is 23 (over some fields)
    
    # Given the complexity, let me return a valid decomposition
    # I'll try to find a decomposition with rank around 48 using a simple method
    
    # Method: group the k-index in pairs and try to combine
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # For k=0,1: C_{i,0} = sum_j A_{i,j} B_{j,0}, C_{i,1} = sum_j A_{i,j} B_{j,1}
    # We can write:
    # C_{i,0} + C_{i,1} = sum_j A_{i,j} (B_{j,0} + B_{j,1})
    # C_{i,0} - C_{i,1} = sum_j A_{i,j} (B_{j,0} - B_{j,1})
    # So we can compute S_{i,j} = A_{i,j} (B_{j,0} + B_{j,1}) and D_{i,j} = A_{i,j} (B_{j,0} - B_{j,1})
    # Then C_{i,0} = (S_{i,j} + D_{i,j}) / 2, C_{i,1} = (S_{i,j} - D_{i,j}) / 2
    # But this doesn't reduce the number of multiplications - we still need 9 multiplications for each pair
    
    # This approach doesn't work directly for reducing rank
    
    # Let me try to use a known result: for M_{m,n,p}, if one dimension is even,
    # we can sometimes use a reduction
    
    # For now, let me return the schoolbook decomposition to ensure validity
    # and hope that future iterations find a better one
    
    rank = schoolbook_rank
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
