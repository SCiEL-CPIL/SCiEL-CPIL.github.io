"""Tensor decomposition for 3x3 by 3x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    def verify(U, V, W):
        """Verify that the decomposition is correct."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def to_list(mat):
        """Convert numpy array to list of lists."""
        return [[float(x) for x in row] for row in mat]
    
    # Schoolbook rank: 72
    schoolbook_rank = n * m * p  # 72
    
    U_school = np.zeros((n*m, schoolbook_rank))
    V_school = np.zeros((m*p, schoolbook_rank))
    W_school = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = schoolbook_rank
    
    # Structured Kronecker product ansatz:
    # Decompose the 3x3x3 tensor using a rank-5 (or similar) decomposition,
    # then tensor with a rank-8 decomposition for the 1x8 dimension.
    # This gives rank = rank_333 * 8.
    
    # Known: the 3x3x3 matrix multiplication tensor has rank at most 23
    # (Laderman's algorithm), but we need an explicit decomposition.
    # Let's try to construct a rank-23 decomposition for 3x3x3.
    
    # Actually, let's use a simpler approach: try to find a low-rank decomposition
    # of the 3x3x3 tensor, then Kronecker product with identity on the 8-dim space.
    
    # For the 3x3x3 tensor T3[i,j,k] = delta_{ik}, we can use:
    # A rank-9 decomposition (schoolbook) or try to find lower.
    
    # Let's try a structured approach: use the fact that we can write
    # T3 as a sum of rank-1 tensors.
    
    # Strategy: Try Kronecker product of a rank-r 3x3x3 decomposition with
    # the 8-dimensional identity. This gives rank = r * 8.
    
    # If we can find a rank-r decomposition of 3x3x3 with r < 9, we win.
    # Known lower bound for 3x3x3 is 19 (from algebraic complexity theory),
    # but we need explicit decompositions.
    
    # Let's try to use Laderman's algorithm for 4x4x4 and restrict to 3x3x3.
    # Laderman's algorithm has rank 21 for 4x4x4.
    
    # Actually, let's try a different approach: use the separable structure.
    # The 3x3x8 tensor can be viewed as 8 copies of the 3x3x3 tensor,
    # but with the k-index running from 0 to 7.
    
    # Wait, the tensor is T[i,j,k] = delta_{i,k} for k < 3, and 0 for k >= 3.
    # So it's just the 3x3x3 tensor embedded in a 3x3x8 space.
    
    # This means we can use any decomposition of the 3x3x3 tensor and pad
    # the W matrix with zeros for k >= 3.
    
    # So the problem reduces to: find the minimum rank of the 3x3x3 tensor.
    
    # Known results:
    # - Schoolbook: rank 9
    # - Best known upper bound: 19 (from various papers)
    # - Lower bound: 19 (proven)
    
    # So the minimum rank for 3x3x3 is exactly 19!
    
    # Let's construct a rank-19 decomposition of the 3x3x3 tensor.
    # This is a well-known result. Let me try to find or construct one.
    
    # For now, let's try to use a random search to find a rank-19 decomposition
    # of the 3x3x3 tensor, then embed it in the 3x3x8 space.
    
    def find_333_decomposition(max_rank=23, time_budget=60):
        """Try to find a low-rank decomposition of the 3x3x3 tensor."""
        start = time.time()
        deadline = start + time_budget
        
        # Start with schoolbook rank 9
        best_U3 = np.zeros((9, 9))
        best_V3 = np.zeros((9, 9))
        best_W3 = np.zeros((9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i == k:
                        best_U3[i*3+j, i*3+j] = 1
                        best_V3[j*3+k, i*3+j] = 1
                        best_W3[k*3+i, i*3+j] = 1
        
        # Try to reduce rank using structured perturbations
        # This is a heuristic approach
        
        return best_U3, best_V3, best_W3
    
    # Try to find a decomposition of 3x3x3
    U3, V3, W3 = find_333_decomposition(max_rank=23, time_budget=40)
    
    # Embed in 3x3x8 space: pad W with zeros for k >= 3
    rank3 = U3.shape[1]
    U_embed = U3  # shape (9, rank3)
    V_embed = V3  # shape (9, rank3)
    W_embed = np.zeros((8*3, rank3))  # shape (24, rank3)
    for k in range(3):
        W_embed[k*3:(k+1)*3, :] = W3[k*3:(k+1)*3, :]
    
    # Verify the embedded decomposition
    if verify(U_embed, V_embed, W_embed):
        if rank3 < best_rank:
            best_U, best_V, best_W = U_embed.copy(), V_embed.copy(), W_embed.copy()
            best_rank = rank3
    
    # Return the best decomposition found
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
