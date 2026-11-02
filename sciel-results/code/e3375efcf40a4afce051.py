"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 3]
DEADLINE_SECONDS = 145  # Safety margin before 150s wall-clock kill

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p, tolerance=1e-9):
    """Check if U, V, W form a valid matrix multiplication decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = sum(U[i * m + j][r] * V[j * p + k][r] for r in range(rank))
                rhs = W[k * n + i][0] if rank == 1 else sum(
                    U[i * m + j][r] * V[j * p + k][r] for r in range(rank)
                )
                # The tensor condition: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 0
                # Actually, the decomposition condition is:
                # (A*B)[i,k] = sum_j A[i,j]*B[j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # Wait, that's not right either. Let me think again.
                #
                # For matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
                # The tensor T has components T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
                # A rank-1 decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # But we need: sum_j A[i,j] B[j,k] = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_j B[j,k] V[j*p+k, r]) * W[k*n+i, r]
                # Hmm, that doesn't look right either.
                #
                # Let me reconsider. The standard form is:
                # C[i,k] = sum_j A[i,j] B[j,k]
                # We want: C[i,k] = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_j B[j,k] V[j*p+k, r]) * W[k*n+i, r]
                # No wait, that would give us products of three sums, which is cubic.
                #
                # The correct formulation:
                # C[i,k] = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_j B[j,k] V[j*p+k, r]) * W[k*n+i, r]
                # is NOT correct because it's cubic in the inputs.
                #
                # The correct bilinear form is:
                # C[i,k] = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_j B[j,k] V[j*p+k, r]) * W[k*n+i, r]
                # Wait, I'm confusing myself. Let me look at this more carefully.
                #
                # The matrix multiplication tensor T has indices (i, j, k) where:
                # - i indexes the row of A (and C), range [0, n)
                # - j indexes the column of A / row of B (and the summation index), range [0, m)  
                # - k indexes the column of B (and C), range [0, p)
                #
                # T[i,j,k] = 1 for all valid (i,j,k)
                #
                # A rank-1 decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                #
                # Then: C[i,k] = sum_j A[i,j] B[j,k] = sum_j T[i,j,k] A[i,j] B[j,k]
                # = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
                # = sum_r W[k*n+i, r] * (sum_j A[i,j] U[i*m+j, r]) * (sum_j B[j,k] V[j*p+k, r])
                #
                # So the decomposition is valid if for all (i,j,k):
                # 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                pass
    
    # Proper verification: check that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r] for r in range(rank))
                if abs(val - 1) > tolerance:
                    return False
    return True


def _search_gf2_decomposition(n, m, p, max_rank, deadline):
    """
    Search for a decomposition over GF(2) using structured rank-1 tensors.
    For small dimensions, we can enumerate possible rank-1 terms.
    """
    dim_u = n * m  # 6
    dim_v = m * p  # 9
    dim_w = p * n  # 6
    
    # Over GF(2), each rank-1 term is a triple of binary vectors (u, v, w)
    # such that the sum of u*r * v*r * w*r over r equals the target tensor.
    #
    # The target tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k), else 0.
    #
    # For GF(2), we need: sum_r (u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]) = T[i*m+j, j*p+k, k*n+i] mod 2
    #
    # This is a hard problem, but for small dimensions we can try greedy/heuristic approaches.
    
    # Build the target tensor as a 3D array
    target = [[[0] * p for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i][j][k] = 1
    
    # Greedy approach: iteratively pick the rank-1 term that best approximates the residual
    residual = [[[0] * p for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                residual[i][j][k] = target[i][j][k]
    
    U_list = []
    V_list = []
    W_list = []
    
    start_time = time.time()
    
    for rank_attempt in range(1, max_rank + 1):
        if time.time() > deadline:
            break
        
        # Try to find a rank-1 term that reduces the residual
        best_term = None
        best_score = -1
        
        # Enumerate possible rank-1 terms over GF(2)
        # u has dim_u bits, v has dim_v bits, w has dim_w bits
        # Too many to enumerate exhaustively, so use a heuristic:
        # Pick u, v, w that maximize the number of 1s in residual where all three are 1
        
        # Simplified heuristic: try random or structured terms
        # For now, just check if we can complete with the current partial decomposition
        # This is a placeholder for a more sophisticated search
        
        # If we've found enough terms, verify
        if len(U_list) >= rank_attempt:
            # Build matrices and verify
            U = [[0] * rank_attempt for _ in range(dim_u)]
            V = [[0] * rank_attempt for _ in range(dim_v)]
            W = [[0] * rank_attempt for _ in range(dim_w)]
            for r in range(len(U_list)):
                for idx in range(dim_u):
                    U[idx][r] = U_list[r][idx]
                for idx in range(dim_v):
                    V[idx][r] = V_list[r][idx]
                for idx in range(dim_w):
                    W[idx][r] = W_list[r][idx]
            
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        # Add a new rank-1 term
        # Heuristic: pick a random or structured term
        # For simplicity, use a structured term based on the residual
        u = [0] * dim_u
        v = [0] * dim_v
        w = [0] * dim_w
        
        # Find positions where residual is 1 and set corresponding entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if residual[i][j][k] == 1:
                        u[i * m + j] = 1
                        v[j * p + k] = 1
                        w[k * n + i] = 1
        
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
        # Update residual (XOR for GF(2))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if u[i * m + j] * v[j * p + k] * w[k * n + i] % 2 == 1:
                        residual[i][j][k] ^= 1
    
    return None


def _search_rational_decomposition(n, m, p, max_rank, deadline):
    """
    Search for a decomposition over rationals using iterative refinement.
    """
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Start with schoolbook and try to reduce rank
    schoolbook = _schoolbook_decomposition(n, m, p)
    current_rank = n * m * p
    
    # Try to find linear dependencies in the rank-1 terms
    # For each pair of terms, check if they can be combined or if one is redundant
    
    U = [row[:] for row in schoolbook["U"]]
    V = [row[:] for row in schoolbook["V"]]
    W = [row[:] for row in schoolbook["W"]]
    
    start_time = time.time()
    
    # Try to eliminate columns by checking if they're linearly dependent
    for target_rank in range(1, current_rank):
        if time.time() > deadline:
            break
        
        # Check if we can reduce to target_rank by finding a subset of columns
        # that still spans the tensor
        # This is a set cover / matroid problem, hard in general
        
        # Simple heuristic: try removing one column at a time and checking validity
        # This is very slow, so just return the best we have
        
        pass
    
    return schoolbook


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # First, always have the schoolbook as a fallback
    schoolbook = _schoolbook_decomposition(n, m, p)
    
    # Try to find a better decomposition
    # For 2x3x3, the known minimum rank is 15 (from Strassen-like constructions)
    # but let's search for it
    
    # Try GF(2) search first (faster due to discrete structure)
    result = _search_gf2_decomposition(n, m, p, 17, deadline)
    if result is not None and _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Try rational search
    result = _search_rational_decomposition(n, m, p, 17, deadline)
    if result is not None and _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Return the schoolbook decomposition as fallback
    return schoolbook
# EVOLVE-BLOCK-END
