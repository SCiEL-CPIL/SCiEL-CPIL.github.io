"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x7 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    # Schoolbook decomposition: rank = n*m*p = 42
    # But we can do better using block decomposition.
    # 
    # Strategy: Use a Kronecker product / block approach.
    # A 2x3 matrix times a 3x7 matrix.
    # We can view the 2x3 as having 2 rows, each of length 3.
    # The 3x7 has 3 rows, each of length 7.
    #
    # Let's think about lower bounds. The slice rank lower bound for matrix multiplication [n,m,p] is:
    # max(n*m, m*p, n*p) ... no, that's not right.
    # The slice rank of the matrix multiplication tensor [n,m,p] is at least max(n*m, m*p, n*p).
    # Actually, for [2,3,7]: max(6, 21, 14) = 21.
    # But slice rank is a lower bound on rank, not always tight.
    #
    # Known results: 
    # - [2,2,2] has rank 7 (Strassen)
    # - [2,3,3] has rank 11 (Strassen-like)
    # - [2,3,7]: Let's try to construct a good decomposition.
    #
    # Approach: Decompose using blocks.
    # A (2x3) * B (3x7) = C (2x7)
    # 
    # Let's try a simple block approach:
    # Split B into two parts: B1 (3x4) and B2 (3x3).
    # Then C = A*B1 + A*B2
    # A*B1 is (2x4), A*B2 is (2x3).
    # 
    # Actually, let's use a more systematic approach.
    # 
    # For [n,m,p], one common technique is to use the fact that:
    # [n,m,p] can be built from smaller tensor products.
    #
    # Let's try: view the 3x7 matrix as a block matrix with blocks of size 3x3 and 3x1.
    # B = [B1 | B2] where B1 is 3x3 and B2 is 3x1 (i.e., 3x4+3).
    # Wait, 7 = 3+4. Let's use B1 (3x3) and B2 (3x4).
    # Then C = A*B1 + A*B2, where A*B1 is 2x3 and A*B2 is 2x4.
    #
    # Rank of [2,3,3] is 11 (known optimal).
    # Rank of [2,3,4]: unknown, but at most 2*3*4=24, and at least max(6,12,8)=12.
    #
    # Alternatively, use a direct construction.
    #
    # Let me try a different approach: use the fact that we can share computations.
    #
    # For a 2x3 times 3x7:
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1,2}, k in {0,...,6}
    #
    # Total terms: 2*3*7 = 42.
    #
    # Can we do better? Let's try to use Strassen-like ideas on the j dimension.
    #
    # Actually, let's just implement a search for a good decomposition using the following:
    # 1. Start with schoolbook (rank 42)
    # 2. Try to merge terms using linear algebra
    #
    # But given time constraints, let's use a known good construction.
    #
    # For [2,3,7], a reasonable upper bound can be obtained by:
    # - Using rank 11 for the [2,3,3] part (Strassen-like)
    # - Using schoolbook for the remaining [2,3,4] part: rank 24
    # - Total: 11 + 24 = 35
    #
    # Or split differently: 7 = 4+3
    # [2,3,4] + [2,3,3] = 24 + 11 = 35
    #
    # Can we do better? Let's try 7 = 2+2+2+1
    # [2,3,2] * 3 + [2,3,1]
    # [2,3,2]: at most 12, at least max(6,6,4)=6. Probably rank 8 or 9.
    # [2,3,1]: rank 6 (each of 6 entries needs its own term)
    # Total: 3*9 + 6 = 33 (if [2,3,2] has rank 9)
    #
    # Let's try to be smarter. Use the following construction:
    # 
    # We'll use a "tensor product" style decomposition.
    #
    # Actually, let me just implement a simple but effective approach:
    # Use the schoolbook but try to find linear dependencies to reduce rank.
    #
    # Given the time budget, let's use a practical approach:
    # 1. Build the schoolbook decomposition (rank 42)
    # 2. Try to find a smaller decomposition using known results
    #
    # For now, let's use a block decomposition with rank 35 as a safe upper bound.
    #
    # Actually, let me try a direct construction with rank around 28-32.
    #
    # Here's a plan:
    # - Use a Strassen-like 2x2 block decomposition on the first two dimensions
    # - Handle the 7 columns separately
    #
    # Let me implement a search-based approach with a time limit.
    
    def schoolbook_decomposition():
        """Standard schoolbook decomposition, rank = n*m*p"""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify that the decomposition is correct."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        # Check dimensions
        if len(U) != n * m:
            return False
        if len(V) != m * p:
            return False
        if len(W) != p * n:
            return False
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        # Verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta
        # Actually, the tensor is: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # This should equal 1 if this is a valid term, but actually the matrix multiplication
        # tensor has T[i,j,k] = 1 for all (i,j,k) in the support.
        # Wait, the matrix multiplication tensor is:
        # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * (A_i * B_j * C_k)
        # No, the tensor for matrix multiplication [n,m,p] is:
        # T[i,j,k] = 1 for all i,j,k (it's the "all ones" tensor in the relevant dimensions)
        # Actually no. The matrix multiplication tensor is:
        # M_{n,m,p} = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
        # where the multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor has T[i,j,k] = 1 for all (i,j,k).
        
        # The decomposition says: M = sum_r U_r ⊗ V_r ⊗ W_r
        # where U_r is a vector of length n*m, V_r of length m*p, W_r of length p*n
        # U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] should sum to 1 for all (i,j,k)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    # Start with schoolbook
    best = schoolbook_decomposition()
    best_rank = n * m * p
    
    # Try block decomposition: split p into parts
    # For [2,3,p], we can split p into p1 + p2
    # Rank <= rank([2,3,p1]) + rank([2,3,p2])
    
    def block_decomposition(p_split):
        """
        Decompose [n,m,p] by splitting p into p_split parts.
        p_split is a list of sizes that sum to p.
        """
        U_list = []
        V_list = []
        W_list = []
        offset = 0
        for ps in p_split:
            # Schoolbook for [n,m,ps]
            rank_ps = n * m * ps
            U_ps = [[0] * rank_ps for _ in range(n * m)]
            V_ps = [[0] * rank_ps for _ in range(m * ps)]
            W_ps = [[0] * rank_ps for _ in range(ps * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(ps):
                        U_ps[i * m + j][col] = 1
                        V_ps[j * ps + k][col] = 1
                        W_ps[k * n + i][col] = 1
                        col += 1
            U_list.append(U_ps)
            V_list.append(V_ps)
            W_list.append(W_ps)
            offset += ps
        
        # Now combine: the V and W need to be padded to full size
        total_rank = sum(len(U_list[t][0]) for t in range(len(p_split)))
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col_offset = 0
        k_offset = 0
        for t, ps in enumerate(p_split):
            rank_ps = n * m * ps
            for i in range(n):
                for j in range(m):
                    for r in range(rank_ps):
                        U[i * m + j][col_offset + r] = U_list[t][i * m + j][r]
            for j in range(m):
                for k in range(ps):
                    for r in range(rank_ps):
                        V[j * p + k_offset + k][col_offset + r] = V_list[t][j * ps + k][r]
            for k in range(ps):
                for i in range(n):
                    for r in range(rank_ps):
                        W[(k_offset + k) * n + i][col_offset + r] = W_list[t][k * n + i][r]
            col_offset += rank_ps
            k_offset += ps
        
        return {"U": U, "V": V, "W": W}
    
    # Try various splits of 7
    splits_to_try = [
        [7],
        [3, 4],
        [4, 3],
        [2, 2, 3],
        [2, 3, 2],
        [3, 2, 2],
        [1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 1],
        [1, 2, 2, 2],
        [2, 1, 2, 2],
        [2, 2, 1, 2],
    ]
    
    for split in splits_to_try:
        if sum(split) != p:
            continue
        if time.time() > deadline - 5:
            break
        dec = block_decomposition(split)
        if verify(dec["U"], dec["V"], dec["W"]):
            rank = len(dec["U"][0])
            if rank < best_rank:
                best_rank = rank
                best = dec
    
    # Try a more sophisticated approach: use known Strassen-like for [2,3,3]
    # Rank of [2,3,3] is 11. Let's implement that.
    #
    # For [2,3,3], one known rank-11 decomposition:
    # This is non-trivial. Let me use a simpler approach for now.
    #
    # Let's try to use a "Kronecker" style: 
    # [2,3,7] = [2,3,3] ⊗ [1,1,1] + [2,3,4]
    # But that's not quite right.
    
    # For now, return the best found
    return best
# EVOLVE-BLOCK-END
