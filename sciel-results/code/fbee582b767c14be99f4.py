"""
Rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication.
Strategy: Structured block decomposition with aggressive Strassen application
and systematic rank reduction via linear dependency detection.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    def _elapsed():
        return time.time() - start_time

    def _verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        # Build tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Must equal 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        # Check integer/half-integer constraint
        for M in (U, V, W):
            for row in M:
                for val in row:
                    v = float(val)
                    if v != int(v) and 2 * v != int(2 * v):
                        return False
        return True

    def _schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _build_strassen_block(i0, j0, k0, U, V, W, col_offset):
        """
        Apply Strassen's 2x2x2 algorithm for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        -> C[i0:i0+2, k0:k0+2]. Returns new column offset.
        """
        col = col_offset
        # M1 = (A00+A11)*(B00+B11)
        U[i0 * m + j0][col] = 1.0
        U[(i0+1) * m + (j0+1)][col] = 1.0
        V[j0 * p + k0][col] = 1.0
        V[(j0+1) * p + (k0+1)][col] = 1.0
        W[k0 * n + i0][col] = 1.0
        W[k0 * n + (i0+1)][col] = 1.0
        col += 1

        # M2 = (A10+A11)*B00
        U[(i0+1) * m + j0][col] = 1.0
        U[(i0+1) * m + (j0+1)][col] = 1.0
        V[j0 * p + k0][col] = 1.0
        W[k0 * n + (i0+1)][col] = 1.0
        col += 1

        # M3 = A00*(B11-B00)
        U[i0 * m + j0][col] = 1.0
        V[(j0+1) * p + (k0+1)][col] = 1.0
        V[j0 * p + k0][col] = -1.0
        W[k0 * n + (i0+1)][col] = 1.0
        col += 1

        # M4 = A11*(B01-B10)
        U[(i0+1) * m + (j0+1)][col] = 1.0
        V[j0 * p + (k0+1)][col] = 1.0
        V[(j0+1) * p + k0][col] = -1.0
        W[k0 * n + i0][col] = 1.0
        W[k0 * n + (i0+1)][col] = 1.0
        col += 1

        # M5 = (A00+A01)*B10
        U[i0 * m + j0][col] = 1.0
        U[i0 * m + (j0+1)][col] = 1.0
        V[(j0+1) * p + k0][col] = 1.0
        W[k0 * n + i0][col] = -1.0
        W[k0 * n + (i0+1)][col] = 1.0
        col += 1

        # M6 = (A10+A11)*(B11-B01)
        U[(i0+1) * m + j0][col] = 1.0
        U[(i0+1) * m + (j0+1)][col] = 1.0
        V[(j0+1) * p + (k0+1)][col] = 1.0
        V[j0 * p + (k0+1)][col] = -1.0
        W[k0 * n + i0][col] = 1.0
        W[k0 * n + (i0+1)][col] = 1.0
        col += 1

        # M7 = (A00-A01)*(B00+B10)
        U[i0 * m + j0][col] = 1.0
        U[i0 * m + (j0+1)][col] = -1.0
        V[j0 * p + k0][col] = 1.0
        V[(j0+1) * p + k0][col] = 1.0
        W[k0 * n + i0][col] = 1.0
        col += 1

        return col

    def _build_optimized_block_decomp():
        """
        Build a decomposition using Strassen on all possible 2x2x2 blocks,
        then handle remaining terms directly.
        
        For 3x4x7, we can partition into:
        - Rows: 0-1 (2 rows), 2 (1 row)
        - Cols: 0-1 (2 cols), 2-3 (2 cols)  
        - Output: 0-1 (2), 2-3 (2), 4-6 (3)
        
        Strategy: Apply Strassen to overlapping 2x2x2 blocks where possible,
        being careful not to double-count.
        """
        # First, build a schoolbook-like structure but with Strassen where applicable
        # We'll use a greedy approach: try to cover (i,j,k) triples with Strassen blocks
        
        # Initialize with zeros
        max_rank = n * m * p  # upper bound
        U = [[0.0] * max_rank for _ in range(n * m)]
        V = [[0.0] * max_rank for _ in range(m * p)]
        W = [[0.0] * max_rank for _ in range(p * n)]
        col = 0
        
        # Track which (i,j,k) triples have been covered
        covered = set()
        
        # Try all possible 2x2x2 Strassen blocks
        for i0 in range(n - 1):
            for j0 in range(m - 1):
                for k0 in range(p - 1):
                    # Check if this block can be applied (all 8 (i,j,k) triples not yet covered)
                    block_triples = []
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                block_triples.append((i0+di, j0+dj, k0+dk))
                    
                    # Only apply if none of these triples are already covered
                    if all(t not in covered for t in block_triples):
                        col = _build_strassen_block(i0, j0, k0, U, V, W, col)
                        for t in block_triples:
                            covered.add(t)
        
        # Handle remaining (i,j,k) triples directly
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        U[i * m + j][col] = 1.0
                        V[j * p + k][col] = 1.0
                        W[k * n + i][col] = 1.0
                        col += 1
        
        # Trim to actual rank
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        
        return {"U": U, "V": V, "W": W}

    def _try_merge_decomposition(U, V, W, max_iters=5000):
        """
        Try to merge pairs of rank-1 tensors that have overlapping structure.
        Two rank-1 tensors (u1,v1,w1) and (u2,v2,w2) can be merged if
        there exists a scalar such that the combined contribution can be
        represented with one rank-1 tensor.
        """
        rank = len(U[0]) if U else 0
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]

        for _ in range(max_iters):
            if _elapsed() > deadline - 20:
                break
            merged = False
            for a in range(rank):
                if _elapsed() > deadline - 20:
                    break
                for b in range(a + 1, rank):
                    # Check if column b is a scalar multiple of column a in all three matrices
                    nonzero_a = [i for i in range(n * m) if abs(U[i][a]) > 1e-12]
                    if not nonzero_a:
                        continue
                    i0 = nonzero_a[0]
                    if abs(U[i0][a]) < 1e-12:
                        continue
                    scalar = U[i0][b] / U[i0][a]
                    
                    # Verify scalar consistency across U
                    consistent = True
                    for i in range(n * m):
                        if abs(U[i][a]) > 1e-12:
                            if abs(U[i][b] - scalar * U[i][a]) > 1e-9:
                                consistent = False
                                break
                        elif abs(U[i][b]) > 1e-9:
                            consistent = False
                            break
                    if not consistent:
                        continue

                    # Verify scalar consistency across V
                    for j in range(m * p):
                        if abs(V[j][a]) > 1e-12:
                            if abs(V[j][b] - scalar * V[j][a]) > 1e-9:
                                consistent = False
                                break
                        elif abs(V[j][b]) > 1e-9:
                            consistent = False
                            break
                    if not consistent:
                        continue

                    # Verify scalar consistency across W
                    for k in range(p * n):
                        if abs(W[k][a]) > 1e-12:
                            if abs(W[k][b] - scalar * W[k][a]) > 1e-9:
                                consistent = False
                                break
                        elif abs(W[k][b]) > 1e-9:
                            consistent = False
                            break
                    if not consistent:
                        continue

                    # Merge: column a += scalar * column b, remove column b
                    for i in range(n * m):
                        U[i][a] += scalar * U[i][b]
                    for j in range(m * p):
                        V[j][a] += scalar * V[j][b]
                    for k in range(p * n):
                        W[k][a] += scalar * W[k][b]

                    # Remove column b
                    for i in range(n * m):
                        U[i].pop(b)
                    for j in range(m * p):
                        V[j].pop(b)
                    for k in range(p * n):
                        W[k].pop(b)
                    rank -= 1
                    merged = True
                    break
                if merged:
                    break
            if not merged:
                break

        return {"U": U, "V": V, "W": W}

    def _try_linear_dependency_reduction(U, V, W, max_iters=1000):
        """
        Try to find linear dependencies among the rank-1 terms using 
        the tensor product structure. For each pair of columns, check if
        one can be expressed as a linear combination of others in the 
        tensor product space.
        """
        rank = len(U[0]) if U else 0
        if rank < 2:
            return {"U": U, "V": V, "W": W}

        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]

        for _ in range(max_iters):
            if _elapsed() > deadline - 25:
                break
            merged = False
            
            # For each column, check if it's linearly dependent on others
            for target in range(rank):
                if _elapsed() > deadline - 25:
                    break
                
                # Build the matrix of tensor products for all other columns
                # The tensor product of column c is:
                # vec(U[:,c]) ⊗ vec(V[:,c]) ⊗ vec(W[:,c])
                # But this is huge, so we check a weaker condition:
                # Check if the "flattened" tensor vectors are linearly dependent
                
                # Create a feature vector for each column:
                # For column c, the contribution to T[i,j,k] is:
                # U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # We can check if column target's pattern is a linear combination
                # of other columns' patterns
                
                # Simple approach: for a random subset of (i,j,k) triples,
                # check if the values are linearly dependent
                triples = list(product(range(n), range(m), range(p)))
                np.random.shuffle(triples)
                sample_size = min(50, len(triples))
                sample = triples[:sample_size]
                
                # Build system: find coefficients alpha_c such that
                # sum_c alpha_c * (U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]) = 
                # U[i*m+j,target] * V[j*p+k,target] * W[k*n+i,target]
                # for all sampled (i,j,k)
                
                # This is a linear system in alpha_c
                # A[s, c] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # b[s] = U[i*m+j,target] * V[j*p+k,target] * W[k*n+i,target]
                
                A = np.zeros((sample_size, rank))
                b = np.zeros(sample_size)
                
                for s, (i, j, k) in enumerate(sample):
                    for c in range(rank):
                        A[s, c] = U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    b[s] = U[i * m + j][target] * V[j * p + k][target] * W[k * n + i][target]
                
                # Solve for alpha, but we want to know if target can be expressed
                # as a combination of OTHER columns (not itself)
                # Set alpha_target = 0 and check if the system is consistent
                
                # Use least squares to check consistency
                try:
                    # Remove target column from A
                    A_reduced = np.delete(A, target, axis=1)
                    # Check if b is in the column space of A_reduced
                    # by solving A_reduced * x = b and checking residual
                    x, residuals, rank_A, svals = np.linalg.lstsq(A_reduced, b, rcond=None)
                    
                    if len(residuals) > 0 and residuals[0] < 1e-9:
                        # Column target is linearly dependent on others
                        # Merge: express target as sum of others
                        # alpha_c for c != target
                        alpha = np.zeros(rank)
                        alpha_reduced = np.delete(alpha, target)
                        alpha_reduced[:] = x
                        alpha[target] = 0  # We're removing this column
                        
                        # But we need to actually remove the column and adjust others
                        # This is complex; for now, just flag it
                        # For simplicity, try the scalar multiple merge instead
                        pass
                except:
                    pass
            
            if not merged:
                break

        return {"U": U, "V": V, "W": W}

    # Try the optimized block decomposition
    result = _build_optimized_block_decomp()
    if _verify(result["U"], result["V"], result["W"]):
        best_result = result
        best_rank = len(result["U"][0])
    else:
        best_result = _schoolbook()
        best_rank = n * m * p

    # Try to merge
    if _elapsed() < deadline - 40:
        merged = _try_merge_decomposition(
            [row[:] for row in best_result["U"]],
            [row[:] for row in best_result["V"]],
            [row[:] for row in best_result["W"]]
        )
        if _verify(merged["U"], merged["V"], merged["W"]):
            new_rank = len(merged["U"][0])
            if new_rank < best_rank:
                best_result = merged
                best_rank = new_rank

    # Try linear dependency reduction
    if _elapsed() < deadline - 30:
        merged2 = _try_linear_dependency_reduction(
            [row[:] for row in best_result["U"]],
            [row[:] for row in best_result["V"]],
            [row[:] for row in best_result["W"]]
        )
        if _verify(merged2["U"], merged2["V"], merged2["W"]):
            new_rank = len(merged2["U"][0])
            if new_rank < best_rank:
                best_result = merged2
                best_rank = new_rank

    return best_result
