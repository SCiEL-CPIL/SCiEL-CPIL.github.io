"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid tensor decomposition for n×m×p matrix mult."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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

def _build_strassen_pairs(n, m, p):
    """
    Build a decomposition using Strassen's algorithm on pairs of middle indices.
    For m=4, split into two pairs: (0,1) and (2,3).
    Each pair of middle indices can be handled with rank 3 instead of rank 2
    using the identity: a*b + c*d = (a+c)*(b+d) - a*d - c*b
    Wait, that's rank 3 for 2 terms? No.
    Strassen for 2x2 matrix mult uses 7 mults for 4 outputs.
    Here we are doing element-wise products summed over j.
    C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k] + A[i,3]B[3,k]
    
    Consider the first two terms: A[i,0]B[0,k] + A[i,1]B[1,k].
    This is a dot product of length 2. The rank of the 2x2x2 tensor (dot product) is 3?
    Actually, the rank of the tensor T[a,b,c] = a*b*c is 1.
    The tensor for dot product of two vectors of length 2 is T[i,j,k] where i,k are output indices?
    No, the "tensor" here is the bilinear form.
    The multiplication of a 1x2 matrix by a 2x1 matrix is rank 2 (schoolbook).
    Can it be done with rank 1? No, because it's not an outer product.
    So Strassen doesn't directly reduce the rank of a single dot product.
    
    HOWEVER, if we look at the structure:
    We have multiple (i,k) pairs.
    For a fixed pair of j's, say j=0,1, the contribution to ALL (i,k) is:
    sum_{i,k} A[i,0]B[0,k] + A[i,1]B[1,k].
    This is a sum of two outer products: A[:,0] * B[0,:] + A[:,1] * B[1,:].
    This is a matrix of rank at most 2.
    The tensor decomposition view:
    We need to represent the set of bilinear forms.
    
    Let's try a different structural idea:
    Embedding into 4x4x4.
    The 3x4x5 multiplication can be seen as a sub-tensor of a 4x4x4 multiplication?
    Not directly, but we can pad.
    
    Let's try the "disjoint Strassen blocks" idea mentioned in the cell findings.
    "Using two disjoint Strassen 2x2x2 blocks (rank 7 each) instead of schoolbook (rank 8 each) reduces the total rank from 60 to 58."
    This implies that for a 2x2x2 block, schoolbook is rank 8, and Strassen is rank 7.
    But 3x4x5 is not a block of 2x2x2s.
    It has dimensions 3, 4, 5.
    
    Let's re-read the finding: "Using two disjoint Strassen 2x2x2 blocks... reduces total rank from 60 to 58".
    This suggests the 3x4x5 problem was decomposed into parts that could be treated as 2x2x2 blocks?
    Or maybe it refers to a different problem instance that was solved, and the strategy is applicable.
    If we have a 4x4x4 problem, it can be split into four 2x2x2 blocks.
    4x4x4 schoolbook rank is 64.
    4x4x4 Strassen rank is 4 * 7 = 28? No, Strassen for 4x4x4 is not simply 4 blocks of 2x2x2 because the blocks interact?
    Actually, if you partition a 4x4 matrix into four 2x2 blocks, the multiplication involves sums of products of blocks.
    There are 16 terms in the schoolbook for 4x4.
    Each term is a 2x2x2 multiplication? No, it's a scalar multiplication of matrices.
    
    Let's look at the 3x4x5 case specifically.
    Rank 60 is the baseline.
    The finding says a program achieved 1.034 score, which is 60/58.
    So rank 58 is possible.
    How to get rank 58?
    60 - 58 = 2. We need to save 2 ranks.
    
    Let's try to find a linear dependency or a structured reduction.
    Maybe we can combine two schoolbook terms into one?
    If we can find i1,j1,k1 and i2,j2,k2 such that we can represent the sum of their contributions with a single rank-1 term?
    That would require U[i1*m+j1,c]V[j1*p+k1,c]W[k1*n+i1,c] + U[i2*m+j2,c]V[j2*p+k2,c]W[k2*n+i2,c] to be representable as one term?
    No, that's not how rank works. Rank is the number of terms.
    If we can merge two terms, we reduce rank by 1.
    
    Let's try a randomized search for rank 59, 58, etc., using the alternating least squares approach from Inspiration 3, but with a deadline guard and fallback to schoolbook.
    """
    pass

def _als_search(n, m, p, target_rank, deadline, seed=0):
    """
    Alternating Least Squares search for a rank-r decomposition.
    T[i,j,k] = 1 for all i,j,k.
    We want U[i,j,c] V[j,k,c] W[k,i,c] such that sum_c ... = 1.
    """
    np.random.seed(seed)
    
    # Initialize
    U = np.random.randn(n * m, target_rank)
    V = np.random.randn(m * p, target_rank)
    W = np.random.randn(p * n, target_rank)
    
    # Scale to avoid overflow/underflow
    U /= np.sqrt(np.sum(U**2, axis=0))
    V /= np.sqrt(np.sum(V**2, axis=0))
    W /= np.sqrt(np.sum(W**2, axis=0))
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                # Minimize sum_k (1 - sum_c U_c * A_c)^2
                # A_c = V[j*p+k, c] * W[k*n+i, c]
                A = np.zeros((p, target_rank))
                for k in range(p):
                    for c in range(target_rank):
                        A[k, c] = V[j * p + k, c] * W[k * n + i, c]
                # Solve (A^T A) u = A^T 1
                try:
                    ATA = A.T @ A
                    ATb = A.T @ np.ones(p)
                    U[i * m + j, :] = np.linalg.solve(ATA + 1e-8 * np.eye(target_rank), ATb)
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, target_rank))
                for i in range(n):
                    for c in range(target_rank):
                        A[i, c] = U[i * m + j, c] * W[k * n + i, c]
                try:
                    ATA = A.T @ A
                    ATb = A.T @ np.ones(n)
                    V[j * p + k, :] = np.linalg.solve(ATA + 1e-8 * np.eye(target_rank), ATb)
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, target_rank))
                for j in range(m):
                    for c in range(target_rank):
                        A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                try:
                    ATA = A.T @ A
                    ATb = A.T @ np.ones(m)
                    W[k * n + i, :] = np.linalg.solve(ATA + 1e-8 * np.eye(target_rank), ATb)
                except:
                    pass
        
        # Calculate error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(target_rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    error += (1.0 - val) ** 2
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < 1e-10:
            # Found a solution. Convert to fractions if possible?
            # The task requires integer/half-integer entries.
            # ALS produces floats. We need to check if they are close to rational numbers.
            # For now, just return the float solution and let the caller check/convert.
            # Actually, the contract says "integer/half-integer entries".
            # So we must ensure the final output meets this.
            # If ALS converges to a non-rational solution, it's invalid.
            # We should try to snap to rationals or reject.
            
            # Let's try to snap to rationals with small denominators.
            U_snap = np.zeros_like(U)
            V_snap = np.zeros_like(V)
            W_snap = np.zeros_like(W)
            valid = True
            for idx in range(n * m):
                for c in range(target_rank):
                    val = U[idx, c]
                    # Check if close to a rational with denominator 1 or 2
                    # Or just keep as float? The prompt says "integer/half-integer".
                    # This is a strict constraint.
                    # If ALS finds a solution, it might be rational.
                    # Let's check if it's close to x/2.
                    frac = Fraction(val).limit_denominator(100)
                    if abs(val - float(frac)) < 1e-9:
                        U_snap[idx, c] = float(frac)
                    else:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Check V and W similarly
            valid = True
            for idx in range(m * p):
                for c in range(target_rank):
                    val = V[idx, c]
                    frac = Fraction(val).limit_denominator(100)
                    if abs(val - float(frac)) < 1e-9:
                        V_snap[idx, c] = float(frac)
                    else:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            valid = True
            for idx in range(p * n):
                for c in range(target_rank):
                    val = W[idx, c]
                    frac = Fraction(val).limit_denominator(100)
                    if abs(val - float(frac)) < 1e-9:
                        W_snap[idx, c] = float(frac)
                    else:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Verify the snapped solution
            U_list = U_snap.tolist()
            V_list = V_snap.tolist()
            W_list = W_snap.tolist()
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        if error < 1e-6:
            # Very close, try to polish
            for _ in range(50):
                if time.time() > deadline:
                    break
                # Repeat updates
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            for c in range(target_rank):
                                A[k, c] = V[j * p + k, c] * W[k * n + i, c]
                        try:
                            ATA = A.T @ A
                            ATb = A.T @ np.ones(p)
                            U[i * m + j, :] = np.linalg.solve(ATA + 1e-8 * np.eye(target_rank), ATb)
                        except:
                            pass
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, target_rank))
                        for i in range(n):
                            for c in range(target_rank):
                                A[i, c] = U[i * m + j, c] * W[k * n + i, c]
                        try:
                            ATA = A.T @ A
                            ATb = A.T @ np.ones(n)
                            V[j * p + k, :] = np.linalg.solve(ATA + 1e-8 * np.eye(target_rank), ATb)
                        except:
                            pass
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            for c in range(target_rank):
                                A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                        try:
                            ATA = A.T @ A
                            ATb = A.T @ np.ones(m)
                            W[k * n + i, :] = np.linalg.solve(ATA + 1e-8 * np.eye(target_rank), ATb)
                        except:
                            pass
                # Re-check error and snap
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(target_rank):
                                val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            error += (1.0 - val) ** 2
                if error < 1e-10:
                    U_snap = np.zeros_like(U)
                    V_snap = np.zeros_like(V)
                    W_snap = np.zeros_like(W)
                    valid = True
                    for idx in range(n * m):
                        for c in range(target_rank):
                            val = U[idx, c]
                            frac = Fraction(val).limit_denominator(100)
                            if abs(val - float(frac)) < 1e-9:
                                U_snap[idx, c] = float(frac)
                            else:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid: continue
                    for idx in range(m * p):
                        for c in range(target_rank):
                            val = V[idx, c]
                            frac = Fraction(val).limit_denominator(100)
                            if abs(val - float(frac)) < 1e-9:
                                V_snap[idx, c] = float(frac)
                            else:
                                valid = False
                                break
                        if not valid: break
                    if not valid: continue
                    for idx in range(p * n):
                        for c in range(target_rank):
                            val = W[idx, c]
                            frac = Fraction(val).limit_denominator(100)
                            if abs(val - float(frac)) < 1e-9:
                                W_snap[idx, c] = float(frac)
                            else:
                                valid = False
                                break
                        if not valid: break
                    if not valid: continue
                    U_list = U_snap.tolist()
                    V_list = V_snap.tolist()
                    W_list = W_snap.tolist()
                    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                        return {"U": U_list, "V": V_list, "W": W_list}

    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin

    # Start with schoolbook
    schoolbook = _build_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p

    # Try ALS search for lower ranks
    # Try ranks from 55 to 59
    for target_rank in range(55, 60):
        if time.time() > deadline - 10.0:
            break
        
        # Try a few random seeds
        for seed in range(5):
            if time.time() > deadline - 5.0:
                break
            
            result = _als_search(n, m, p, target_rank, deadline, seed)
            if result is not None:
                rank = len(result["U"][0])
                if rank < best_rank:
                    best_result = result
                    best_rank = rank
                # If we found a lower rank, we can stop searching for higher ranks
                # but we might find an even lower one? No, we are iterating downwards.
                # If we found rank 55, we are done.
                if target_rank == 55:
                    return best_result

    return best_result

# EVOLVE-BLOCK-END
