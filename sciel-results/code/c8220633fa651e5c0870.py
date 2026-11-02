"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a low-rank matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    schoolbook_rank = n * m * p  # 48
    
    # Start with schoolbook as fallback
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_best, V_best, W_best, rank_best = make_schoolbook()
    
    # Verify a decomposition is valid
    def verify(U, V, W):
        """Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for rr in range(r):
                        total += U[i * m + j, rr] * V[j * p + k, rr] * W[k * n + i, rr]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    # Try to find lower-rank decompositions
    # Strategy: Try structured patterns where some schoolbook columns are merged
    
    # For rank 47: try merging two columns that share structure
    # Column (i,j,k) has U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    
    # Two columns (i1,j1,k1) and (i2,j2,k2) can potentially be merged if
    # we can find weights a,b such that the combined tensor matches
    
    # Actually, for exact decomposition with integer/half-integer entries,
    # let's try a more systematic approach
    
    # Try rank from 47 down to 30 with structured search
    for target_rank in range(schoolbook_rank - 1, 29, -1):
        if time.time() > deadline:
            break
        
        # Try structured decomposition:
        # Keep most schoolbook columns, but try to merge some pairs
        
        # For rank 47: remove one column and adjust neighbors
        # This is complex, so let's try a different approach
        
        # Use numpy to solve a least-squares problem for small rank reductions
        # Build the constraint matrix
        
        # The tensor T has 48 entries (all 1s)
        # For rank r, we have r * (6 + 24 + 16) = 40r variables
        # But the constraints are 48 equations (one per (i,j,k))
        
        # For r < 48, this is underdetermined, so solutions may exist
        
        # Let's try a greedy approach: start with schoolbook and try to eliminate columns
        
        if target_rank == schoolbook_rank - 1:
            # Try to eliminate one column by expressing it as combination of others
            # Column (i,j,k) can be eliminated if its "fingerprint" is in the span of others
            
            # For simplicity, try all pairs and see if one can be merged
            for elim_k in range(p):
                if time.time() > deadline:
                    break
                for elim_j in range(m):
                    if time.time() > deadline:
                        break
                    for elim_i in range(n):
                        if time.time() > deadline:
                            break
                        
                        # Try to express column (elim_i, elim_j, elim_k) as combination
                        # of other columns
                        # This requires solving a linear system
                        
                        # For now, just try a specific pattern:
                        # Merge (i,j,k) and (i,j,k+1) if they share U and V patterns
                        pass
        
        # Try a constructive approach for specific ranks
        # For 2x3x8, try using the structure:
        # A (2x3) * B (3x8) = C (2x8)
        
        # Split B into B1 (3x4) and B2 (3x4)
        # C = A*B1 + A*B2
        
        # For each 2x3x4, try to find a rank-11 or rank-12 decomposition
        # If we can get rank 11 for each, total rank = 22, but that seems too optimistic
        
        # More realistically, try rank 12 for each block (schoolbook)
        # Total: 24, but we might share some factors
        
        # Let's try a direct construction for rank 44-47
        # by using half-integer weights to merge columns
        
        # For rank 47: try to merge two adjacent k-columns
        # Columns (i,j,k) and (i,j,k+1) share U[i*m+j] and differ in V and W
        # V[j*p+k] vs V[j*p+k+1], W[k*n+i] vs W[(k+1)*n+i]
        
        # If we set U[i*m+j] = 1/2 for both, and adjust V,W appropriately...
        # This is getting complex. Let's just return schoolbook if we can't find better quickly.
    
    # Return the best found (schoolbook for now)
    # Convert to lists for the required format
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
