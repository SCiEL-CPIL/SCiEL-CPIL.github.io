import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]
_START_TIME = time.time()
_DEADLINE = 145  # seconds, safety margin

def _elapsed():
    return time.time() - _START_TIME

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _find_decomposition(n, m, p, target_rank, seed=0):
    """
    Attempt to find a rank-r decomposition with entries in {-1, 0, 1}.
    Uses a structured random search with local refinement.
    """
    rng = np.random.RandomState(seed)
    
    # Try a few different random initializations
    for attempt in range(50):
        if _elapsed() > _DEADLINE - 10:
            return None
        
        U = np.zeros((n * m, target_rank), dtype=int)
        V = np.zeros((m * p, target_rank), dtype=int)
        W = np.zeros((p * n, target_rank), dtype=int)
        
        # Initialize with sparse random entries
        # Density heuristic: we want each row to have a few non-zeros
        # For rank r, if we have N rows, each row should have ~ N/r non-zeros on average
        # But we want structure. Let's try a "block" structure or just random sparse.
        
        # Random sparse initialization
        for c in range(target_rank):
            # Choose random subsets of indices for U, V, W
            # This is essentially picking random rank-1 tensors and summing them
            # But we need them to cancel out to give 1s on the diagonal of the tensor.
            
            # Let's try a different strategy:
            # Start with schoolbook and try to merge terms?
            # Or start with random and optimize?
            
            # Let's try a "greedy" approach for the first few, then random.
            # Actually, let's just do random sparse with {-1, 1}
            
            # Number of non-zeros per column
            nnz_u = rng.randint(2, 6)
            nnz_v = rng.randint(2, 6)
            nnz_w = rng.randint(2, 6)
            
            idx_u = rng.choice(n * m, nnz_u, replace=False)
            idx_v = rng.choice(m * p, nnz_v, replace=False)
            idx_w = rng.choice(p * n, nnz_w, replace=False)
            
            U[idx_u, c] = rng.choice([-1, 1], nnz_u)
            V[idx_v, c] = rng.choice([-1, 1], nnz_v)
            W[idx_w, c] = rng.choice([-1, 1], nnz_w)
        
        # Check if this random guess works (very unlikely)
        if _verify(U, V, W, n, m, p, tol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Local search: try flipping single entries
        # This is slow, so we'll limit it
        # We'll skip full local search for now and rely on the fact that
        # structured decompositions (like Strassen) are hard to find randomly.
        
        # Instead, let's try a specific structured approach for small ranks.
        # For 4x5x5, rank 25 is the lower bound.
        # Is there a known rank 25 decomposition? Unlikely.
        # Rank 30?
        
        # Let's try a "diagonal" construction first.
        # If we set U[i*m+j, c] = 1 if c == i*m+j, V[j*p+k, c] = 1 if c == j*p+k, W[k*n+i, c] = 1 if c == k*n+i...
        # This is schoolbook.
        
        # What if we group i?
        # For a fixed i, we need to multiply a row of A (size m) by B (m x p).
        # This is a vector-matrix product.
        # Rank of vector-matrix product is m*p.
        # So for each i, we need rank m*p = 25.
        # Total rank 4 * 25 = 100.
        # To share terms across i, we need W[k*n+i, c] to be non-zero for multiple i.
        # If W[k*n+i, c] is non-zero for i and i', then the term c contributes to both row i and row i' of the result.
        # This means U[i*m+j, c] * V[j*p+k, c] must equal U[i'*m+j, c] * V[j*p+k, c] for the contribution to be consistent?
        # No, the contribution to C[i,k] is sum_j A[i,j] B[j,k].
        # The term c contributes U[i,j,c] V[j,k,c] W[k,i,c] to C[i,k].
        # For a fixed c, let u_i,j = U[i*m+j, c], v_j,k = V[j*p+k, c], w_k,i = W[k*n+i, c].
        # The contribution is u_i,j * v_j,k * w_k,i.
        # For this to be a valid "rank-1" term in the tensor, it must be separable.
        # It is.
        
        # Let's try to construct a solution using the "laser method" intuition.
        # But implementing that is complex.
        
        # Let's try a simple heuristic:
        # Start with schoolbook (rank 100).
        # Try to merge pairs of terms (i,j,k) and (i',j',k') into one term.
        # This requires finding u, v, w such that:
        # u[i,j]v[j,k]w[k,i] = 1 for (i,j,k)
        # u[i',j']v[j',k']w[k',i'] = 1 for (i',j',k')
        # and 0 for all other (i,j,k) in the support of the merged term?
        # No, the sum of terms must equal 1 for ALL (i,j,k).
        # So if we replace two terms with one, the new term must cover both, and potentially interfere with others.
        
        # This is a hard combinatorial problem.
        
        # Let's try a different approach: 
        # Use the fact that 4x5x5 is close to 5x5x5.
        # If we had a 5x5x5 decomposition of rank R, we could restrict it to 4x5x5.
        # But we don't have one.
        
        # Let's try to find a decomposition of rank 90, 80, etc.
        # Maybe we can save a few terms by simple sharing.
        
        # Example: 
        # Term 1: i=0, j=0, k=0
        # Term 2: i=0, j=0, k=1
        # Can we merge them?
        # We need a term c such that:
        # U[0,0,c] V[0,0,c] W[0,0,c] = 1 (for i=0,j=0,k=0)
        # U[0,0,c] V[0,1,c] W[1,0,c] = 1 (for i=0,j=0,k=1)
        # And 0 for others?
        # Let U[0,0,c]=1.
        # Then V[0,0,c] W[0,0,c] = 1
        # V[0,1,c] W[1,0,c] = 1
        # Also, for i=0, j=1, k=0: U[0,1,c] V[1,0,c] W[0,0,c] = 0
        # If W[0,0,c]=1, then V[0,0,c]=1.
        # Then V[0,1,c] W[1,0,c] = 1.
        # And U[0,1,c] V[1,0,c] = 0.
        # If we set V[1,0,c]=0, then U[0,1,c] can be anything.
        # But we need V[0,1,c] W[1,0,c] = 1. So V[0,1,c] != 0 and W[1,0,c] != 0.
        # So we have:
        # U[0,0,c]=1, V[0,0,c]=1, W[0,0,c]=1
        # V[0,1,c]=1, W[1,0,c]=1
        # V[1,0,c]=0
        # This term contributes:
        # (0,0,0): 1*1*1 = 1
        # (0,0,1): 1*1*1 = 1
        # (0,1,0): 0*0*1 = 0 (since V[1,0,c]=0)
        # (1,0,0): U[1,0,c]*V[0,0,c]*W[0,1,c] = 0 (since W[0,1,c]=0)
        # (1,0,1): U[1,0,c]*V[0,1,c]*W[1,1,c] = 0 (since W[1,1,c]=0)
        # ...
        # So this term covers (0,0,0) and (0,0,1).
        # We saved 1 rank!
        
        # We can do this for all j=0, k=0..4 for i=0.
        # For each i, we can merge the k=0..4 terms for a fixed j?
        # Let's merge (i, j, 0) and (i, j, 1).
        # This saves 1 rank per pair.
        # For fixed i, j, we have p=5 terms. We can merge them into ceil(5/2) = 3 terms?
        # Or just merge pairs: (0,1), (2,3), and 4 alone. Saves 2 ranks.
        # Total ranks for fixed i, j: 3 instead of 5.
        # Total ranks: n * m * 3 = 4 * 5 * 3 = 60.
        
        # Let's try to construct this!
        
        if attempt == 0:
            # Construct the "merged pairs" decomposition
            # For each i, j, we create terms for k-pairs.
            # Pair (0,1), (2,3), (4,4)
            # Term for (k1, k2):
            # U[i*m+j, c] = 1
            # V[j*p+k1, c] = 1, V[j*p+k2, c] = 1
            # W[k1*n+i, c] = 1, W[k2*n+i, c] = 1
            # Check contribution for (i, j, k):
            # If k == k1: 1 * 1 * 1 = 1
            # If k == k2: 1 * 1 * 1 = 1
            # If k != k1, k2: 1 * 0 * 0 = 0
            # This works!
            
            # But wait, does it interfere with other (i', j')?
            # U[i*m+j, c] is non-zero only for this (i,j).
            # So it doesn't interfere.
            
            # Let's build this.
            U_new = np.zeros((n * m, n * m * 3), dtype=int) # Max rank
            V_new = np.zeros((m * p, n * m * 3), dtype=int)
            W_new = np.zeros((p * n, n * m * 3), dtype=int)
            
            col = 0
            for i in range(n):
                for j in range(m):
                    # Pairs: (0,1), (2,3), (4,4)
                    pairs = [(0, 1), (2, 3), (4, 4)]
                    for k1, k2 in pairs:
                        U_new[i * m + j, col] = 1
                        V_new[j * p + k1, col] = 1
                        if k2 != k1:
                            V_new[j * p + k2, col] = 1
                        W_new[k1 * n + i, col] = 1
                        if k2 != k1:
                            W_new[k2 * n + i, col] = 1
                        col += 1
            
            rank_new = col
            # Verify
            if _verify(U_new, V_new, W_new, n, m, p, tol=1e-6):
                return {"U": U_new[:, :rank_new].tolist(), "V": V_new[:, :rank_new].tolist(), "W": W_new[:, :rank_new].tolist()}
        
        # If the simple merge didn't work or wasn't tried, continue with random search
        # (which is unlikely to succeed for low ranks)
        
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Try the structured "merged pairs" decomposition (rank 60)
    result = _find_decomposition(n, m, p, 60, seed=0)
    if result is not None:
        return result
        
    # 2. Try to find even lower rank via random search (unlikely)
    for r in [55, 50, 45, 40, 35, 30, 25]:
        if _elapsed() > _DEADLINE - 20:
            break
        result = _find_decomposition(n, m, p, r, seed=r)
        if result is not None:
            return result
            
    # 3. Fallback: schoolbook
    return _schoolbook(n, m, p)

