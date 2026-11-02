import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # 5s safety margin
    
    # Schoolbook decomposition (rank = n*m*p = 48)
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    def to_list(mat):
        return mat.tolist()
    
    # Verify schoolbook
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    tensor_from_decomp = np.einsum('ac,bc,dc->abd', U, V, W)
    assert np.allclose(tensor_from_decomp, T), "Schoolbook verification failed"
    
    # Try to find rank 47 decomposition
    # Strategy: try to merge pairs of schoolbook terms
    # For each pair of terms, try to see if they can be combined
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try structured search for rank 47
    if time.time() < deadline - 20:
        target_rank = rank - 1  # 47
        # Try all pairs of schoolbook terms to see if they can be merged
        # This is a heuristic approach
        
        # For a pair of terms (i1,j1,k1) and (i2,j2,k2), 
        # we need to find U', V', W' such that the combined term
        # reproduces both original terms' contributions
        # 
        # Original term 1: U1[i1*m+j1]=1, V1[j1*p+k1]=1, W1[k1*n+i1]=1
        # Original term 2: U2[i2*m+j2]=1, V2[j2*p+k2]=1, W2[k2*n+i2]=1
        #
        # Merged term: U', V', W' such that
        # U'[a]*V'[b]*W'[c] = (1 if (a,b,c) matches term1) + (1 if (a,b,c) matches term2)
        #
        # This requires U'[a]*V'[b]*W'[c] to be 1 at two positions and 0 elsewhere.
        # This is generally impossible with a single rank-1 term unless the two positions
        # are "compatible" in a specific way.
        
        # Let's try a different approach: use linear algebra
        # Fix 46 of the 48 schoolbook terms, and try to fit the remaining 2
        # with a single term.
        
        # For each pair of terms to remove, check if a single rank-1 term can replace them
        # This is O(48^2) = 2304 checks, each involving solving a small system
        
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append((i, j, k))
        
        found = False
        for idx1 in range(len(terms)):
            if time.time() > deadline - 10:
                break
            for idx2 in range(idx1 + 1, len(terms)):
                if time.time() > deadline - 10:
                    break
                
                i1, j1, k1 = terms[idx1]
                i2, j2, k2 = terms[idx2]
                
                # The two positions in the tensor that need to be covered:
                # pos1 = (i1*m+j1, j1*p+k1, k1*n+i1)
                # pos2 = (i2*m+j2, j2*p+k2, k2*n+i2)
                
                # A single rank-1 term U'*V'*W' must satisfy:
                # U'[a1]*V'[b1]*W'[c1] = 1 where (a1,b1,c1) = pos1
                # U'[a2]*V'[b2]*W'[c2] = 1 where (a2,b2,c2) = pos2
                # And U'[a]*V'[b]*W'[c] = 0 for all other (a,b,c)
                
                # This is very restrictive. Let's check if it's possible.
                # For the term to be zero at position (a,b,c), we need at least one of
                # U'[a]=0, V'[b]=0, W'[c]=0.
                
                # The non-zero positions are pos1 and pos2.
                # At pos1: U'[a1]≠0, V'[b1]≠0, W'[c1]≠0
                # At pos2: U'[a2]≠0, V'[b2]≠0, W'[c2]≠0
                
                # For all other positions, at least one component must be zero.
                # This means the support of U' is {a1, a2}, support of V' is {b1, b2},
                # support of W' is {c1, c2}, and the cross terms must cancel or be zero.
                
                # If a1≠a2, b1≠b2, c1≠c2, then:
                # Position (a1, b2, c1): U'[a1]*V'[b2]*W'[c1] must be 0
                # Position (a2, b1, c2): U'[a2]*V'[b1]*W'[c2] must be 0
                # But U'[a1], V'[b2], W'[c1] are all potentially non-zero...
                
                # This is getting complex. Let's just try to solve it numerically.
                a1, b1, c1 = i1*m+j1, j1*p+k1, k1*n+i1
                a2, b2, c2 = i2*m+j2, j2*p+k2, k2*n+i2
                
                # Try to find U', V', W' with minimal support
                # U' non-zero only at a1, a2 (if a1≠a2) or just a1 (if a1=a2)
                # Similarly for V', W'
                
                if a1 == a2 and b1 == b2 and c1 == c2:
                    continue  # Same position, can't happen for distinct terms
                
                # Build the new decomposition
                new_U = U.copy()
                new_V = V.copy()
                new_W = W.copy()
                
                # Remove the two terms
                new_U[:, idx1] = 0
                new_V[:, idx1] = 0
                new_W[:, idx1] = 0
                new_U[:, idx2] = 0
                new_V[:, idx2] = 0
                new_W[:, idx2] = 0
                
                # Try to add a single term at position idx1 that covers both
                # We need to find U'[a], V'[b], W'[c] such that:
                # U'[a1]*V'[b1]*W'[c1] = 1
                # U'[a2]*V'[b2]*W'[c2] = 1
                # And the term doesn't interfere with other positions
                
                # Simple approach: try U'[a1]=1, U'[a2]=1, V'[b1]=1, V'[b2]=1, W'[c1]=1, W'[c2]=1
                # But this creates non-zero entries at (a1,b2,c1), (a2,b1,c2), etc.
                
                # Check if those positions are already covered by other terms
                # If not, this won't work.
                
                # This approach is too crude. Let's just verify the schoolbook and return it.
                pass
        
        # If we found a better decomposition, use it
        if found:
            best_rank = target_rank
            best_U, best_V, best_W = new_U, new_V, new_W
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
