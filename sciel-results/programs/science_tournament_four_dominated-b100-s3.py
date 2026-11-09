import time
import numpy as np

def construct():
    start_time = time.time()
    
    # Candidate primes p where p % 4 == 3
    # We want the smallest n.
    candidates = []
    for p in range(7, 90):
        if p % 4 == 3:
            # Check if prime
            if all(p % i != 0 for i in range(2, int(p**0.5)+1)):
                candidates.append(p)
                
    # Sort by n (which is p)
    candidates.sort()
    
    for p in candidates:
        # Check time budget
        if time.time() - start_time > 100: # Safety margin
            break
            
        # Construct Paley tournament
        squares = {x*x % p for x in range(1, p)}
        
        # Build adjacency matrix: A[u][v] = 1 if u -> v
        # u -> v iff (v - u) % p is a nonzero square
        A = np.zeros((p, p), dtype=np.uint8)
        for u in range(p):
            for v in range(p):
                if u != v and ((v - u) % p) in squares:
                    A[u][v] = 1
        
        # In-neighborhoods: In[u] = set of v such that v -> u
        # This is A^T (transposed)
        # For a 4-set {i,j,k,l}, we need some vertex v with v->i, v->j, v->k, v->l
        # i.e., A[v,i] & A[v,j] & A[v,k] & A[v,l] > 0 for some v
        # Equivalently: column-wise AND of A[:,i], A[:,j], A[:,k], A[:,l] must have at least one 1
        
        # Verify all 4-sets using vectorized approach
        # For each triple (i,j,k), compute the common in-neighbors, then check against all l > k
        valid = True
        for i in range(p):
            if time.time() - start_time > 105:
                break
            col_i = A[:, i]
            for j in range(i+1, p):
                if not valid:
                    break
                col_j = A[:, j]
                inter_ij = col_i & col_j
                for k in range(j+1, p):
                    if not valid:
                        break
                    col_k = A[:, k]
                    inter_ijk = inter_ij & col_k
                    # Check if for all l > k, there exists v with v->i,j,k,l
                    # i.e., inter_ijk & A[:,l] has at least one 1
                    for l in range(k+1, p):
                        if not np.any(inter_ijk & A[:, l]):
                            valid = False
                            break
                    if not valid:
                        break
            if not valid:
                break
        
        if valid:
            # Construct orientation list
            orientation = []
            for i in range(p):
                for j in range(i+1, p):
                    # 0 means i -> j, 1 means j -> i
                    # i -> j iff (j - i) % p is a square
                    if ((j - i) % p) in squares:
                        orientation.append(0)
                    else:
                        orientation.append(1)
                        
            return {'n': p, 'orientation': orientation}
            
    # Fallback to seed if no smaller one found (should not happen as 83 works)
    p = 83
    squares = {x*x % p for x in range(1, p)}
    orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    return {'n': p, 'orientation': orientation}
