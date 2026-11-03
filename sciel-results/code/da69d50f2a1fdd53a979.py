"""Matrix multiplication 3x3x7 - structured decomposition with algebraic elimination."""
import time
import numpy as np
from itertools import product, combinations

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    Strategy: Use structured factors, then apply algebraic elimination to remove linear dependencies.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 63
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def make_schoolbook():
        rank = schoolbook_rank
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
    
    def verify_decomp_fast(U_arr, V_arr, W_arr, tol=1e-8):
        """
        Verify exact tensor decomposition using efficient tensor contraction.
        U_arr: (n*m, r), V_arr: (m*p, r), W_arr: (p*n, r)
        """
        r = U_arr.shape[1]
        
        # Build the full tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # T should be 1 when a=(i,j), b=(j,k), c=(k,i) and 0 otherwise
        
        # Efficient check: for each r, the rank-1 contribution is outer product
        # Sum over r: U[:,r] * V[:,r] * W[:,r]
        
        # We need T[a,b,c] = delta(a->(i,j), b->(j,k), c->(k,i))
        # where i=a//m, j=a%m, k=b%m, and check j==b//p, i==c%n, k==c//n
        
        # Let's compute all required entries
        # Required non-zero: for each i,j,k: T[i*m+j, j*p+k, k*n+i] = 1
        # All other entries should be 0
        
        # Compute tensor via einsum or manual
        # T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)  # This is (n*m, m*p, p*n)
        # But that's huge: 9*21*21 = 3969 entries, fine
        
        # Actually let's be more careful - we only need specific entries
        # But computing the full tensor is fast enough for small sizes
        
        T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        # Check required non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if abs(T[a, b, c] - 1.0) > tol:
                        return False
        
        # Check that all other entries are zero
        # Create a mask of valid (a,b,c) triples
        # Valid: a=i*m+j, b=j'*p+k, c=k'*n+i' where j==j', i==i', k==k'
        # Invalid: any other combination
        
        # For efficiency, check specific categories of invalid entries:
        # Category 1: a=(i,j), b=(j,k), c=(k,i') with i' != i
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    for i3 in range(n):
                        if i3 == i:
                            continue
                        c = k * n + i3
                        if abs(T[a, b, c]) > tol:
                            return False
        
        # Category 2: a=(i,j), b=(j',k), c=(k,i) with j' != j
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    c = k * n + i
                    for j2 in range(m):
                        if j2 == j:
                            continue
                        b = j2 * p + k
                        if abs(T[a, b, c]) > tol:
                            return False
        
        # Category 3: a=(i,j), b=(j,k'), c=(k,i) with k' != k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    for k2 in range(p):
                        if k2 == k:
                            continue
                        c2 = k2 * n + i
                        if abs(T[a, b, c2]) > tol:
                            return False
        
        return True
    
    def verify_decomp(U, V, W, tol=1e-8):
        """Verify exact tensor decomposition."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        return verify_decomp_fast(U_arr, V_arr, W_arr, tol)
    
    def try_elimination(U, V, W, max_iterations=100):
        """
        Try to reduce rank by finding linear dependencies among the rank-1 terms.
        If term c is a linear combination of others, we can eliminate it.
        """
        U_arr = np.array(U, dtype=float).copy()
        V_arr = np.array(V, dtype=float).copy()
        W_arr = np.array(W, dtype=float).copy()
        r = U_arr.shape[1]
        
        best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        best_rank = r
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try to eliminate each term
            eliminated = False
            for c in range(r):
                if time.time() > deadline:
                    break
                
                # Check if term c is redundant
                # Remove term c and see if we can still verify
                U_test = np.delete(U_arr, c, axis=1)
                V_test = np.delete(V_arr, c, axis=1)
                W_test = np.delete(W_arr, c, axis=1)
                
                if verify_decomp_fast(U_test, V_test, W_test):
                    # Term c is redundant, eliminate it
                    U_arr = U_test
                    V_arr = V_test
                    W_arr = W_test
                    r -= 1
                    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
                    best_rank = r
                    eliminated = True
                    break
            
            if not eliminated:
                break
        
        return best_U, best_V, best_W, best_rank
    
    def try_pairwise_elimination(U_arr, V_arr, W_arr, max_iterations=50):
        """
        Try to eliminate pairs of terms simultaneously.
        """
        r = U_arr.shape[1]
        best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        best_rank = r
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            eliminated = False
            for c1 in range(r):
                if time.time() > deadline:
                    break
                for c2 in range(c1 + 1, r):
                    if time.time() > deadline:
                        break
                    
                    # Remove terms c1 and c2
                    U_test = np.delete(U_arr, [c1, c2], axis=1)
                    V_test = np.delete(V_arr, [c1, c2], axis=1)
                    W_test = np.delete(W_arr, [c1, c2], axis=1)
                    
                    if verify_decomp_fast(U_test, V_test, W_test):
                        U_arr = U_test
                        V_arr = V_test
                        W_arr = W_test
                        r -= 2
                        best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
                        best_rank = r
                        eliminated = True
                        break
                if eliminated:
                    break
            
            if not eliminated:
                break
        
        return best_U, best_V, best_W, best_rank
    
    # Build schoolbook decomposition
    result = make_schoolbook()
    U_arr = np.array(result["U"], dtype=float)
    V_arr = np.array(result["V"], dtype=float)
    W_arr = np.array(result["W"], dtype=float)
    
    # Try single-term elimination
    best_U, best_V, best_W, best_rank = try_elimination(U_arr, V_arr, W_arr)
    
    # Try pairwise elimination on the result
    if best_rank < schoolbook_rank:
        best_U, best_V, best_W, best_rank2 = try_pairwise_elimination(best_U, best_V, best_W)
        if best_rank2 < best_rank:
            best_rank = best_rank2
    
    if best_rank < schoolbook_rank:
        result = {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Final verification
    if verify_decomp(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook
    return make_schoolbook()
# EVOLVE-BLOCK-END
