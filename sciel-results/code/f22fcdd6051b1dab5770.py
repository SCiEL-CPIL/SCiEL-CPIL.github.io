"""Tensor decomposition search for 3x3 x 3x4 matrix multiplication."""
import numpy as np
import time
from fractions import Fraction
import itertools

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x4."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = 3, 3, 4
    
    # The multiplication tensor T: R^{n*m} x R^{m*p} x R^{p*n} -> R
    # T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # Otherwise T[a,b,c] = 0
    # 
    # For decomposition: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # 
    # We want to find minimal rank R such that there exist U (nm x R), V (mp x R), W (pn x R)
    # with integer/half-integer entries satisfying the tensor equation.
    
    nm, mp, pn = n*m, m*p, p*n  # 9, 12, 12
    
    # First, construct the schoolbook decomposition as a fallback
    def schoolbook():
        R = n * m * p
        U = np.zeros((nm, R), dtype=np.float64)
        V = np.zeros((mp, R), dtype=np.float64)
        W = np.zeros((pn, R), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W, R
    
    U_sch, V_sch, W_sch, R_sch = schoolbook()
    
    def verify(U, V, W, R):
        """Verify that the decomposition is exact."""
        # Check tensor equation
        max_err = 0.0
        for a in range(nm):
            i = a // m
            for b in range(mp):
                j = b // p
                k_val = b % p
                # Compute expected: sum over i',j',k' where a=i'm+j', b=j'p+k', c=k'n+i
                # T[a,b,c] is 1 only if there exist i',j',k' such that:
                #   a = i'*m + j', b = j'*p + k', c = k'*n + i
                # With a = i*m+j, b = j*p+k, we need i'=i, j'=j, k'=k for the match
                # So T[i*m+j, j*p+k, c] = 1 if c = k*n+i, else 0
                for c in range(pn):
                    expected = 1.0 if c == k_val*n + i else 0.0
                    computed = 0.0
                    for r in range(R):
                        computed += U[a, r] * V[b, r] * W[c, r]
                    err = abs(computed - expected)
                    if err > max_err:
                        max_err = err
                    if max_err > 1e-10:
                        return False
        return True
    
    def try_random_search(R, n_iters, timeout):
        """Try random initialization and local search for rank R."""
        if R >= R_sch:
            return None  # Don't need to search if we already have schoolbook
        
        best_U, best_V, best_W = None, None, None
        
        for trial in range(n_iters):
            if time.time() > deadline:
                break
            
            # Initialize with small random values
            np.random.seed(42 + trial)
            U = np.random.randn(nm, R) * 0.5
            V = np.random.randn(mp, R) * 0.5
            W = np.random.randn(pn, R) * 0.5
            
            # Iterative refinement
            for iteration in range(50):
                if time.time() > deadline:
                    return None
                
                # Fix U, solve for V and W
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed U, this is linear in each pair (V[b,r], W[c,r])
                
                # Residual: res[a,b,c] = T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r]
                # We want to update V and W to reduce residual
                
                # Gradient-based approach
                # dLoss/dV[b,r] = sum_a,c res[a,b,c] * U[a,r] * W[c,r]
                # dLoss/dW[c,r] = sum_a,b res[a,b,c] * U[a,r] * V[b,r]
                
                # Compute residual
                # Tensor product: sum_r U[a,r] V[b,r] W[c,r]
                # This is like a tensor contraction
                
                # Efficient computation using numpy
                # Let's use einsum
                try:
                    # T_comp[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                    T_comp = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                    
                    # Build T
                    T = np.zeros((nm, mp, pn))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i * m + j
                                b = j * p + k
                                c = k * n + i
                                T[a, b, c] = 1.0
                    
                    res = T - T_comp
                    loss = np.sum(res**2)
                    
                    if loss < 1e-12:
                        # Check if entries are close to integers or half-integers
                        all_close = True
                        for mat in [U, V, W]:
                            for val in mat:
                                closest_half = round(val * 2) / 2
                                if abs(val - closest_half) > 1e-8:
                                    all_close = False
                                    break
                            if not all_close:
                                break
                        if all_close:
                            # Snap to half-integers
                            U_snap = np.round(U * 2) / 2
                            V_snap = np.round(V * 2) / 2
                            W_snap = np.round(W * 2) / 2
                            if verify(U_snap, V_snap, W_snap, R):
                                return U_snap, V_snap, W_snap, R
                        
                        return U, V, W, R
                    
                    # Update V: V[b,r] -= lr * sum_{a,c} res[a,b,c] * U[a,r] * W[c,r]
                    lr = 0.05
                    dV = np.einsum('abc,ar,cr->br', res, U, W)
                    dW = np.einsum('abc,ar,br->cr', res, U, V)
                    
                    U -= lr * np.einsum('abc,br,cr->ar', res, V, W)
                    V -= lr * dV
                    W -= lr * dW
                    
                except Exception:
                    return None
            
            # Final check
            T_comp = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            T = np.zeros((nm, mp, pn))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i*m + j, j*p + k, k*n + i] = 1.0
            
            if np.max(np.abs(T - T_comp)) < 1e-10:
                # Try to snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if verify(U_snap, V_snap, W_snap, R):
                    return U_snap, V_snap, W_snap, R
                return U, V, W, R
        
        return None
    
    # Try rank values from low to high
    # Known bounds: rank >= 27 (for 3x3x3), our case is 3x3x3x4
    # Theoretical lower bound for 3x3 x 3x4: at least the rank of 3x3 x 3x3 which is <= 23 (Winograd)
    # Upper bound: 36 (schoolbook) or potentially less
    
    # Try a structured approach: use Winograd-style FMM-3 for the 3x3 part
    # For 3x3 matrix multiplication, Winograd FMM-3 has rank 23
    # For 3x3 x 3x4, we can think of it as: C[i,k] = sum_j A[i,j] B[j,k]
    # Split B into B1 (3x3) and B2 (3x1)
    # C = A*B1 + A*B2
    # Rank(A*B1) <= rank(3x3x3) and rank(A*B2) <= rank(3x3x1) = 9
    # So total rank <= 23 + 9 = 32
    
    # Let me try to construct a rank-32 or lower decomposition
    
    # Actually, let's try a more systematic search
    # Start with rank 30 and go up
    
    best_result = (U_sch, V_sch, W_sch, R_sch)
    
    for R in range(24, min(R_sch + 1, 40)):
        if time.time() > deadline - 30:  # Leave time for fallback
            break
        
        print(f"Trying rank {R}...", flush=True)
        result = try_random_search(R, n_iters=5, timeout=deadline)
        if result is not None:
            U_r, V_r, W_r, R_r = result
            if verify(U_r, V_r, W_r, R_r):
                best_result = (U_r, V_r, W_r, R_r)
                break
        elif time.time() > deadline:
            break
    
    U_best, V_best, W_best, R_best = best_result
    
    # Convert to lists with half-integer values
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                # Snap to nearest half-integer
                snapped = round(val * 2) / 2
                if abs(snapped).val < 1e-10:
                    row_list.append(0)
                elif abs(snapped - round(snapped)) < 1e-10:
                    row_list.append(int(snapped))
                else:
                    # Use Fraction for exact representation
                    f = Fraction(snapped).limit_denominator(100)
                    if f.denominator == 1:
                        row_list.append(f.numerator)
                    else:
                        row_list.append(f.numerator / f.denominator)
            result.append(row_list)
        return result
    
    U_out = to_list(U_best)
    V_out = to_list(V_best)
    W_out = to_list(W_best)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
