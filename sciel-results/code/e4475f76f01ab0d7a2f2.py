"""Tensor decomposition for 2x3 matrix multiplication (A@B, B is 3x10)."""
import time
import numpy as np

def construct():
    _t0 = time.time()
    n, m, p = 2, 3, 10
    
    # Use the middle-product strategy: split by middle index j
    # For each j in {0,1,2}, we have a 2x1x10 sub-problem
    # We can exploit linear dependencies across the 3 middle indices
    
    # Start with a base decomposition and try to reduce rank
    # Known lower bound: ceil(2*3*10 / max(2,3,10)) = ceil(60/10) = 6 (too optimistic)
    # Realistic: use Strassen-like tricks on the 2x3 part
    
    # Try rank 50 first, then 48, 46, etc.
    for target_rank in [50, 48, 46, 44, 42, 40]:
        if time.time() - _t0 > 110:
            break
        
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        col = 0
        # Process in groups of 2 columns to use 2x3x2 rank-7 decomposition
        for t in range(5):
            k0 = 2 * t
            k1 = 2 * t + 1
            
            # Use a more efficient 2x3x2 decomposition (rank 7 instead of 11)
            # Term 1: (A00+A11)*(B0k0+B1k1) -> C0k0, C1k1
            if col < target_rank:
                U[0, col] += 1; U[4, col] += 1
                V[0*10+k0, col] += 1; V[1*10+k1, col] += 1
                W[k0*2+0, col] += 1; W[k1*2+1, col] += 1
                col += 1
            
            # Term 2: (A10+A11)*B0k0 -> C1k0, C1k1
            if col < target_rank:
                U[3, col] += 1; U[4, col] += 1
                V[0*10+k0, col] += 1
                W[k0*2+1, col] += 1; W[k1*2+1, col] -= 1
                col += 1
            
            # Term 3: A00*(B0k1-B1k1) -> C0k1, C1k1
            if col < target_rank:
                U[0, col] += 1
                V[0*10+k1, col] += 1; V[1*10+k1, col] -= 1
                W[k1*2+0, col] += 1; W[k1*2+1, col] += 1
                col += 1
            
            # Term 4: A11*(B1k0-B0k0) -> C0k0, C1k0
            if col < target_rank:
                U[4, col] += 1
                V[1*10+k0, col] += 1; V[0*10+k0, col] -= 1
                W[k0*2+0, col] += 1; W[k0*2+1, col] += 1
                col += 1
            
            # Term 5: (A00+A01)*B1k1 -> C0k0(-1), C0k1
            if col < target_rank:
                U[0, col] += 1; U[1, col] += 1
                V[1*10+k1, col] += 1
                W[k0*2+0, col] -= 1; W[k1*2+0, col] += 1
                col += 1
            
            # Term 6: (A10-A00)*(B0k0+B0k1) -> C1k1
            if col < target_rank:
                U[3, col] += 1; U[0, col] -= 1
                V[0*10+k0, col] += 1; V[0*10+k1, col] += 1
                W[k1*2+1, col] += 1
                col += 1
            
            # Term 7: (A01-A11)*(B1k0+B1k1) -> C0k0
            if col < target_rank:
                U[1, col] += 1; U[4, col] -= 1
                V[1*10+k0, col] += 1; V[1*10+k1, col] += 1
                W[k0*2+0, col] += 1
                col += 1
            
            # Terms 8-11: direct terms for row 2 of B
            if col < target_rank:
                U[2, col] += 1; V[2*10+k0, col] += 1; W[k0*2+0, col] += 1
                col += 1
            if col < target_rank:
                U[2, col] += 1; V[2*10+k1, col] += 1; W[k1*2+0, col] += 1
                col += 1
            if col < target_rank:
                U[5, col] += 1; V[2*10+k0, col] += 1; W[k0*2+1, col] += 1
                col += 1
            if col < target_rank:
                U[5, col] += 1; V[2*10+k1, col] += 1; W[k1*2+1, col] += 1
                col += 1
        
        # Verify the decomposition
        if col < target_rank:
            # Pad remaining columns with zeros (shouldn't happen if we have enough rank)
            pass
        
        # Check if decomposition is exact
        # Build the tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # Should equal A[i,j]*B[j,k] for matrix multiplication
        
        # Quick check: verify a few entries
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,k,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # Should equal 1 if this is the multiplication tensor
                    expected = 1.0
                    actual = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(actual - expected) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Convert to lists for return
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return the original rank-55 solution
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        U[0][col] += 1; U[4][col] += 1
        V[0*10+k0][col] += 1; V[1*10+k1][col] += 1
        W[k0*2+0][col] += 1; W[k1*2+1][col] += 1
        col += 1
        U[3][col] += 1; U[4][col] += 1
        V[0*10+k0][col] += 1
        W[k0*2+1][col] += 1; W[k1*2+1][col] -= 1
        col += 1
        U[0][col] += 1
        V[0*10+k1][col] += 1; V[1*10+k1][col] -= 1
        W[k1*2+0][col] += 1; W[k1*2+1][col] += 1
        col += 1
        U[4][col] += 1
        V[1*10+k0][col] += 1; V[0*10+k0][col] -= 1
        W[k0*2+0][col] += 1; W[k0*2+1][col] += 1
        col += 1
        U[0][col] += 1; U[1][col] += 1
        V[1*10+k1][col] += 1
        W[k0*2+0][col] -= 1; W[k1*2+0][col] += 1
        col += 1
        U[3][col] += 1; U[0][col] -= 1
        V[0*10+k0][col] += 1; V[0*10+k1][col] += 1
        W[k1*2+1][col] += 1
        col += 1
        U[1][col] += 1; U[4][col] -= 1
        V[1*10+k0][col] += 1; V[1*10+k1][col] += 1
        W[k0*2+0][col] += 1
        col += 1
        U[2][col] += 1; V[2*10+k0][col] += 1; W[k0*2+0][col] += 1
        col += 1
        U[2][col] += 1; V[2*10+k1][col] += 1; W[k1*2+0][col] += 1
        col += 1
        U[5][col] += 1; V[2*10+k0][col] += 1; W[k0*2+1][col] += 1
        col += 1
        U[5][col] += 1; V[2*10+k1][col] += 1; W[k1*2+1][col] += 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
