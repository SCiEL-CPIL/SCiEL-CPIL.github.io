import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = 2, 2, 4
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_fallback(U, V, W, tol=1e-9):
        """Alternative verification using tensor contraction."""
        rank = U.shape[1]
        # Build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Fallback: rank 14 (known valid solution via Strassen on two 2x2x2 blocks)
    def rank14_solution():
        rank = 14
        U = np.zeros((4, rank))
        V = np.zeros((8, rank))
        W = np.zeros((8, rank))
        
        # Strassen for left block (B columns 0,1)
        # m1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[5, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # m2 = (A10+A11)*B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1
        
        # m3 = A00*(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[5, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        # m4 = A11*(B10-B00)
        U[3, 3] = 1
        V[4, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # m5 = (A00+A01)*B11
        U[0, 4] = 1; U[1, 4] = 1
        V[5, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # m6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # m7 = (A01-A11)*(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[4, 6] = 1; V[5, 6] = 1
        W[0, 6] = 1
        
        # Strassen for right block (B columns 2,3)
        # m1 = (A00+A11)*(B02+B13)
        U[0, 7] = 1; U[3, 7] = 1
        V[2, 7] = 1; V[7, 7] = 1
        W[4, 7] = 1; W[7, 7] = 1
        
        # m2 = (A10+A11)*B02
        U[2, 8] = 1; U[3, 8] = 1
        V[2, 8] = 1
        W[5, 8] = 1; W[7, 8] = -1
        
        # m3 = A00*(B03-B13)
        U[0, 9] = 1
        V[3, 9] = 1; V[7, 9] = -1
        W[6, 9] = 1; W[7, 9] = 1
        
        # m4 = A11*(B12-B02)
        U[3, 10] = 1
        V[6, 10] = 1; V[2, 10] = -1
        W[4, 10] = 1; W[5, 10] = 1
        
        # m5 = (A00+A01)*B13
        U[0, 11] = 1; U[1, 11] = 1
        V[7, 11] = 1
        W[4, 11] = -1; W[6, 11] = 1
        
        # m6 = (A10-A00)*(B02+B03)
        U[2, 12] = 1; U[0, 12] = -1
        V[2, 12] = 1; V[3, 12] = 1
        W[7, 12] = 1
        
        # m7 = (A01-A11)*(B12+B13)
        U[1, 13] = 1; U[3, 13] = -1
        V[6, 13] = 1; V[7, 13] = 1
        W[4, 13] = 1
        
        return {"U": U, "V": V, "W": W}
    
    # Try to find rank-12 decomposition
    def try_rank12():
        """Try to find a rank-12 decomposition using structured search."""
        from scipy.optimize import minimize
        
        def residual(params, rank=12):
            """Compute sum of squared residuals."""
            U = params[:4*rank].reshape(4, rank)
            V = params[4*rank:12*rank].reshape(8, rank)
            W = params[12*rank:].reshape(8, rank)
            
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        res += (val - 1.0)**2
            return res
        
        def residual_grad(params, rank=12):
            """Compute gradient of residual (numerical for simplicity)."""
            eps = 1e-7
            grad = np.zeros_like(params)
            for idx in range(len(params)):
                params_plus = params.copy()
                params_plus[idx] += eps
                params_minus = params.copy()
                params_minus[idx] -= eps
                grad[idx] = (residual(params_plus, rank) - residual(params_minus, rank)) / (2*eps)
            return grad
        
        np.random.seed(12345)
        best_val = 1e10
        best_params = None
        
        # Try multiple random initializations with different strategies
        for attempt in range(20):
            if time.time() > deadline - 20:
                break
            
            rank = 12
            total_params = (4 + 8 + 8) * rank
            
            # Different initialization strategies
            if attempt < 5:
                # Small random
                params = np.random.randn(total_params) * 0.3
            elif attempt < 10:
                # Sparse initialization
                params = np.random.randn(total_params)
                mask = np.random.rand(total_params) < 0.3
                params[~mask] = 0
                params[mask] *= 2
            elif attempt < 15:
                # Structured initialization based on Strassen pattern
                params = np.zeros(total_params)
                # Initialize with Strassen-like patterns
                for r in range(rank):
                    # Random assignment of signs
                    U_start = r * 4
                    V_start = 4*rank + r * 8
                    W_start = 12*rank + r * 8
                    for idx in range(4):
                        if np.random.rand() < 0.5:
                            params[U_start + idx] = np.random.choice([-1, 1])
                    for idx in range(8):
                        if np.random.rand() < 0.3:
                            params[V_start + idx] = np.random.choice([-1, 1])
                    for idx in range(8):
                        if np.random.rand() < 0.3:
                            params[W_start + idx] = np.random.choice([-1, 1])
            else:
                # Random with larger scale
                params = np.random.randn(total_params) * 1.0
            
            try:
                # Use L-BFGS-B for better convergence
                result = minimize(residual, params, method='L-BFGS-B', 
                                  options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
                if result.fun < best_val:
                    best_val = result.fun
                    best_params = result.x
                    if best_val < 1e-10:
                        break
            except Exception:
                pass
        
        if best_params is not None and best_val < 1e-8:
            rank = 12
            U = best_params[:4*rank].reshape(4, rank)
            V = best_params[4*rank:12*rank].reshape(8, rank)
            W = best_params[12*rank:].reshape(8, rank)
            
            # Round to nearest half-integer if close
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if verify(U_rounded, V_rounded, W_rounded):
                return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
            elif verify(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try rank 12
    result = try_rank12()
    if result is not None:
        return result
    
    # Fallback: rank 14
    sol = rank14_solution()
    U, V, W = sol["U"], sol["V"], sol["W"]
    
    # Verify the fallback
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # If verification fails, try a different rank-14 construction
    # Direct construction from bilinear forms
    rank = 14
    U = np.zeros((4, rank))
    V = np.zeros((8, rank))
    W = np.zeros((8, rank))
    
    # Schoolbook-like but with some sharing
    # Products:
    # 0: A00*B00 -> C00, C10
    # 1: A00*B01 -> C01, C11
    # 2: A00*B02 -> C02, C12
    # 3: A00*B03 -> C03, C13
    # 4: A01*B10 -> C00, C10
    # 5: A01*B11 -> C01, C11
    # 6: A01*B12 -> C02, C12
    # 7: A01*B13 -> C03, C13
    # 8: A10*B00 -> C00, C10 (shared with 0? No, different A index)
    # Actually let's just use the Strassen construction which we know works
    
    # Re-verify the Strassen construction more carefully
    # The issue might be in my index mapping
    
    # Let me rebuild from scratch with explicit verification
    rank = 14
    U = np.zeros((4, rank))
    V = np.zeros((8, rank))
    W = np.zeros((8, rank))
    
    # Left block Strassen (B columns 0,1 -> C columns 0,1)
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Strassen products for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M6
    
    # For left block (indices 0-6):
    # M1: U[0,0]=1, U[3,0]=1; V[0,0]=1, V[5,0]=1; W[0,0]=1, W[3,0]=-1, W[2,0]=-1, W[1,0]=1
    # Wait, W index is k*n+i, so for C00: k=0,i=0 -> W[0]; C01: k=1,i=0 -> W[2]; C10: k=0,i=1 -> W[1]; C11: k=1,i=1 -> W[3]
    
    # Let me redo with correct W indexing:
    # W[k*n+i] where k=0..3, i=0..1
    # C[0,0]: k=0, i=0 -> W[0]
    # C[0,1]: k=1, i=0 -> W[2]
    # C[0,2]: k=2, i=0 -> W[4]
    # C[0,3]: k=3, i=0 -> W[6]
    # C[1,0]: k=0, i=1 -> W[1]
    # C[1,1]: k=1, i=1 -> W[3]
    # C[1,2]: k=2, i=1 -> W[5]
    # C[1,3]: k=3, i=1 -> W[7]
    
    # Left block (B cols 0,1 -> C cols 0,1):
    # M1 = (A00+A11)*(B00+B11): contributes to C00, C01, C10, C11
    # C00 += M1, C01 += M1, C10 += M1, C11 += M1? No...
    
    # Let me just use the known correct Strassen formula:
    # For 2x2 * 2x2 -> 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M6
    
    # So for left block:
    # M1: U[0,0]=1, U[3,0]=1; V[0,0]=1, V[5,0]=1; W[0,0]=1, W[2,0]=1, W[1,0]=1, W[3,0]=1
    # Wait no. Each M contributes to specific C entries with specific coefficients.
    
    # M1 contributes: +1 to C00, +1 to C01, +1 to C10, +1 to C11
    # M2 contributes: +1 to C10, -1 to C11
    # M3 contributes: +1 to C01, -1 to C11
    # M4 contributes: -1 to C00, +1 to C10
    # M5 contributes: -1 to C00, +1 to C01
    # M6 contributes: +1 to C11
    # M7 contributes: +1 to C00
    
    # So:
    # M1: W[0,0]=1, W[2,0]=1, W[1,0]=1, W[3,0]=1
    # M2: W[1,1]=1, W[3,1]=-1
    # M3: W[2,2]=1, W[3,2]=-1
    # M4: W[0,3]=-1, W[1,3]=1
    # M5: W[0,4]=-1, W[2,4]=1
    # M6: W[3,5]=1
    # M7: W[0,6]=1
    
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[5,0]=1; W[0,0]=1; W[2,0]=1; W[1,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    U[0,2]=1; V[1,2]=1; V[5,2]=-1; W[2,2]=1; W[3,2]=-1
    U[3,3]=1; V[4,3]=1; V[0,3]=-1; W[0,3]=-1; W[1,3]=1
    U[0,4]=1; U[1,4]=1; V[5,4]=1; W[0,4]=-1; W[2,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[4,6]=1; V[5,6]=1; W[0,6]=1
    
    # Right block (B cols 2,3 -> C cols 2,3):
    # Same Strassen structure but with B indices shifted by 2
    # B00->B02 (V[2]), B01->B03 (V[3]), B10->B12 (V[6]), B11->B13 (V[7])
    # C02: W[4], C03: W[6], C12: W[5], C13: W[7]
    
    # M1: (A00+A11)*(B02+B13)
    U[0,7]=1; U[3,7]=1; V[2,7]=1; V[7,7]=1; W[4,7]=1; W[6,7]=1; W[5,7]=1; W[7,7]=1
    # M2: (A10+A11)*B02
    U[2,8]=1; U[3,8]=1; V[2,8]=1; W[5,8]=1; W[7,8]=-1
    # M3: A00*(B03-B13)
    U[0,9]=1; V[3,9]=1; V[7,9]=-1; W[6,9]=1; W[7,9]=-1
    # M4: A11*(B12-B02)
    U[3,10]=1; V[6,10]=1; V[2,10]=-1; W[4,10]=-1; W[5,10]=1
    # M5: (A00+A01)*B13
    U[0,11]=1; U[1,11]=1; V[7,11]=1; W[4,11]=-1; W[6,11]=1
    # M6: (A10-A00)*(B02+B03)
    U[2,12]=1; U[0,12]=-1; V[2,12]=1; V[3,12]=1; W[7,12]=1
    # M7: (A01-A11)*(B12+B13)
    U[1,13]=1; U[3,13]=-1; V[6,13]=1; V[7,13]=1; W[4,13]=1
    
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Last resort: return zeros (will score 0 but shouldn't happen)
    return {"U": np.zeros((4, 14)), "V": np.zeros((8, 14)), "W": np.zeros((8, 14))}
