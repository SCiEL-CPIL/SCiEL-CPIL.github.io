import numpy as np
import time
from fractions import Fraction

def _verify_decomp(U, V, W, n, m, p, trials=10, seed=42):
    """Verify the bilinear identity for the flattened index convention."""
    U_t = np.array(U, dtype=np.float64).reshape(n, m, -1)
    V_t = np.array(V, dtype=np.float64).reshape(m, p, -1)
    W_t = np.array(W, dtype=np.float64).reshape(p, n, -1)
    try:
        result = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    except Exception:
        return False
    return np.allclose(result, 1.0, atol=1e-9)

def _round_to_halfint(arr, tol=1e-6):
    """Round array to nearest half-integer."""
    rounded = np.round(arr * 2) / 2.0
    return rounded

def _try_round_and_verify(U, V, W, n, m, p):
    """Round to half-integers and verify. Returns (U_r, V_r, W_r) or None."""
    U_r = _round_to_halfint(U)
    V_r = _round_to_halfint(V)
    W_r = _round_to_halfint(W)
    if _verify_decomp(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
        return U_r, V_r, W_r
    return None

def _als_tensor_rank(target_tensor, n, m, p, rank, max_iter=200, tol=1e-10):
    """
    Alternating Least Squares for tensor rank decomposition.
    target_tensor: shape (n, m, p) with entries 0 or 1 (the multiplication tensor)
    Returns U (n*m, rank), V (m*p, rank), W (p*n, rank) or None.
    """
    rng = np.random.RandomState(0)
    
    # Initialize randomly
    U = rng.randn(n, m, rank)
    V = rng.randn(m, p, rank)
    W = rng.randn(p, n, rank)
    
    for it in range(max_iter):
        # Update U: for each (i,j), solve linear system
        # target[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # For fixed i,j: target[i,j,:] = U[i,j,:] @ (V[j,:,r] * W[:,i,r])^T ... 
        # Actually: target[i,j,k] = sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r])
        # Let M[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Fix j: for each i, k: target[i,j,k] = sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r])
        # This is: for fixed j, the matrix target[:,j,:] (shape n x p) 
        # = sum_r U[:,j,r] outer (V[j,:,r] * W[:,j... wait indices are tricky.
        
        # Let me re-derive. The constraint is:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # where the target is 1 everywhere (since it's the multiplication tensor for this specific case).
        
        # Actually for 2x2 by 2x11, the tensor T[i,j,k] = 1 for all i,j,k? 
        # No! The multiplication tensor has T[i,j,k] = 1 if and only if the term 
        # A[i,j]*B[j,k] contributes to C[i,k]. In standard matrix mult, every 
        # (i,j,k) triple has T[i,j,k]=1 because C[i,k] = sum_j A[i,j]B[j,k].
        # So yes, T[i,j,k] = 1 for all i in [0,2), j in [0,2), k in [0,11).
        
        # So we need: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
        
        # ALS: Fix V,W. For each (i,j), we need:
        # sum_r U[i,j,r] * (V[j,:,r] * W[:,i,r])[k] = 1 for all k
        # Let A_ij[k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is a linear system in U[i,j,:]:
        # Let B[j,k,r] = V[j,k,r] * W[k,i,r] for fixed i,j
        # Then sum_r U[i,j,r] * B[j,k,r] = 1 for all k
        # This is: U[i,j,:] @ B^T = ones(p)
        # B has shape (p, rank), so B^T has shape (rank, p)
        # We want U[i,j,:] (shape rank) such that U[i,j,:] @ B^T = ones(p)
        # i.e., B @ U[i,j,:]^T = ones(p)  ... wait
        # sum_r U[i,j,r] * B[j,k,r] = 1
        # Let u = U[i,j,:] (vector of length rank)
        # Then (B^T @ u)[k] = sum_r B[j,k,r] * u[r] = 1
        # So B^T @ u = ones(p)
        # B^T has shape (p, rank), u has shape (rank,)
        # Solve: B^T @ u = ones(p)
        
        for i in range(n):
            for j in range(m):
                # Build B^T: shape (p, rank)
                # B^T[k, r] = V[j, k, r] * W[k, i, r]
                BT = V[j, :, :] * W[:, i, :]  # shape (p, rank)
                try:
                    # Solve BT @ u = ones(p)
                    u, residuals, rank_s, sv = np.linalg.lstsq(BT, np.ones(p), rcond=None)
                    U[i, j, :] = u
                except np.linalg.LinAlgError:
                    pass
        
        # Update V: Fix U,W. For each (j,k):
        # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i
        # Let C[i, r] = U[i,j,r] * W[k,i,r] for fixed j,k
        # Then sum_r V[j,k,r] * C[i,r] = 1 for all i
        # C^T @ v = ones(n) where v = V[j,k,:]
        for j in range(m):
            for k in range(p):
                # C[i, r] = U[i, j, r] * W[k, i, r]
                C = U[:, j, :] * W[k, :, :]  # shape (n, rank)
                try:
                    v, residuals, rank_s, sv = np.linalg.lstsq(C.T, np.ones(n), rcond=None)
                    V[j, k, :] = v
                except np.linalg.LinAlgError:
                    pass
        
        # Update W: Fix U,V. For each (k,i):
        # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all j
        # Let D[j, r] = U[i,j,r] * V[j,k,r] for fixed k,i
        # Then sum_r W[k,i,r] * D[j,r] = 1 for all j
        # D^T @ w = ones(m) where w = W[k,i,:]
        for k in range(p):
            for i in range(n):
                # D[j, r] = U[i, j, r] * V[j, k, r]
                D = U[i, :, :] * V[:, k, :]  # shape (m, rank)
                try:
                    w, residuals, rank_s, sv = np.linalg.lstsq(D.T, np.ones(m), rcond=None)
                    W[k, i, :] = w
                except np.linalg.LinAlgError:
                    pass
        
        # Check convergence
        # Compute max error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i, j, :] * V[j, k, :] * W[k, i, :])
                    err = max(err, abs(val - 1.0))
        if err < tol:
            break
    
    # Flatten and check
    U_flat = U.reshape(n * m, rank)
    V_flat = V.reshape(m * p, rank)
    W_flat = W.reshape(p * n, rank)
    
    if _verify_decomp(U_flat.tolist(), V_flat.tolist(), W_flat.tolist(), n, m, p):
        return U_flat, V_flat, W_flat
    return None

def _construct_strassen_block(n, m, p):
    """
    Construct a decomposition using Strassen for the 2x2 part.
    R(2,2,2) = 7 (Strassen). For 2x2 by 2x11, we can use Strassen on the 2x2 block
    and schoolbook for the rest, or use a hybrid.
    
    Actually, let's think about this differently.
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    Strassen for 2x2 * 2x2 uses 7 multiplications:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1+M4-M5+M7
    C01 = M3+M5
    C10 = M2+M4
    C11 = M1-M2+M3+M6
    
    For 2x2 * 2x11, we can apply Strassen-like decomposition column by column?
    No, Strassen works on the full 2x2 * 2x2. For 2x2 * 2x11, we need a different approach.
    
    Let's use the following: view B as having 11 columns. We can group them.
    Actually, the simplest improvement over schoolbook (rank 44) is to use the fact that
    for each pair of columns of B, we can use a 2x2*2x2 Strassen decomposition.
    
    If we group the 11 columns into 5 pairs + 1 single:
    - 5 groups of 2 columns: each uses rank 7 (Strassen for 2x2*2x2)
    - 1 group of 1 column: uses rank 4 (schoolbook for 2x2*2x1)
    Total: 5*7 + 4 = 39
    
    Let's construct this explicitly.
    """
    # We'll build U, V, W for rank 39
    # Group columns of B: (0,1), (2,3), (4,5), (6,7), (8,9), (10)
    # For each pair (2c, 2c+1), use Strassen decomposition
    # For column 10, use schoolbook
    
    rank = 39
    U = np.zeros((n * m, rank))  # (4, 39)
    V = np.zeros((m * p, rank))  # (22, 39)
    W = np.zeros((p * n, rank))  # (22, 39)
    
    col = 0
    for pair_idx in range(5):
        k0 = 2 * pair_idx
        k1 = 2 * pair_idx + 1
        # Strassen for 2x2 * 2x2 with columns k0, k1
        # M1 = (A00+A11)(B00+B11) -> here B00=B[0,k0], B11=B[1,k1]
        # M2 = (A10+A11)B00 -> B[0,k0]
        # M3 = A00(B01-B11) -> B[0,k1]-B[1,k1]
        # M4 = A11(B10-B00) -> B[1,k0]-B[0,k0]
        # M5 = (A00+A01)B11 -> B[1,k1]
        # M6 = (A10-A00)(B00+B10) -> B[0,k0]+B[1,k0]
        # M7 = (A01-A11)(B01+B11) -> B[0,k1]+B[1,k1]
        
        # U factors (for A):
        # M1: A00+A11 -> U[0,0]=1, U[1,1]=1 (i.e., U[0*m+0]=1, U[1*m+1]=1)
        # M2: A10+A11 -> U[1*m+0]=1, U[1*m+1]=1
        # M3: A00 -> U[0*m+0]=1
        # M4: A11 -> U[1*m+1]=1
        # M5: A00+A01 -> U[0*m+0]=1, U[0*m+1]=1
        # M6: A10-A00 -> U[1*m+0]=1, U[0*m+0]=-1
        # M7: A01-A11 -> U[0*m+1]=1, U[1*m+1]=-1
        
        # V factors (for B):
        # M1: B00+B11 -> V[0*p+k0]=1, V[1*p+k1]=1
        # M2: B00 -> V[0*p+k0]=1
        # M3: B01-B11 -> V[0*p+k1]=1, V[1*p+k1]=-1
        # M4: B10-B00 -> V[1*p+k0]=1, V[0*p+k0]=-1
        # M5: B11 -> V[1*p+k1]=1
        # M6: B00+B10 -> V[0*p+k0]=1, V[1*p+k0]=1
        # M7: B01+B11 -> V[0*p+k1]=1, V[1*p+k1]=1
        
        # W factors (for C):
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # W index: k*n + i
        # C00: W[k0*n+0] += 1, W[k1*n+0] += 1 (from M1, M4, -M5, M7)
        # Wait, let me be more careful.
        # M1 contributes to C00 and C11
        # M2 contributes to C10 and C11
        # M3 contributes to C01 and C11
        # M4 contributes to C00 and C10
        # M5 contributes to C00 and C01
        # M6 contributes to C11
        # M7 contributes to C00
        
        # For each Strassen multiplication r in 0..6:
        #   The output C[i,k] gets contributions from the relevant M's.
        #   W[k*n+i, col] is the coefficient.
        
        # M1 (col): C00 += 1, C11 += 1
        #   W[k0*n+0, col] = 1, W[k1*n+1, col] = 1
        # M2 (col+1): C10 += 1, C11 -= 1
        #   W[k0*n+1, col+1] = 1, W[k1*n+1, col+1] = -1
        # M3 (col+2): C01 += 1, C11 += 1
        #   W[k1*n+0, col+2] = 1, W[k1*n+1, col+2] = 1
        # M4 (col+3): C00 += 1, C10 += 1
        #   W[k0*n+0, col+3] = 1, W[k0*n+1, col+3] = 1
        # M5 (col+4): C00 -= 1, C01 += 1
        #   W[k0*n+0, col+4] = -1, W[k1*n+0, col+4] = 1
        # M6 (col+5): C11 += 1
        #   W[k1*n+1, col+5] = 1
        # M7 (col+6): C00 += 1
        #   W[k0*n+0, col+6] = 1
        
        # Let me re-verify Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So:
        # M1: C00+=1, C11+=1
        # M2: C10+=1, C11-=1
        # M3: C01+=1, C11+=1
        # M4: C00+=1, C10+=1
        # M5: C00-=1, C01+=1
        # M6: C11+=1
        # M7: C00+=1
        
        U[col, 0*m+0] = 1;  U[col, 1*m+1] = 1       # M1: A00+A11
        V[col, 0*p+k0] = 1;  V[col, 1*p+k1] = 1       # M1: B00+B11
        W[col, k0*n+0] = 1;  W[col, k1*n+1] = 1       # M1: C00+=1, C11+=1
        
        U[col+1, 1*m+0] = 1; U[col+1, 1*m+1] = 1     # M2: A10+A11
        V[col+1, 0*p+k0] = 1                          # M2: B00
        W[col+1, k0*n+1] = 1; W[col+1, k1*n+1] = -1  # M2: C10+=1, C11-=1
        
        U[col+2, 0*m+0] = 1                           # M3: A00
        V[col+2, 0*p+k1] = 1; V[col+2, 1*p+k1] = -1  # M3: B01-B11
        W[col+2, k1*n+0] = 1; W[col+2, k1*n+1] = 1   # M3: C01+=1, C11+=1
        
        U[col+3, 1*m+1] = 1                           # M4: A11
        V[col+3, 1*p+k0] = 1; V[col+3, 0*p+k0] = -1  # M4: B10-B00
        W[col+3, k0*n+0] = 1; W[col+3, k0*n+1] = 1   # M4: C00+=1, C10+=1
        
        U[col+4, 0*m+0] = 1; U[col+4, 0*m+1] = 1     # M5: A00+A01
        V[col+4, 1*p+k1] = 1                          # M5: B11
        W[col+4, k0*n+0] = -1; W[col+4, k1*n+0] = 1  # M5: C00-=1, C01+=1
        
        U[col+5, 1*m+0] = 1; U[col+5, 0*m+0] = -1    # M6: A10-A00
        V[col+5, 0*p+k0] = 1; V[col+5, 1*p+k0] = 1   # M6: B00+B10
        W[col+5, k1*n+1] = 1                          # M6: C11+=1
        
        U[col+6, 0*m+1] = 1; U[col+6, 1*m+1] = -1    # M7: A01-A11
        V[col+6, 0*p+k1] = 1; V[col+6, 1*p+k1] = 1   # M7: B01+B11
        W[col+6, k0*n+0] = 1                          # M7: C00+=1
        
        col += 7
    
    # Last column (k=10): schoolbook for 2x2 * 2x1
    # C[i,10] = A[i,0]*B[0,10] + A[i,1]*B[1,10]
    # 4 terms: (i,j) for i in {0,1}, j in {0,1}
    for i in range(2):
        for j in range(2):
            U[col, i*m+j] = 1
            V[col, j*p+10] = 1
            W[col, 10*n+i] = 1
            col += 1
    
    # Verify
    if _verify_decomp(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return U, V, W
    return None

def _try_strassen_hybrid(n, m, p):
    """Try various Strassen hybrid constructions."""
    # Standard: 5 pairs + 1 single = 39
    result = _construct_strassen_block(n, m, p)
    if result is not None:
        return result
    return None

def _try_fft_construction(n, m, p):
    """
    Try FFT-based construction.
    View the multiplication as polynomial multiplication with structure.
    For 2x2 * 2x11, we can potentially use a smaller FFT.
    """
    # This is complex; skip for now and rely on Strassen hybrid
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s kill
    
    n, m, p = 2, 2, 11
    
    # First, try the Strassen hybrid construction (rank 39)
    result = _try_strassen_hybrid(n, m, p)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If Strassen hybrid failed, try ALS optimization
    if time.time() < deadline - 30:
        for rank in [37, 38, 39, 40, 41, 42, 43, 44]:
            if time.time() > deadline - 20:
                break
            # Try multiple random initializations
            for trial in range(20):
                if time.time() > deadline - 20:
                    break
                np.random.seed(trial * 1000 + rank)
                U, V, W = _als_tensor_rank(None, n, m, p, rank, max_iter=100, tol=1e-10)
                if U is not None:
                    # Try rounding to half-integers
                    rounded = _try_round_and_verify(U, V, W, n, m, p)
                    if rounded is not None:
                        Ur, Vr, Wr = rounded
                        return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
                    # If not roundable but valid as float, check if entries are already half-integers
                    if _verify_decomp(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                        # Check if all entries are half-integers
                        all_half = all(np.allclose(x, np.round(x*2)/2, atol=1e-8) 
                                       for x in [U, V, W])
                        if all_half:
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook (rank 44)
    rank = 44
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))

