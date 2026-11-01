"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]
DEADLINE = 140.0  # seconds (safety margin under 150 s wall-clock)

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the n x m by m x p multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: W[k*n + i, c]
    C[i, k] = sum_j A[i,j]*B[j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    So for each c: C += W_c^T outer U_c^T ... wait.
    Actually: C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) ... no.
    
    The tensor T[i,j,k] = delta(i,k via j) ... T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k].
    T[i,j,k] = 1 for all i,j,k (it's the identity in the middle).
    Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Let's verify by building the full tensor and comparing.
    """
    rank = U.shape[1]
    # Build LHS: for each c, outer product of reshaped U_c, V_c, W_c
    # U_c has shape (n, m) when reshaped
    # V_c has shape (m, p)
    # W_c has shape (p, n)
    # T_reconstructed[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
    T_rec = np.zeros((n, m, p))
    for c in range(rank):
        Uc = U[:, c].reshape(n, m)
        Vc = V[:, c].reshape(m, p)
        Wc = W[:, c].reshape(p, n)
        # T[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
        # For each k: T[:, :, k] += Uc * (Vc * Wc[:, k, None])
        for k in range(p):
            T_rec[:, :, k] += Uc * (Vc * Wc[k, :].reshape(1, -1) if Wc[k, 0] != 0 else 0)
            # Wait, Wc is (p, n), so Wc[k, i] for fixed k.
            # Vc[j,k] * Wc[k,i] for fixed k gives a vector over j? No.
            # Let me redo: for fixed k, T[:, :, k] += sum_i? No.
            # T[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
            # For fixed k: T[:, :, k][i,j] += Uc[i,j] * Vc[j,k] * Wc[k,i]
            # = Uc * (Wc[k,:] * Vc[:,k])  -- elementwise
            Wck = Wc[k, :]  # shape (n,)
            Vck = Vc[:, k]  # shape (m,)
            T_rec[:, :, k] += Uc * np.outer(Wck, Vck)
    
    # True tensor: T[i,j,k] = 1 for all i,j,k
    T_true = np.ones((n, m, p))
    return np.allclose(T_rec, T_true, atol=tol)


def _build_strassen_block(A_blocks, B_blocks, C_block_indices, n, m, p):
    """
    Build Strassen 7-multiplication decomposition for a 2x2 block structure.
    A is 2x2 blocks of size (1x2) x (2x2)? 
    Actually for 4x4 x 4x4 with 2x2 blocks of 2x2:
    A = [[A11, A12], [A21, A22]] each 2x2
    B = [[B11, B12], [B21, B22]] each 2x2
    C = [[C11, C12], [C21, C22]] each 2x2
    
    Strassen:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22) B11
    M3 = A11 (B12 - B22)
    M4 = A22 (B21 - B11)
    M5 = (A11 + A12) B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    pass  # We'll build the full U, V, W directly


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    start_time = time.time()
    
    # Strategy: Build rank-28 decomposition using 4 independent Strassen 7-mult blocks
    # But we only need the top 2 rows of the 4x4 result.
    # Actually, let's just build the full rank-32 schoolbook first as fallback,
    # then try to find a lower-rank decomposition.
    
    # First, try the direct approach: build a rank-r decomposition by solving
    # a linear system. We'll try ranks from 27 down to 14.
    
    # The tensor T has dimensions (n*m) x (m*p) x (p*n) = 8 x 16 x 8
    # We want T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # with T[i,j,k] = 1 for all i,j,k.
    
    # Flatten: for each c, we have u_c (8,), v_c (16,), w_c (8,)
    # T[i,j,k] = sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
    
    # This is equivalent to: for each pair (i,k), sum_j T[i,j,k] = sum_j 1 = m = 4
    # and T[i,j,k] = 1.
    
    # Let's build the system. For a given rank r, we choose random U and V,
    # then solve for W. But the system is:
    # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is linear in W if U and V are fixed.
    
    # Let's try rank 28 first (4 Strassen blocks, but we can omit unused ones).
    # Actually, let's just build the Strassen-based rank-14 decomposition.
    
    # For 2x4 by 4x4:
    # A is 2x4, B is 4x4, C is 2x4
    # Think of A as [[A11, A12]] where A11 is 2x2, A12 is 2x2
    # B as [[B11, B12], [B21, B22]] each 2x2
    # C as [[C11, C12]] where C11 = A11*B11 + A12*B21, C12 = A11*B12 + A12*B22
    
    # Using Strassen on 2x2 blocks:
    # For C11: use 7 mults on A11, A12 (as the "A" matrix, but A is 2x2 here)
    # Actually, A11 and A12 form a 2x2 block matrix A_top = [[A11, A12]] (2x4)
    # B is 4x4 = 2x2 blocks of 2x2
    
    # Let's use the block Strassen approach:
    # A_top (2x4) * B (4x4) = C_top (2x4)
    # Split B into B_left (4x2) and B_right (4x2)
    # C_top = A_top * B_left gives C11 (2x2), A_top * B_right gives C12 (2x2)
    
    # For A_top * B_left: A_top is 2x4, B_left is 4x2
    # This is like multiplying a 2x4 by 4x2, which can be done with Strassen
    # by embedding into 4x4 x 4x4.
    
    # Simpler: just use the rank-32 schoolbook but with some zero columns removed
    # if possible. Actually, let's try a linear algebra approach.
    
    # Build the tensor and try to find a low-rank decomposition via random projection.
    
    best_result = None
    best_rank = n * m * p  # 32
    
    # Build schoolbook as fallback
    rank_fallback = n * m * p
    U_fb = np.zeros((n * m, rank_fallback))
    V_fb = np.zeros((m * p, rank_fallback))
    W_fb = np.zeros((p * n, rank_fallback))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    best_result = {'U': U_fb.tolist(), 'V': V_fb.tolist(), 'W': W_fb.tolist()}
    best_rank = rank_fallback
    
    # Now try to find lower-rank decompositions
    # Try ranks from 27 down to 14
    for target_rank in range(27, 13, -1):
        if time.time() > DEADLINE:
            break
        
        # Try several random initializations
        for trial in range(20):
            if time.time() > DEADLINE:
                break
            
            np.random.seed(trial * 1000 + target_rank)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            
            # Build the linear system for W
            # For each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # Group by (i, k): the unknowns are W[k*n+i, c] for c in range(rank)
            # Actually W is indexed by (k*n+i, c), so for fixed (i,k), the unknowns are W[k*n+i, :]
            
            # For fixed (i, k): sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = sum_j 1 = m
            # Wait, T[i,j,k] = 1 for all j, so sum_j T[i,j,k] = m
            # But we need T[i,j,k] = 1 for each individual j.
            
            # Let's set up the system properly.
            # Unknowns: W[k*n+i, c] for all k, i, c -> total p*n*rank unknowns
            # Equations: for each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # Total equations: n*m*p = 32
            # Total unknowns: p*n*rank = 8*rank
            
            # For rank >= 4, we have more unknowns than equations, so a solution should exist.
            
            # Build the system: A_sys * W_flat = b
            # W_flat is of size (p*n*rank,)
            # For equation (i,j,k): index in b is i*m*p + j*p + k
            # The coefficients: for c in range(rank), coeff of W[k*n+i, c] is U[i*m+j, c] * V[j*p+k, c]
            
            num_eqs = n * m * p
            num_unk = p * n * target_rank
            A_sys = np.zeros((num_eqs, num_unk))
            b = np.ones(num_eqs)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for c in range(target_rank):
                            unk_idx = (k * n + i) * target_rank + c
                            A_sys[eq_idx, unk_idx] = U[i * m + j, c] * V[j * p + k, c]
            
            # Solve the least squares system
            W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b, rcond=None)
            W = W_flat.reshape(p * n, target_rank)
            
            # Verify
            if _verify(U, V, W, n, m, p, tol=1e-6):
                # Found a valid decomposition
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = target_rank
                break  # Break out of trial loop
            else:
                # Check if it's close
                # Compute max error
                T_rec = np.zeros((n, m, p))
                for c in range(target_rank):
                    Uc = U[:, c].reshape(n, m)
                    Vc = V[:, c].reshape(m, p)
                    Wc = W[:, c].reshape(p, n)
                    for k in range(p):
                        Wck = Wc[k, :]
                        Vck = Vc[:, k]
                        T_rec[:, :, k] += Uc * np.outer(Wck, Vck)
                max_err = np.max(np.abs(T_rec - 1.0))
                if max_err < 1e-4:
                    # Close, but not exact. Try to refine with integer rounding
                    # Scale to make entries half-integers
                    scale = 2.0
                    U_int = np.round(U * scale) / scale
                    V_int = np.round(V * scale) / scale
                    W_int = np.round(W * scale) / scale
                    if _verify(U_int, V_int, W_int, n, m, p, tol=1e-9):
                        best_result = {'U': U_int.tolist(), 'V': V_int.tolist(), 'W': W_int.tolist()}
                        best_rank = target_rank
                        break
    
    if time.time() > DEADLINE - 10:
        # Return best found so far
        pass
    
    return best_result

# EVOLVE-BLOCK-END
