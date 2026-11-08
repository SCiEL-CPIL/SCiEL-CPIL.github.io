"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Build the target tensor T[i,j,k] = 1 if this corresponds to a valid multiplication term
    # T[i*m+j, j*p+k, k*n+i] should be 1 for the decomposition
    # Actually, the tensor is defined such that:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for the "identity" in the multiplication
    
    # The tensor we're decomposing has entries:
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # and T[a,b,c] = 1 if the indices are consistent, else 0
    
    # More precisely, for the matrix multiplication tensor:
    # T is a 3D tensor of size (n*m) x (m*p) x (p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # All other entries are 0
    
    # We want to find U (n*m x rank), V (m*p x rank), W (p*n x rank) such that:
    # sum_c U[a,c] * V[b,c] * W[c_idx,c] = T[a,b,c_idx]
    
    # Start with schoolbook rank
    rank_schoolbook = n * m * p
    
    # Build schoolbook decomposition as baseline
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Verify schoolbook is valid
    def verify_decomposition(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        # Check all valid (i,j,k) triples give 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    recon = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                    if abs(recon - 1.0) > tol:
                        return False
        # Check all invalid triples give 0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Determine if (a,b,c) is valid
                    i1, j1 = divmod(a, m)
                    j2, k1 = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    is_valid = (j1 == j2 and k1 == k2 and i1 == i2)
                    recon = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                    if is_valid:
                        if abs(recon - 1.0) > tol:
                            return False
                    else:
                        if abs(recon) > tol:
                            return False
        return True
    
    # Try Strassen-based construction for 4x4x4 block, then extend to 4x4x5
    # Strategy: Use Strassen's 7-term algorithm for the 4x4x4 part,
    # and handle the extra column separately
    
    # For 4x4x4 matrix multiplication, we can use a rank-7 decomposition
    # (generalized Strassen). Let's construct it explicitly.
    
    # Actually, let's try a more systematic approach:
    # Use the fact that 4x4x4 can be done in rank 7 (Strassen-like)
    # Then 4x4x5 = 4x4x4 + 4x4x1
    
    # For 4x4x1, schoolbook needs 16 terms
    # For 4x4x4, we can use rank 7
    
    # Total: 7 + 16 = 23? But this is not quite right because the structures differ.
    
    # Let me try a different approach: direct construction using Kronecker products
    # or explicit algebraic identities.
    
    # First, let's try to find a good decomposition using structured random search
    # with algebraic constraints (entries in {-1, 0, 1, 0.5})
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Store schoolbook as fallback
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound for matrix multiplication tensor: at least max(n, m, p)
    # But for 4x4x5, known results suggest rank around 20-30
    
    # Let's try structured search with small integer/half-integer entries
    min_rank_to_try = 15
    
    for rank in range(min_rank_to_try, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try a few structured random initializations
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(rank * 1000 + trial)
            
            # Use small integer/half-integer entries: {-1, -0.5, 0, 0.5, 1}
            possible_vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            U = np.array([[possible_vals[np.random.randint(len(possible_vals))] for _ in range(rank)] for _ in range(n * m)])
            V = np.array([[possible_vals[np.random.randint(len(possible_vals))] for _ in range(rank)] for _ in range(m * p)])
            W = np.array([[possible_vals[np.random.randint(len(possible_vals))] for _ in range(rank)] for _ in range(p * n)])
            
            # Quick check: verify only the valid (i,j,k) entries
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        recon = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                        if abs(recon - 1.0) > 1e-10:
                            valid = False
                            break
            if not valid:
                continue
            
            # Full verification
            if verify_decomposition(U, V, W):
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    if time.time() > deadline - 5:
        return best_result
    
    # Ensure output format is correct
    U_out = best_result["U"]
    V_out = best_result["V"]
    W_out = best_result["W"]
    
    # Round to nearest half-integer if close
    def round_to_half(x):
        return round(x * 2) / 2
    
    if isinstance(U_out, list):
        U_out = [[round_to_half(v) for v in row] for row in U_out]
        V_out = [[round_to_half(v) for v in row] for row in V_out]
        W_out = [[round_to_half(v) for v in row] for row in W_out]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
