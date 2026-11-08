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
    
    # Try to find a lower rank decomposition
    # Use a simple approach: try ranks from a lower bound up to schoolbook
    
    # Lower bound: the tensor has rank at least max of the mode ranks
    # Mode 1 rank: for fixed b,c, the slice in a has rank <= min(n*m, something)
    # A better lower bound: the number of non-zero "fibers"
    
    # For now, let's try a few ranks below schoolbook and see if we can find a valid decomposition
    # Use random initialization and check if the reconstruction is exact
    
    best_result = None
    best_rank = rank_schoolbook
    
    # First, create the schoolbook decomposition as fallback
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
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # The minimum possible rank is at least the number of non-zero entries in any 2D slice
    # For a fixed j, the slice T[:, j*p+?, ?] ... this is complex
    
    # Let's try ranks from max(20, rank_schoolbook - 5) down to some lower bound
    # Actually, let's try a few specific ranks
    
    # Lower bound: the tensor has at least n*m*p / min(n,m,p) non-zero entries in some sense
    # A simple lower bound is max(n, m, p) but that's too weak
    
    # For matrix multiplication, a known lower bound is (n*m*p) / max(n,m,p) but that's not tight
    
    # Let's just try a few ranks and use random restarts
    min_rank_to_try = max(10, rank_schoolbook // 2)
    
    for rank in range(min_rank_to_try, best_rank):
        if time.time() > deadline - 5:
            break
        
        # Try a few random initializations for this rank
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(rank * 100 + trial)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Check if this decomposition is valid
            # Reconstruct the tensor and compare
            valid = True
            for i in range(n):
                if time.time() > deadline - 5:
                    valid = False
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        # The reconstructed value should be 1
                        recon = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                        if abs(recon - 1.0) > 1e-10:
                            valid = False
                            break
                        # Also check that other entries are 0
                        # This is expensive, so skip for now
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Also need to check that non-diagonal entries are 0
                # This is the hard part
                all_zero = True
                for i in range(n):
                    if time.time() > deadline - 5:
                        all_zero = False
                        break
                    for j in range(m):
                        if not all_zero:
                            break
                        for k in range(p):
                            if not all_zero:
                                break
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            # Check a few random other c values
                            for c_test in range(p * n):
                                if c_test == c:
                                    continue
                                recon_test = np.dot(U[a, :], np.multiply(V[b, :], W[c_test, :]))
                                if abs(recon_test) > 1e-10:
                                    all_zero = False
                                    break
                            if not all_zero:
                                break
                    if not all_zero:
                        break
                if not all_zero:
                    continue
                
                # Found a valid decomposition
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    if time.time() > deadline - 5:
        return best_result
    
    # Convert to list of lists with integer/half-integer entries if possible
    # For the schoolbook, entries are 0 or 1, which are integers
    # For random, they're floats, which may not be half-integers
    # The task says "integer/half-integer entries", so we need to round or ensure this
    
    # For the schoolbook decomposition, entries are 0 or 1, which are fine
    # For random decompositions, we may need to check if they can be made half-integer
    
    # Since we're returning the best result, and the schoolbook is always valid,
    # let's ensure the output format is correct
    
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
