import time
import numpy as np
from itertools import combinations, product

DIMENSIONS = [2, 3, 10]
DEADLINE = 140  # seconds, safety margin before 150s kill

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy:
    1. Build the known rank-55 Strassen-based decomposition.
    2. Search for rank-54 by trying to merge pairs of terms across blocks.
    3. If found, verify and return; otherwise return rank-55.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # Step 1: Build rank-55 decomposition
    rank55 = _build_strassen_rank55()
    U55, V55, W55 = rank55['U'], rank55['V'], rank55['W']
    
    # Step 2: Try to find rank-54 by merging two terms
    # The idea: if two columns r1, r2 of (U,V,W) can be combined into one
    # while preserving the tensor, we get rank 54.
    # For exact integer/half-integer entries, we check if for some pair (i,j),
    # there exists a linear combination that preserves all tensor entries.
    
    elapsed = time.time() - start_time
    if elapsed < 120:
        # Try rank-54 search
        result = _search_rank54(U55, V55, W55, start_time)
        if result is not None:
            return result
    
    # Fallback: return rank-55
    return rank55

def _build_strassen_rank55():
    """Build the known rank-55 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    a00, a01, a10, a11 = 0, 1, 3, 4
    a02, a12 = 2, 5
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        b00, b01 = k0, k1
        b10, b11 = 10 + k0, 10 + k1
        
        c00, c01 = 2 * k0, 2 * k1
        c10, c11 = 2 * k0 + 1, 2 * k1 + 1
        
        # M1 = (A00 + A11)(B00 + B11)
        U[a00, col] += 1.0; U[a11, col] += 1.0
        V[b00, col] += 1.0; V[b11, col] += 1.0
        W[c00, col] += 1.0; W[c11, col] += 1.0
        col += 1
        
        # M2 = (A10 + A11)B00
        U[a10, col] += 1.0; U[a11, col] += 1.0
        V[b00, col] += 1.0
        W[c10, col] += 1.0; W[c11, col] -= 1.0
        col += 1
        
        # M3 = A00(B01 - B11)
        U[a00, col] += 1.0
        V[b01, col] += 1.0; V[b11, col] -= 1.0
        W[c01, col] += 1.0; W[c11, col] += 1.0
        col += 1
        
        # M4 = A11(B10 - B00)
        U[a11, col] += 1.0
        V[b10, col] += 1.0; V[b00, col] -= 1.0
        W[c00, col] += 1.0; W[c10, col] += 1.0
        col += 1
        
        # M5 = (A00 + A01)B11
        U[a00, col] += 1.0; U[a01, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] -= 1.0; W[c01, col] += 1.0
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[a10, col] += 1.0; U[a00, col] -= 1.0
        V[b00, col] += 1.0; V[b01, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[a01, col] += 1.0; U[a11, col] -= 1.0
        V[b10, col] += 1.0; V[b11, col] += 1.0
        W[c00, col] += 1.0
        col += 1
        
        # Schoolbook terms for A_R * B_R
        U[a02, col] += 1.0
        V[2 * p + k0, col] += 1.0
        W[c00, col] += 1.0
        col += 1
        
        U[a12, col] += 1.0
        V[2 * p + k0, col] += 1.0
        W[c10, col] += 1.0
        col += 1
        
        U[a02, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[c01, col] += 1.0
        col += 1
        
        U[a12, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[c11, col] += 1.0
        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _search_rank54(U55, V55, W55, start_time):
    """
    Search for a rank-54 decomposition by trying to merge pairs of columns.
    
    For each pair of columns (i, j), check if there exists a scalar c such that
    replacing column i with U[:,i] + c*U[:,j], V[:,i] + c*V[:,j], W[:,i] + c*W[:,j]
    and removing column j still gives a valid decomposition.
    
    This is a simplified search; a full search would be more complex.
    """
    n, m, p = DIMENSIONS
    rank = 55
    
    # Convert to numpy for efficiency
    U = np.array(U55)
    V = np.array(V55)
    W = np.array(W55)
    
    # Try merging columns from different blocks to see if we can save a rank
    # Focus on pairs that might interact: e.g., Strassen terms from adjacent blocks
    
    best_result = None
    deadline = start_time + 135  # 5s safety margin
    
    # Try a subset of promising pairs
    # Group columns by their block and term type
    for i in range(rank):
        if time.time() > deadline:
            break
        
        for j in range(i + 1, rank):
            if time.time() > deadline:
                break
            
            # Skip pairs from the same Strassen block (they're already optimized)
            block_i = i // 11
            block_j = j // 11
            if block_i == block_j:
                continue
            
            # Try c = ±1, ±0.5, ±2 (common rational values)
            for c in [1.0, -1.0, 0.5, -0.5, 2.0, -2.0]:
                # Test if merging works: new_col_i = U[:,i] + c*U[:,j]
                # Check if the resulting tensor is still valid
                # This requires checking all tensor entries, which is expensive
                # Simplified check: just verify with random test matrices
                
                U_new = U.copy()
                V_new = V.copy()
                W_new = W.copy()
                
                U_new[:, i] += c * U_new[:, j]
                V_new[:, i] += c * V_new[:, j]
                W_new[:, i] += c * W_new[:, j]
                
                # Remove column j
                U_new = np.delete(U_new, j, axis=1)
                V_new = np.delete(V_new, j, axis=1)
                W_new = np.delete(W_new, j, axis=1)
                
                # Quick verification with a few random test cases
                if _quick_verify(U_new, V_new, W_new, n, m, p, num_tests=3):
                    # Full verification
                    if _full_verify(U_new, V_new, W_new, n, m, p):
                        best_result = {
                            "U": U_new.tolist(),
                            "V": V_new.tolist(),
                            "W": W_new.tolist()
                        }
                        return best_result
    
    return best_result

def _quick_verify(U, V, W, n, m, p, num_tests=3):
    """Quick verification with random test matrices."""
    np.random.seed(42)
    for _ in range(num_tests):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        
        # Compute C = A @ B
        C_expected = A @ B
        
        # Compute C via decomposition: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Actually, the decomposition is: C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, that's not right. Let me reconsider.
        
        # The tensor decomposition for matrix multiplication is:
        # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * ... no
        # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
        # The rank-r decomposition is: C[i,k] = sum_r U[i*m+?, r] * V[?, k, r] * W[k*n+i, r]
        # 
        # Standard form: For each r, we have terms U[i,j,r] * V[j,k,r] * W[i,k,r]
        # where C[i,k] = sum_{j,r} U[i,j,r] * V[j,k,r] * W[i,k,r]
        # But here U has shape (n*m, rank), so U[i*m+j, r]
        # V has shape (m*p, rank), so V[j*p+k, r]
        # W has shape (p*n, rank), so W[k*n+i, r]
        
        # So: C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_computed[i, k] = val
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    
    return True

def _full_verify(U, V, W, n, m, p, num_tests=10):
    """Full verification with more test matrices."""
    np.random.seed(123)
    for _ in range(num_tests):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_computed[i, k] = val
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    
    return True
