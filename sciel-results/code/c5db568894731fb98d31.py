"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x7 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns:
        dict: {'U': U, 'V': V, 'W': W} where:
            - U has shape (n*m, rank) = (8, rank)
            - V has shape (m*p, rank) = (28, rank)
            - W has shape (p*n, rank) = (14, rank)
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # Deadline guard: kill after 140s, leave 10s margin before 150s limit
    start_time = time.time()
    DEADLINE = 140.0
    
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 2*4*7 = 56
    
    def build_schoolbook():
        """Build the standard schoolbook decomposition."""
        U = np.zeros((n * m, rank_schoolbook))
        V = np.zeros((m * p, rank_schoolbook))
        W = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = build_schoolbook()
    best_rank = rank_schoolbook
    
    # Try to find better decompositions using structured approaches
    
    # Strategy: Use fast matrix multiplication algorithms where applicable
    # For rectangular matrices, we can try block-based approaches
    
    # Approach 1: Try to exploit structure by considering the problem as
    # multiple smaller matrix multiplications combined
    
    # For 2x4 * 4x7, we can view this as:
    # C = A * B where A is 2x4, B is 4x7
    # C is 2x7
    
    # One approach: split B into columns and do 4x7 -> 4x1 multiplications
    # But that's just schoolbook again.
    
    # Better approach: Use Strassen-like techniques for rectangular matrices
    
    # Let's try a different strategy: random search with validation
    # We'll try to find a lower-rank decomposition by solving a system of equations
    
    def validate_decomposition(U, V, W, tol=1e-10):
        """
        Validate that U, V, W form a valid decomposition.
        
        For each output element C[i,k] = sum_r U[i*4+j, r] * V[j*7+k, r] * W[k*2+i, r]
        This should equal the standard matrix product.
        """
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        
        # Build random test matrices
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        
        # Standard product
        C_standard = A @ B
        
        # Product using decomposition
        # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
        # Actually the correct formula is:
        # C[i,k] = sum_r U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r] * A[i,j1] * B[j2,k]
        # No wait, the tensor decomposition means:
        # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
        # That's not right either. Let me think more carefully.
        
        # The matrix multiplication tensor T[i,j,k] = delta (where T[i,j,k]=1 if i,j,k form a valid product term)
        # A rank-r decomposition gives:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k), so:
        # 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k
        
        # And the actual product is:
        # C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r (U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]) * A[i,j] * B[j,k]
        
        # Wait, that's not the standard form. Let me reconsider.
        
        # The correct interpretation: The bilinear map for matrix multiplication is:
        # (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
        
        # A rank-r decomposition of this bilinear map gives:
        # C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k]) * W[k*n+i, r]
        
        # This must equal sum_j A[i,j] B[j,k] for all A, B.
        
        # So for each i,k:
        # sum_j A[i,j] B[j,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
        
        # For this to hold for all A, B, we need the bilinear forms to match.
        
        # Let's verify with specific test cases.
        
        for test_idx in range(10):
            A_test = np.random.rand(n, m)
            B_test = np.random.rand(m, p)
            C_standard_test = A_test @ B_test
            
            # Compute using decomposition
            C_decomp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    sum_val = 0.0
                    for r in range(rank):
                        # Compute intermediate products
                        prod_A = 0.0
                        for j in range(m):
                            prod_A += U[i * m + j, r] * A_test[i, j]
                        prod_B = 0.0
                        for j in range(m):
                            prod_B += V[j * p + k, r] * B_test[j, k]
                        sum_val += W[k * n + i, r] * prod_A * prod_B
                    C_decomp[i, k] = sum_val
            
            if not np.allclose(C_decomp, C_standard_test, atol=tol):
                return False
        
        return True
    
    def try_rank(r, max_iters=1000):
        """
        Try to find a rank-r decomposition using random initialization and gradient descent.
        Returns True if a valid decomposition is found, False otherwise.
        """
        n, m, p = DIMENSIONS
        
        # Initialize with random values
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Scale to make initial values reasonable
        scale = 1.0 / np.sqrt(r)
        U *= scale
        V *= scale
        W *= scale
        
        # Use gradient descent to minimize the error
        lr = 0.01
        best_err = np.inf
        
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                return False
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Target: 1 (since T[i,j,k] = 1 for matrix multiplication)
                        # Predicted: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        pred = 0.0
                        for r_idx in range(r):
                            pred += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                        err += (pred - 1.0) ** 2
            
            if err < best_err:
                best_err = err
            
            # Check convergence
            if best_err < 1e-8:
                return validate_decomposition(U, V, W)
            
            # Gradient computation (simplified - this is a placeholder)
            # Full gradient descent would be complex, so let's use a different approach
            
            # Instead, let's try random restarts
            if it % 100 == 99:
                U = np.random.randn(n * m, r) * scale
                V = np.random.randn(m * p, r) * scale
                W = np.random.randn(p * n, r) * scale
                best_err = np.inf
        
        return False
    
    # Try various ranks below schoolbook
    # Known lower bounds for matrix multiplication rank:
    # For n x m * m x p, the rank is at least max(n*m, m*p, n*p)
    # For 2x4 * 4x7: max(8, 28, 14) = 28
    # But actual minimum is unknown and likely higher
    
    # Let's try ranks from 28 to 55
    for rank in range(28, rank_schoolbook):
        if time.time() - start_time > DEADLINE:
            break
        
        # Try to find a decomposition of this rank
        # Using a structured approach based on known fast multiplication algorithms
        
        # For now, let's use a heuristic: try random search with structured initialization
        found = False
        
        # Structured initialization based on block structure
        # Split the 4x7 matrix B into blocks and use fast multiplication
        
        # One idea: Use the fact that we can compute 2x4 * 4x7 as:
        # Treat A as two 2x2 blocks (but A is 2x4, so split columns)
        # A = [A1 | A2] where A1, A2 are 2x2
        # B = [B1; B2] where B1, B2 are 2x7
        # C = A1*B1 + A2*B2
        
        # This gives us a sum of two 2x2 * 2x7 multiplications
        # Each 2x2 * 2x7 can be done with rank at most 2*2*7 = 28, but we can do better
        
        # For 2x2 * 2x7:
        # C[i,k] = sum_j A[i,j] B[j,k] for j=0,1
        # We can use Strassen-like techniques
        
        # Let's try to build a structured decomposition
        
        # For the 2x2 * 2x2 case, Strassen uses rank 7
        # For 2x2 * 2x7, we can potentially use rank less than 28
        
        # Let's try to construct such a decomposition
        if not found:
            # Try a structured approach
            # C[i,k] for i in {0,1}, k in {0,...,6}
            # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
            
            # Use Strassen-inspired approach:
            # Define intermediate products:
            # P0 = A[i,0]*B[0,k]
            # P1 = A[i,1]*B[1,k]
            # C[i,k] = P0 + P1
            
            # But we need to share computations across different k values
            
            # For fixed i, C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
            # This is a linear combination of B[0,k] and B[1,k]
            
            # We can write: C[i,k] = sum_r U[i,k,r] * V[r,k] * W[r,i]
            # where U[i,k,r] depends on A, V[r,k] depends on B, W[r,i] is constant
            
            # Let's try rank 7 for each 2x2 * 2x2 block, then combine
            # But 2x2 * 2x7 is not 2x2 * 2x2
            
            # Alternative: just use schoolbook for now, but try to optimize
            
            pass
        
        if not found:
            # Try random search with small number of iterations
            for attempt in range(50):
                if time.time() - start_time > DEADLINE:
                    break
                
                # Random initialization with structure
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                
                # Try to use a pattern-based initialization
                # This is a heuristic approach
                for r_idx in range(rank):
                    # Assign random sparse patterns
                    U[r_idx, r_idx % (n * m)] = np.random.choice([-1, 1])
                    V[r_idx, r_idx % (m * p)] = np.random.choice([-1, 1])
                    W[r_idx, r_idx % (p * n)] = np.random.choice([-1, 1])
                
                # Add some density
                for r_idx in range(rank):
                    if np.random.rand() < 0.3:
                        U[r_idx, np.random.randint(n * m)] += np.random.choice([-1, 1])
                        V[r_idx, np.random.randint(m * p)] += np.random.choice([-1, 1])
                        W[r_idx, np.random.randint(p * n)] += np.random.choice([-1, 1])
                
                # Validate
                if validate_decomposition(U, V, W):
                    found = True
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {"U": U.copy(), "V": V.copy(), "W": W.copy()}
                    break
        
        if found and rank < best_rank:
            break
    
    # If no better decomposition found, return schoolbook
    return best_result
# EVOLVE-BLOCK-END

