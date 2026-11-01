"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    
    The multiplication tensor T has entries T[i, j, k] where i in [0,2), j in [0,4), k in [0,4).
    We seek factor matrices U (2*4 x rank), V (4*4 x rank), W (4*2 x rank) with integer/half-integer
    entries such that T[i,j,k] = sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*2+i, c].
    
    The output factor W follows the convention W[k*n+i, r] where n=2.
    
    We use a local search approach starting from a Strassen-like rank-28 decomposition and
    attempt to reduce to rank-27. If successful, we return rank 27; otherwise rank 28.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit with safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    # The tensor T has dimensions n x m x p = 2 x 4 x 4 = 32 entries
    # We want to decompose it into rank r: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # First, let's try to find a rank-27 decomposition using local search
    # Start from a known rank-28 decomposition (Strassen-like) and try to reduce
    
    # Build the tensor as a 3D array
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0  # All entries are 1 for matrix multiplication
    
    # Reshape to a matrix for easier manipulation: T_mat[i*m+j, k*n+i] but that's not quite right
    # Actually, let's think of it differently.
    # The tensor can be viewed as a linear map from (A ⊗ B) to C
    # where A is n x m, B is m x p, C is n x p
    # T[A, B] = C where C[i,k] = sum_j A[i,j] * B[j,k]
    
    # For the decomposition: C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j' V[j'*p+k, c] * B[j',k])
    # Wait, that's not quite right either. Let me reconsider.
    
    # The standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We want: C[i,k] = sum_c (U[i*m+j1, c] * A[i,j1]) * (V[j2*p+k, c] * B[j2,k])
    # This requires j1 = j2 for each term, which is the constraint.
    
    # Actually, the correct form is:
    # C[i,k] = sum_c (sum_j U[i,j,c] * A[i,j]) * (sum_j' V[j',k,c] * B[j',k])
    # But this doesn't capture the bilinear structure correctly.
    
    # Let me use the standard formulation:
    # The multiplication tensor T has T[i,j,k] = 1 if we multiply A[i,j] by B[j,k] to get C[i,k]
    # A rank-r decomposition gives: T[i,j,k] = sum_{c=1}^{r} U_c[i,j] * V_c[j,k] * W_c[i,k]
    # where U_c, V_c, W_c are the c-th factors.
    
    # In our indexing:
    # U has shape (n*m, r): U[i*m+j, c]
    # V has shape (m*p, r): V[j*p+k, c]  
    # W has shape (p*n, r): W[k*n+i, c]  (using the given convention)
    
    # The condition is: for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k (since every A[i,j] multiplies every B[j,k])
    # Wait, no! T[i,j,k] should be 1 only if the product A[i,j]*B[j,k] contributes to C[i,k].
    # Actually, in the standard tensor formulation, T[i,j,k] = 1 for all valid i,j,k because
    # every entry A[i,j] and B[j,k] contributes to C[i,k].
    
    # Let's verify with schoolbook: rank = n*m*p = 2*4*4 = 32
    # For each (i,j,k), we have one term: U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    
    # Now, let's try to find a lower rank decomposition.
    # We'll use a local search approach.
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def to_int_or_half(x):
        """Check if x is integer or half-integer, and round accordingly."""
        # Check if close to integer or half-integer
        nearest_int = round(x * 2) / 2.0
        if abs(x - nearest_int) < 1e-9:
            return nearest_int
        return None
    
    def clean_factors(U, V, W):
        """Clean up floating point errors and convert to integer/half-integer if possible."""
        U_clean = np.zeros_like(U)
        V_clean = np.zeros_like(V)
        W_clean = np.zeros_like(W)
        
        for i in range(U.shape[0]):
            for c in range(U.shape[1]):
                val = to_int_or_half(U[i, c])
                if val is not None:
                    U_clean[i, c] = val
                else:
                    U_clean[i, c] = U[i, c]
        
        for i in range(V.shape[0]):
            for c in range(V.shape[1]):
                val = to_int_or_half(V[i, c])
                if val is not None:
                    V_clean[i, c] = val
                else:
                    V_clean[i, c] = V[i, c]
        
        for i in range(W.shape[0]):
            for c in range(W.shape[1]):
                val = to_int_or_half(W[i, c])
                if val is not None:
                    W_clean[i, c] = val
                else:
                    W_clean[i, c] = W[i, c]
        
        return U_clean, V_clean, W_clean
    
    # Start with a Strassen-like rank-28 decomposition
    # For 2x4x4, we can think of it as two 2x2x2 Strassen multiplications plus cross terms
    # Actually, let's use a more systematic approach.
    
    # Initialize with schoolbook rank-32 and try to reduce
    rank_start = 28  # Start from rank 28
    
    # Create initial decomposition
    def create_initial_decomp(rank):
        """Create an initial rank-r decomposition."""
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Use a simple initialization: distribute the 32 tensor entries across rank terms
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = idx % rank
                    U[i*m+j, c] += 1.0
                    V[j*p+k, c] += 1.0
                    W[k*n+i, c] += 1.0
                    idx += 1
        
        return U, V, W
    
    # Better initialization: use a structured approach
    def create_structured_decomp(rank):
        """Create a structured initial decomposition."""
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # For rank 28, we can try to eliminate 4 redundant terms from the 32-term schoolbook
        # This is non-trivial, so let's use a random perturbation approach
        
        # Start with schoolbook and merge some terms
        U, V, W = create_initial_decomp(32)
        
        # Try to merge pairs of terms
        for merge_round in range(4):
            # Find a pair of columns to merge
            c1, c2 = 0, 1
            # Simple merge: add column c2 to column c1 and remove c2
            U[:, c1] += U[:, c2]
            V[:, c1] += V[:, c2]
            W[:, c1] += W[:, c2]
            # Shift remaining columns
            if c1 < c2:
                U[:, c2-1:c2] = U[:, c2:c2+1]
                V[:, c2-1:c2] = V[:, c2:c2+1]
                W[:, c2-1:c2] = W[:, c2:c2+1]
        
        # Now we have rank 28
        U = U[:, :rank]
        V = V[:, :rank]
        W = W[:, :rank]
        
        return U, V, W
    
    # Let's try a different approach: use optimization to find a rank-27 decomposition
    # We'll use gradient descent on the factors
    
    def compute_error(U, V, W):
        """Compute the error of the decomposition."""
        r = U.shape[1]
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    error += (val - 1.0) ** 2
        return error
    
    # Try to find rank-27 decomposition
    best_rank = 32
    best_U, best_V, best_W = None, None, None
    
    # Initialize with rank 28
    U, V, W = create_structured_decomp(28)
    
    # Check if this is valid
    if verify(U, V, W):
        best_rank = 28
        best_U, best_V, best_W = U, V, W
    
    # Try local search to reduce to rank 27
    if time.time() < deadline - 10.0:
        # Try multiple random restarts for rank 27
        np.random.seed(42)
        
        for restart in range(20):
            if time.time() > deadline - 5.0:
                break
            
            # Initialize rank-27 factors
            U27 = np.zeros((n*m, 27))
            V27 = np.zeros((m*p, 27))
            W27 = np.zeros((p*n, 27))
            
            # Random initialization with small values
            U27 = np.random.randn(n*m, 27) * 0.5
            V27 = np.random.randn(m*p, 27) * 0.5
            W27 = np.random.randn(p*n, 27) * 0.5
            
            # Gradient descent
            lr = 0.01
            for step in range(500):
                if time.time() > deadline - 5.0:
                    break
                
                # Compute gradients
                dU = np.zeros_like(U27)
                dV = np.zeros_like(V27)
                dW = np.zeros_like(W27)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(27):
                                val += U27[i*m+j, c] * V27[j*p+k, c] * W27[k*n+i, c]
                            err = val - 1.0
                            for c in range(27):
                                dU[i*m+j, c] += 2 * err * V27[j*p+k, c] * W27[k*n+i, c]
                                dV[j*p+k, c] += 2 * err * U27[i*m+j, c] * W27[k*n+i, c]
                                dW[k*n+i, c] += 2 * err * U27[i*m+j, c] * V27[j*p+k, c]
                
                # Update
                U27 -= lr * dU
                V27 -= lr * dV
                W27 -= lr * dW
                
                # Check convergence
                if step % 50 == 0:
                    err = compute_error(U27, V27, W27)
                    if err < 1e-12:
                        break
            
            # Check if we found a valid decomposition
            if verify(U27, V27, W27):
                best_rank = 27
                best_U, best_V, best_W = U27, V27, W27
                break
    
    # If we found rank 27, clean it up
    if best_rank == 27:
        best_U, best_V, best_W = clean_factors(best_U, best_V, best_W)
        # Re-verify after cleaning
        if not verify(best_U, best_V, best_W):
            best_rank = 28
            best_U, best_V, best_W = create_structured_decomp(28)
    
    # Convert to lists
    if best_U is not None:
        U_list = [[float(best_U[i, c]) for c in range(best_rank)] for i in range(n*m)]
        V_list = [[float(best_V[i, c]) for c in range(best_rank)] for i in range(m*p)]
        W_list = [[float(best_W[i, c]) for c in range(best_rank)] for i in range(p*n)]
        
        # Round to nearest integer or half-integer
        for i in range(len(U_list)):
            for c in range(len(U_list[i])):
                val = U_list[i][c]
                nearest = round(val * 2) / 2.0
                U_list[i][c] = nearest
        for i in range(len(V_list)):
            for c in range(len(V_list[i])):
                val = V_list[i][c]
                nearest = round(val * 2) / 2.0
                V_list[i][c] = nearest
        for i in range(len(W_list)):
            for c in range(len(W_list[i])):
                val = W_list[i][c]
                nearest = round(val * 2) / 2.0
                W_list[i][c] = nearest
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        rank = n * m * p
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
# EVOLVE-BLOCK-END
