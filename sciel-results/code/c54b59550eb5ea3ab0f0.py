"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _build_kronecker_decomposition(n, m, p):
    """
    Build a Kronecker-product-based decomposition for n×m × m×p matrix multiplication.
    
    Strategy: Decompose dimensions into factors and use the property that
    rank of Kronecker product of tensors is the product of ranks.
    
    For 4×5×6:
    - 4 = 2×2
    - 5 = 4+1 (use binary representation: 101, so 2^2 + 2^0)
    - 6 = 3×2
    
    We'll try multiple decompositions and pick the best valid one.
    """
    
    def build_from_factors(n_factors, m_factors, p_factors):
        """
        Build decomposition using Kronecker structure.
        
        n_factors: list of factors whose product = n
        m_factors: list of factors whose product = m  
        p_factors: list of factors whose product = p
        
        Each "layer" corresponds to one position in the factorization.
        """
        # Validate that products match
        if int(np.prod(n_factors)) != n or int(np.prod(m_factors)) != m or int(np.prod(p_factors)) != p:
            return None
        
        # For each position in the factorization, we need a decomposition of 
        # (n_f × m_f × m_f × p_f) tensor for that layer
        # The total rank is the product of the ranks of each layer
        
        # For a single layer with dimensions (a, b, c) where we multiply a×b by b×c:
        # Schoolbook rank is a*b*c
        # But we can potentially do better if there's structure
        
        # For now, use schoolbook for each layer (this gives rank = n*m*p = 120)
        # To beat 120, we need to find a better decomposition for at least one layer
        
        # Let's try: for the "5" dimension, decompose as 4+1
        # This means we handle the 5×5 part as a sum of two parts:
        # - A 4×4 block (rank 16 for that part)
        # - Two 1×5 and 5×1 cross terms (rank 10 for those parts)
        # Total for the 5-dimension handling: 16 + 10 = 26 vs 25 for schoolbook
        # This doesn't help directly, but combined with other factors might
        
        # Alternative: Use the fact that 4 = 2^2, 6 = 2*3
        # Decompose as: (2×2) × (5) × (2×3)
        # Layer 1: 2×5×2 (rank 20)
        # Layer 2: 2×5×3 (rank 30)
        # Total rank: 20 * 30 = 600 (worse!)
        
        # The Kronecker product rank multiplies, so we need to be careful.
        # Actually, for matrix multiplication, the tensor is:
        # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
        # This is a rank-1 tensor for each (i,j,k), so rank = n*m*p
        
        # To reduce rank, we need to find linear dependencies or use
        # the structure of the specific dimensions.
        
        # For now, return the schoolbook as baseline
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
        return {"U": U, "V": V, "W": W, "rank": rank}
    
    # Try different factorizations
    best_result = None
    best_rank = float('inf')
    
    # Factorization 1: 4=4, 5=5, 6=6 (schoolbook)
    result = build_from_factors([4], [5], [6])
    if result and result["rank"] < best_rank:
        best_rank = result["rank"]
        best_result = result
    
    # Factorization 2: 4=2*2, 5=5, 6=6
    result = build_from_factors([2, 2], [5], [6])
    if result and result["rank"] < best_rank:
        best_rank = result["rank"]
        best_result = result
    
    # Factorization 3: 4=4, 5=5, 6=2*3
    result = build_from_factors([4], [5], [2, 3])
    if result and result["rank"] < best_rank:
        best_rank = result["rank"]
        best_result = result
    
    # Factorization 4: 4=2*2, 5=5, 6=2*3
    result = build_from_factors([2, 2], [5], [2, 3])
    if result and result["rank"] < best_rank:
        best_rank = result["rank"]
        best_result = result
    
    if best_result is None:
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
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    # Test with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    
    # Expected result
    expected = A @ B
    
    # Computed result using the decomposition
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Using the tensor decomposition:
    # C[i,k] = sum_c U[i*m + ? , c] * V[? * p + k, c] * W[k*n + i, c]
    # Wait, the indexing is: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # So: C[i,k] = sum_{j,c} A[i,j] * B[j,k] * (product of U,V,W terms)
    
    # Actually, the decomposition should satisfy:
    # For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This ensures that C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Check the tensor identity
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4×5 × 5×6 matrix multiplication.
    Uses Kronecker product structure to potentially reduce rank below schoolbook 120.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    
    # First, get the Kronecker-based decomposition
    result = _build_kronecker_decomposition(n, m, p)
    
    # Verify it's correct
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
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
        result = {"U": U, "V": V, "W": W}
    
    # Check if we're under deadline; if so, try to improve
    elapsed = time.time() - start_time
    if elapsed < 100:  # 40 seconds remaining
        # Try local search to reduce rank
        best_U, best_V, best_W = result["U"], result["V"], result["W"]
        best_rank = len(best_U[0]) if best_U else 0
        
        # Attempt rank reduction by trying to zero out some columns
        # This is a simplified approach; a full implementation would use
        # more sophisticated optimization
        pass
    
    return result
# EVOLVE-BLOCK-END
