"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]
DEADLINE = 145.0  # Safety margin before 150s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition baseline (rank 84)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor using vectorized ops."""
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_mat = U[:, r].reshape(n, m)
            v_mat = V[:, r].reshape(m, p)
            w_mat = W[:, r].reshape(p, n)
            for j in range(m):
                outer = np.outer(u_mat[:, j], v_mat[j, :])
                T[:, j, :] += outer * w_mat.T
        if np.max(np.abs(T - 1.0)) > 1e-9:
            return False
        return True
    
    def try_kronecker_ansatz(rank, time_limit):
        """
        Structural Kronecker-product ansatz:
        Express the multiplication tensor as a sum of structured rank-1 terms.
        
        Key idea: For M_{n,m,p}, use the fact that we can write:
        T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        Use a Kronecker structure: partition indices and use tensor products of
        smaller structured matrices.
        
        Specifically, try:
        - U[i,j,r] = sum over factors of products of small structured entries
        - Use the "bit decomposition" or "digit" approach
        
        Alternative structural approach: use the fact that for specific (n,m,p),
        there exist known algebraic decompositions. For 3x4x7, try to find
        a decomposition where the factors have a specific cyclic or group structure.
        """
        if time.time() - start_time > time_limit:
            return None
        
        # Strategy 1: Use a "block" structure where we group indices
        # For 3x4x7: try to use the fact that 4 = 2*2, 7 is prime
        # Try a decomposition based on the structure of the multiplication
        
        # Strategy 2: Use a "sum of Kronecker products" approach
        # T = sum_s (A_s ⊗ B_s ⊗ C_s) where each term is a rank-1 tensor
        # with structured entries
        
        # Let's try a more direct approach: parameterize with a small set of
        # "template" patterns and solve for coefficients
        
        # Template approach: Assume U, V, W have entries from a small set
        # {0, ±1, ±1/2} and use a systematic search over patterns
        
        # For rank r, we have r columns. Each column (r-th) defines:
        # u_r[i,j], v_r[j,k], w_r[k,i]
        # such that sum_r u_r[i,j]*v_r[j,k]*w_r[k,i] = 1
        
        # Key structural insight: if we can find a set of "basis" patterns
        # for u, v, w that are "orthogonal" in some sense, we can combine them.
        
        # Try: use the "Fourier" structure on cyclic groups
        # For n=3: use 3rd roots of unity (but we need real/half-integer)
        # Instead, use structured ±1 patterns (Hadamard-like)
        
        # Hadamard matrices for small sizes:
        # H_2 = [[1,1],[1,-1]]
        # H_4 = [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]
        
        # For m=4, we can use H_4 structure for V
        # For n=3, we need a 3x3 structured matrix
        # For p=7, we need a 7x7 structured matrix
        
        # Let's try a "separable" ansatz:
        # U[i,j,r] = a_r[i] * b_r[j]
        # V[j,k,r] = b_r[j] * c_r[k]  
        # W[k,i,r] = c_r[k] * a_r[i]
        # Then: sum_r a_r[i]*b_r[j]^2*c_r[k]*a_r[i] = sum_r a_r[i]^2 * b_r[j]^2 * c_r[k]
        # For this to equal 1 for all i,j,k, we need:
        # sum_r (a_r[i]*b_r[j])^2 * c_r[k] = 1
        # This is a system that may have solutions.
        
        # More general: U[i,j,r] = u_r[i] * v_r[j], V[j,k,r] = v_r[j] * w_r[k], W[k,i,r] = w_r[k] * u_r[i]
        # Then T[i,j,k] = sum_r u_r[i]^2 * v_r[j]^2 * w_r[k]^2
        # Need: sum_r (u_r[i]*v_r[j]*w_r[k])^2 = 1 for all i,j,k
        
        # This is equivalent to finding a set of vectors such that their
        # "coordinate-wise product squared sum" is constant.
        
        # For a simpler case: if we set u_r[i] = 1/sqrt(n) * delta_{i, r mod n}, etc.
        # But we need exact half-integer solutions.
        
        # Let's try a direct constructive approach:
        # Use the "lattice" method: express the tensor as a sum of "cuboids"
        
        # For rank r, try to assign each (i,j,k) triple to one or more rank slots
        # such that the contributions add up to 1.
        
        # Systematic approach: use integer linear programming-style search
        # over small coefficient sets
        
        coeff_set = [0, 1, -1, 0.5, -0.5]
        
        # For small ranks, try exhaustive search over structured patterns
        for rank_attempt in range(12, min(rank + 1, 25)):
            if time.time() - start_time > time_limit:
                return None
            
            # Try a "grid" structure: divide the (i,j,k) space into blocks
            # and assign each block to a rank slot with appropriate weights
            
            # For 3x4x7 = 84 elements, try to cover with rank_attempt slots
            # Each slot covers a "slab" or "block"
            
            # Block structure: group by i (3 groups), j (4 groups), k (7 groups)
            # Try: slot r covers all (i,j,k) where (i+j+k) mod rank_attempt == r
            # Then U[i,j,r] = 1 if (i+j) mod rank_attempt == r else 0
            # V[j,k,r] = 1 if (j+k) mod rank_attempt == r else 0  
            # W[k,i,r] = 1 if (k+i) mod rank_attempt == r else 0
            # This won't work directly, but let's try variations
            
            # Better: use a "tiled" approach
            # For rank r, we can think of it as r "layers"
            # Layer r handles a subset of the 84 (i,j,k) triples
            
            # Try: for each rank slot r, define a "support" pattern
            # U[i,j,r] = 1 if (i,j) is in support_U[r], else 0
            # V[j,k,r] = 1 if (j,k) is in support_V[r], else 0
            # W[k,i,r] = 1 if (k,i) is in support_W[r], else 0
            # Then T[i,j,k] = number of r where all three supports contain the respective pairs
            # We need this to be exactly 1 for all (i,j,k)
            
            # This is a combinatorial design problem!
            # We need a "3-dimensional covering code" or "3-uniform hypergraph"
            # where each (i,j,k) is covered exactly once.
            
            # For small ranks, try random constructions with verification
            
            for trial in range(30):
                if time.time() - start_time > time_limit:
                    return None
                
                U = np.zeros((n*m, rank_attempt))
                V = np.zeros((m*p, rank_attempt))
                W = np.zeros((p*n, rank_attempt))
                
                # Random sparse assignment with weights
                for r in range(rank_attempt):
                    # Each rank slot has a random "density" of non-zeros
                    density = random.uniform(0.3, 0.8)
                    
                    # U[i*m+j, r]
                    for idx in range(n*m):
                        if random.random() < density:
                            U[idx, r] = random.choice([1.0, -1.0])
                    
                    # V[j*p+k, r]
                    for idx in range(m*p):
                        if random.random() < density:
                            V[idx, r] = random.choice([1.0, -1.0])
                    
                    # W[k*n+i, r]
                    for idx in range(p*n):
                        if random.random() < density:
                            W[idx, r] = random.choice([1.0, -1.0])
                
                # Check if this works
                if check_decomposition(U, V, W, rank_attempt):
                    return (U, V, W)
        
        # Strategy 3: Use a "structured" ansatz based on the specific dimensions
        # For 3x4x7, note that 3*4*7 = 84
        # Try to find a decomposition with rank < 84 using the algebraic structure
        
        # Key insight: the tensor is "multiplicative" in structure
        # We can use the fact that matrix multiplication is bilinear
        # and exploit this to reduce the rank
        
        # Try: use a "factorization" approach where we write
        # M_{3,4,7} = sum_s M_{a_s, b_s, c_s} (Kronecker products)
        # where a_s*b_s*c_s < a_s+b_s+c_s (winning conditions)
        
        # Known: M_{2,2,2} has rank 7 (Strassen)
        # M_{3,3,3} has rank <= 23 (Landsberg)
        
        # For 3x4x7, we can try:
        # M_{3,4,7} contains M_{2,2,2} as a sub-tensor
        # Use Strassen's algorithm on the 2x2x2 blocks
        
        # Construct using Strassen-like blocks:
        # Divide the 4x4 middle dimension into 2x2 blocks
        # Use 7 multiplications per 2x2x2 block
        
        # This gives a rank of roughly: 3 * 7 * (something)
        # Let's try to construct this explicitly
        
        if time.time() - start_time > time_limit:
            return None
        
        # Strassen-based construction for 3x4x7:
        # The 4x4 matrix can be viewed as 2x2 blocks of 2x2
        # Each 2x2x2 multiplication can be done with 7 terms (Strassen)
        # But we have 3 rows and 7 columns, so we need to be careful
        
        # Alternative: use a "tensor product" of known decompositions
        # M_{3,4,7} = M_{3,2,7} ⊗ M_{1,2,1} + ... (not quite)
        
        # Let's try a direct "Strassen-inspired" construction:
        # For the 4x4 middle, use 2x2 Strassen blocks
        # The 3x4 matrix has 3 rows, 4 columns
        # The 4x7 matrix has 4 rows, 7 columns
        
        # Split the 4 middle indices into two groups of 2
        # For each group, use Strassen's 7-term algorithm
        # Total rank: 3 * 2 * 7 = 42? No, that's not right either.
        
        # Let me think more carefully...
        # Actually, the standard Strassen algorithm for 2x2x2 uses rank 7
        # For 4x4x4, we can use 4x4 = 2x2 blocks, giving rank 7^2 = 49
        # For 3x4x7, it's more complex
        
        # Let's try a different structural approach:
        # Use the "lattice" method with a specific group structure
        
        # For rank r, try to find U, V, W with entries in {0, ±1, ±1/2}
        # such that the tensor equation is satisfied
        
        # Use a "greedy" approach: start with the schoolbook and try to
        # merge terms by finding linear dependencies
        
        # For each pair of schoolbook terms, check if they can be merged
        # (i.e., if there's a rank-2 decomposition of their sum)
        
        # This is getting complex. Let's try a more direct approach:
        # Use a "basis" of structured tensors and solve for coefficients
        
        # Define a set of "template" rank-1 tensors and find a linear combination
        # that equals the multiplication tensor
        
        # Template: for each (i0, j0, k0), define a "shifted" Kronecker product
        # T_{i0,j0,k0}[i,j,k] = delta_{i,i0} * delta_{j,j0} * delta_{k,k0}
        # This is just the schoolbook basis, so we need something more clever
        
        # Try: use "product" templates
        # T_{a,b,c}[i,j,k] = f_a(i) * g_b(j) * h_c(k)
        # where f_a, g_b, h_c are structured functions
        
        # For example:
        # f_a(i) = 1 if i == a else 0 (delta)
        # g_b(j) = 1 if j == b else 0 (delta)
        # h_c(k) = 1 if k == c else 0 (delta)
        # This gives the schoolbook decomposition
        
        # More interesting:
        # f_a(i) = (-1)^{i*a} (character of Z_3)
        # But we need real/half-integer values, so use ±1 patterns
        
        # For n=3: use patterns [1,1,1], [1,-1,0], [0,1,-1] (Hadamard-like for 3)
        # For m=4: use H_4 patterns
        # For p=7: use structured ±1 patterns
        
        # Let's define structured basis vectors:
        # For n=3: B_3 = [[1,1,1], [1,-1,0], [1,0,-1]] (not orthogonal but structured)
        # For m=4: B_4 = H_4 (Hadamard 4x4)
        # For p=7: B_7 = structured ±1 matrix
        
        # Then: U[i,j,r] = B3[i, r%n] * B4[j, r%m]
        # V[j,k,r] = B4[j, r%m] * B7[k, r%p]
        # W[k,i,r] = B7[k, r%p] * B3[i, r%n]
        
        # Check if this works... it's a specific ansatz worth trying
        
        B3 = np.array([
            [1, 1, 1],
            [1, -1, 0],
            [1, 0, -1]
        ], dtype=float)
        
        B4 = np.array([
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1]
        ], dtype=float)
        
        # For p=7, use a 7x7 structured matrix
        # Use a "cyclic" pattern: B7[k, r] = 1 if k == r else (-1 if (k-r) % 7 == 1 or (k-r) % 7 == 6 else 0)
        # Or simpler: use a random ±1 matrix with some structure
        B7 = np.ones((7, 7))
        for k in range(7):
            for r in range(7):
                if k != r:
                    B7[k, r] = 0.0
        # This is just identity, not useful. Let's use a different structure.
        # Use: B7[k, r] = 1 if (k - r) % 7 in {0, 1, 2} else -1
        B7 = np.zeros((7, 7))
        for k in range(7):
            for r in range(7):
                diff = (k - r) % 7
                if diff in [0, 1, 2]:
                    B7[k, r] = 1.0
                else:
                    B7[k, r] = -1.0
        
        # Try this ansatz with various rank values
        for rank_attempt in range(12, 30):
            if time.time() - start_time > time_limit:
                return None
            
            # The ansatz above has a fixed structure with period (3,4,7) = 84
            # So the natural rank is 84. We need to find a way to reduce it.
            
            # Try: use a "compressed" version where we identify equivalent patterns
            # If two rank slots have the same (or related) patterns, we can merge them
            
            # For now, just try the direct construction with rank = lcm(3,4,7) = 84
            # and see if we can find a smaller rank by adjusting the basis matrices
            
            pass
        
        return None
    
    def try_strassen_blocks(rank, time_limit):
        """
        Use Strassen's algorithm structure to build a decomposition.
        For 3x4x7, exploit the 4=2*2 structure in the middle dimension.
        """
        if time.time() - start_time > time_limit:
            return None
        
        # Strassen's algorithm for 2x2x2 uses 7 rank-1 terms:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # For our 3x4x7 problem:
        # A is 3x4, B is 4x7
        # Split the 4 middle indices into two groups: {0,1} and {2,3}
        # A = [A_left | A_right] where A_left is 3x2, A_right is 3x2
        # B = [B_top; B_bottom] where B_top is 2x7, B_bottom is 2x7
        
        # Product: C = A_left @ B_top + A_right @ B_bottom
        
        # Each 3x2x7 multiplication can be done with rank 3*2*7 = 42 (schoolbook)
        # But we can do better using Strassen-like ideas
        
        # For 3x2x7: split the 3 rows into {0,1} and {2}
        # A_left = [A11; A21] where A11 is 2x2, A21 is 1x2
        # B_top = [B11 | B12] where B11 is 2x7... wait, B_top is 2x7
        
        # This is getting complicated. Let me try a simpler approach.
        
        # Direct construction using the "block" structure:
        # For the 4 middle indices, use 2 groups of 2
        # For each group, use a 3x2x7 multiplication
        # Use Strassen for the 2x2 parts within each 3x2x7
        
        # Actually, let's just try to construct a rank-42 decomposition
        # by using 7 Strassen terms for each of the 6 "3x2x7" sub-problems
        # (3 row-groups x 2 col-groups... no, this doesn't quite work)
        
        # Let me try a different structural approach:
        # Use the "tensor rank" of the multiplication tensor
        # and try to find a decomposition using known algebraic identities
        
        # For now, return None and rely on the other methods
        return None
    
    # Try the Kronecker ansatz
    result = try_kronecker_ansatz(84, DEADLINE - 10)
    if result is not None:
        U, V, W = result
        best_rank = len(U[0]) if len(U) > 0 else rank_sb
        best_U = U
        best_V = V
        best_W = W
    
    # If we haven't improved, try the Strassen block approach
    if best_rank == rank_sb:
        result = try_strassen_blocks(84, DEADLINE - 10)
        if result is not None:
            U, V, W = result
            best_rank = len(U[0]) if len(U) > 0 else rank_sb
            best_U = U
            best_V = V
            best_W = W
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
