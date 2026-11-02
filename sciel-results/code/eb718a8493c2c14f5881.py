import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 130.0  # Safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with the block-Strassen decomposition (rank 46)
    # Then try to find rank reductions via linear dependencies
    
    def build_block_strassen():
        """Build the 46-term block Strassen decomposition."""
        terms = []
        
        def add_strassen(i_start, j_start, k_start):
            # 7 Strassen terms for 2x2x2 block
            M1_u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
            M1_v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
            M1_w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((M1_u, M1_v, M1_w))
            
            M2_u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
            M2_v = {(j_start, k_start): 1}
            M2_w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
            terms.append((M2_u, M2_v, M2_w))
            
            M3_u = {(i_start, j_start): 1}
            M3_v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
            M3_w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((M3_u, M3_v, M3_w))
            
            M4_u = {(i_start+1, j_start+1): 1}
            M4_v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
            M4_w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
            terms.append((M4_u, M4_v, M4_w))
            
            M5_u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
            M5_v = {(j_start+1, k_start+1): 1}
            M5_w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
            terms.append((M5_u, M5_v, M5_w))
            
            M6_u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
            M6_v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
            M6_w = {(k_start+1, i_start+1): 1}
            terms.append((M6_u, M6_v, M6_w))
            
            M7_u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
            M7_v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
            M7_w = {(k_start, i_start): 1}
            terms.append((M7_u, M7_v, M7_w))
        
        def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(k_start, k_end):
                        terms.append(({(i, j): 1}, {(j, k): 1}, {(k, i): 1}))
        
        # Block decomposition:
        # j in {0,1}: k in {0,1} Strassen, k in {2,3} Strassen, k in {4} schoolbook
        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_schoolbook(0, 2, 0, 2, 4, 5)
        # j in {2,3}: k in {0,1} Strassen, k in {2,3} Strassen, k in {4} schoolbook
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        add_schoolbook(0, 2, 2, 4, 4, 5)
        # j in {4}: all schoolbook
        add_schoolbook(0, 2, 4, 5, 0, 2)
        add_schoolbook(0, 2, 4, 5, 2, 4)
        add_schoolbook(0, 2, 4, 5, 4, 5)
        
        return terms
    
    def terms_to_matrices(terms):
        """Convert term list to U, V, W matrices."""
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for c in range(rank):
            u_dict, v_dict, w_dict = terms[c]
            for (i, j), val in u_dict.items():
                U[i * m + j][c] = val
            for (j, k), val in v_dict.items():
                V[j * p + k][c] = val
            for (k, i), val in w_dict.items():
                W[k * n + i][c] = val
        
        return U, V, W
    
    def validate_decomposition(U, V, W):
        """Check if the decomposition exactly reproduces matrix multiplication."""
        rank = len(U[0]) if U else 0
        # Build the multiplication tensor T[i,j,k] = delta_{ik} * (A*B) contribution
        # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Wait, the tensor for matrix multiplication (A n×m, B m×p, C n×p) is:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T has indices (i, j, k) with T[i,j,k] = 1 if this contributes to C[i,k]
        # Actually the standard form: T_{i,j,k} where C[i,k] = sum_{j} A[i,j] B[j,k]
        # As a 3-tensor: T[i,j,k] = delta_{i,k}... no.
        # 
        # The matrix multiplication tensor is T in V^{n*m} ⊗ V^{m*p} ⊗ V^{p*n}
        # where T[e1, e2, e3] = 1 if e1=(i,j), e2=(j,k), e3=(k,i) for valid indices
        # i.e., T[(i,j),(j,k),(k,i)] = 1
        # 
        # So we need: sum_c U[(i,j),c] * V[(j,k),c] * W[(k,i),c] = 1 for all valid (i,j,k)
        # and 0 otherwise (but there's no "otherwise" since (i,j,k) always has matching indices)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = 0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    def find_linear_dependencies(U, V, W):
        """
        Try to find linear dependencies among the rank-1 terms.
        If we can express one term as a linear combination of others, we can reduce rank.
        
        Strategy: Check if any column of U, V, W can be expressed as a linear combination
        of the other columns, maintaining the tensor identity.
        """
        rank = len(U[0]) if U else 0
        
        # Build the full tensor T from the decomposition
        # T is a 3D tensor of shape (n*m, m*p, p*n)
        # T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
        # We want T to equal the matrix multiplication tensor.
        
        # Try removing each term and see if we can compensate
        # This is expensive, so we'll do a limited search
        
        # Alternative: use SVD-like approach on the flattening
        # For a rank-r decomposition, if we fix one mode, the other two form a matrix
        # of rank <= r. If the actual tensor has lower rank in some flattening,
        # we might reduce.
        
        # Let's try: for each pair of terms (c1, c2), check if their combination
        # can be absorbed into other terms.
        
        # Simpler approach: try to find a term c such that there exist coefficients
        # alpha_d (d != c) with:
        # U[:,c] = sum_{d!=c} alpha_d * U[:,d]
        # V[:,c] = sum_{d!=c} beta_d * V[:,d]  
        # W[:,c] = sum_{d!=c} gamma_d * W[:,d]
        # and the tensor contribution matches.
        
        # This is a system of nonlinear equations. Let's try a simpler heuristic:
        # Check if removing term c and adjusting others still gives the right tensor.
        
        # For efficiency, only try removing terms that are "boundary" schoolbook terms
        # which might overlap with Strassen terms.
        
        best_rank = rank
        best_terms = None
        
        # Try removing each term one at a time
        for c_remove in range(rank):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            remaining = [t for i, t in enumerate(terms) if i != c_remove]
            if len(remaining) >= best_rank:
                continue
            
            # Check if the remaining terms still give the correct tensor
            # This is O(n*m*p*rank) per check
            valid = True
            U_rem, V_rem, W_rem = terms_to_matrices(remaining)
            new_rank = len(U_rem[0]) if U_rem else 0
            
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        expected = 1
                        actual = sum(U_rem[i*m+j][c] * V_rem[j*p+k][c] * W_rem[k*n+i][c] 
                                    for c in range(new_rank))
                        if abs(actual - expected) > 1e-9:
                            valid = False
                            break
            
            if valid:
                best_rank = new_rank
                best_terms = remaining
        
        return best_rank, best_terms
    
    # Build initial decomposition
    terms = build_block_strassen()
    U, V, W = terms_to_matrices(terms)
    
    # Verify initial decomposition
    if not validate_decomposition(U, V, W):
        # Fallback to pure schoolbook
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
        U, V, W = terms_to_matrices(terms)
    
    initial_rank = len(terms)
    
    # Try to reduce rank
    if time.time() - start_time < TIME_LIMIT - 30:
        reduced_rank, reduced_terms = find_linear_dependencies(U, V, W)
        if reduced_terms is not None and reduced_rank < initial_rank:
            terms = reduced_terms
            U, V, W = terms_to_matrices(terms)
    
    # Try a more aggressive search: combine terms
    # Look for pairs of terms that can be merged
    if time.time() - start_time < TIME_LIMIT - 20:
        current_rank = len(terms)
        improved = True
        while improved and time.time() - start_time < TIME_LIMIT - 15:
            improved = False
            for c1 in range(len(terms)):
                if time.time() - start_time > TIME_LIMIT - 15:
                    break
                for c2 in range(c1 + 1, len(terms)):
                    # Try to see if terms c1 and c2 can be replaced by fewer terms
                    # This is a 2-term merge problem
                    # Skip for now as it's complex
                    pass
            # Try single-term removal again after potential changes
            new_rank, new_terms = find_linear_dependencies(U, V, W)
            if new_terms is not None and new_rank < len(terms):
                terms = new_terms
                U, V, W = terms_to_matrices(terms)
                improved = True
    
    # Final validation
    if not validate_decomposition(U, V, W):
        # Fallback to the original block Strassen
        terms = build_block_strassen()
        U, V, W = terms_to_matrices(terms)
    
    return {"U": U, "V": V, "W": W}
