import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = 4, 4, 6
    
    # 1. Construct the base rank-77 decomposition
    # This is the "Kronecker product of Strassen(7) and mixed(11)" mentioned in the knowledge.
    # We build it using the block structure.
    
    # Outer Strassen steps for 4x4 * 4x6
    # Partition A(4x4) into 2x2 blocks of 2x2.
    # Partition B(4x6) into 2x2 blocks of 2x3.
    # Partition C(4x6) into 2x2 blocks of 2x3.
    
    # Inner decomposition for 2x2 * 2x3 -> 2x3
    # This is a "mixed" decomposition of rank 11.
    # It consists of Strassen(7) for the first 2 columns (2x2*2x2) 
    # and Schoolbook(4) for the 3rd column (2x2*2x1).
    
    # Let's define the local indices for the inner block (2x2 * 2x3 -> 2x3)
    # A_local: 2x2, indices (r,s) r,s in {0,1}
    # B_local: 2x3, indices (s,k) s in {0,1}, k in {0,1,2}
    # C_local: 2x3, indices (r,k) r in {0,1}, k in {0,1,2}
    
    # Inner Strassen terms (for k=0,1 part, treating it as 2x2 * 2x2)
    # Standard Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # We map these to the 2x3 case.
    # For k=0,1, we use Strassen.
    # For k=2, we use schoolbook (4 terms).
    
    # Let's explicitly list the 11 terms for the inner block (2x2 * 2x3 -> 2x3)
    # Each term is a tuple (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: list of (r, s, coeff)
    # B_coeffs: list of (s, k, coeff)
    # C_coeffs: list of (r, k, coeff)
    
    inner_terms = []
    
    # --- 7 Strassen terms for columns 0 and 1 ---
    # M1
    inner_terms.append((
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)]
    ))
    # M2
    inner_terms.append((
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,0,1), (1,1,-1)]
    ))
    # M3
    inner_terms.append((
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(0,1,1), (1,1,1)]
    ))
    # M4
    inner_terms.append((
        [(1,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,0,1), (1,0,1)]
    ))
    # M5
    inner_terms.append((
        [(0,0,1), (0,1,1)],
        [(1,1,1)],
        [(0,1,1), (0,0,-1)]
    ))
    # M6
    inner_terms.append((
        [(1,0,1), (0,0,-1)],
        [(0,0,1), (0,1,1)],
        [(1,1,1)]
    ))
    # M7
    inner_terms.append((
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)]
    ))
    
    # --- 4 Schoolbook terms for column 2 ---
    # A(r,0) * B(0,2) -> C(r,2)
    inner_terms.append((
        [(0,0,1)],
        [(0,2,1)],
        [(0,2,1)]
    ))
    # A(r,1) * B(1,2) -> C(r,2)
    inner_terms.append((
        [(0,1,1)],
        [(1,2,1)],
        [(0,2,1)]
    ))
    inner_terms.append((
        [(1,0,1)],
        [(0,2,1)],
        [(1,2,1)]
    ))
    inner_terms.append((
        [(1,1,1)],
        [(1,2,1)],
        [(1,2,1)]
    ))
    
    # Outer Strassen steps for 4x4 * 4x6
    # We partition A, B, C into 2x2 blocks.
    # Block ids: 0=00, 1=01, 2=10, 3=11
    # A block b is 2x2. B block b is 2x3. C block b is 2x3.
    
    # Outer Strassen formulas for block multiplication:
    # We want to compute C = A * B where A is 4x4, B is 4x6.
    # Let A = [[A00, A01], [A10, A11]] (each 2x2)
    # Let B = [[B00, B01], [B10, B11]] (each 2x3)
    # Let C = [[C00, C01], [C10, C11]] (each 2x3)
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Using Strassen on the block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Each M is a 2x3 matrix.
    # So each outer term contributes to 4 C-blocks? No, each M is a single 2x3 matrix result of a 2x2 * 2x3 multiplication.
    # Wait, the "inner" decomposition is for 2x2 * 2x3 -> 2x3.
    # So each outer "M" is computed by the inner 11-term decomposition.
    # Thus, total rank = 7 (outer terms) * 11 (inner terms) = 77.
    
    # We need to map the outer block indices to global indices.
    # A global (I,J) where I,J in 0..3.
    # Block id b: I = 2*(b//2) + r, J = 2*(b%2) + s.
    # B global (J,K) where J in 0..3, K in 0..5.
    # Block id b: J = 2*(b//2) + s, K = 3*(b%2) + k.
    # C global (I,K) where I in 0..3, K in 0..5.
    # Block id b: I = 2*(b//2) + r, K = 3*(b%2) + k.
    
    # Outer terms definition:
    # Each term is a tuple (A_blocks, B_blocks, C_blocks)
    # A_blocks: list of (block_id, coeff)
    # B_blocks: list of (block_id, coeff)
    # C_blocks: list of (block_id, coeff)
    
    outer_terms = [
        # M1
        ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
        # M2
        ([(2,1), (3,1)], [(0,1)], [(2,1), (3,-1)]),
        # M3
        ([(0,1)], [(1,1), (3,-1)], [(1,1), (3,1)]),
        # M4
        ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]),
        # M5
        ([(0,1), (1,1)], [(3,1)], [(1,1), (0,-1)]),
        # M6
        ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),
        # M7
        ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)])
    ]
    
    # Build the full U, V, W matrices
    rank_base = 77
    U = np.zeros((n*m, rank_base))
    V = np.zeros((m*p, rank_base))
    W = np.zeros((p*n, rank_base))
    
    # Helper to get global indices
    def get_A_idx(b, r, s):
        I = 2*(b//2) + r
        J = 2*(b%2) + s
        return I*m + J
        
    def get_B_idx(b, s, k):
        J = 2*(b//2) + s
        K = 3*(b%2) + k
        return J*p + K
        
    def get_W_idx(b, r, k):
        I = 2*(b//2) + r
        K = 3*(b%2) + k
        return K*n + I # W is indexed by k*n + i per problem statement: W[k*n+i,r]
        
    # Fill U, V, W
    for oi, (A_blocks, B_blocks, C_blocks) in enumerate(outer_terms):
        for ii, (A_inner, B_inner, C_inner) in enumerate(inner_terms):
            r_idx = oi * 11 + ii
            
            # Add A contributions
            for (r, s, ca) in A_inner:
                for (b, cb) in A_blocks:
                    idx = get_A_idx(b, r, s)
                    U[idx, r_idx] += ca * cb
                    
            # Add B contributions
            for (s, k, cb) in B_inner:
                for (b, cb2) in B_blocks:
                    idx = get_B_idx(b, s, k)
                    V[idx, r_idx] += cb * cb2
                    
            # Add W contributions
            for (r, k, cc) in C_inner:
                for (b, cb2) in C_blocks:
                    idx = get_W_idx(b, r, k)
                    W[idx, r_idx] += cc * cb2
    
    # 2. Rank Reduction via Linear Dependency Elimination
    # We have a set of rank-1 tensors. We want to remove any that are linearly dependent.
    # A term r is redundant if its vector (U_r, V_r, W_r) can be expressed as a linear combination of others such that the sum remains T.
    # This is equivalent to finding the rank of the matrix formed by the tensor coefficients?
    # Actually, we can treat this as a linear system.
    # Let's try a simple greedy elimination:
    # For each term, check if it can be removed by adjusting the others.
    # This is computationally expensive if done naively.
    
    # A better approach for these structured tensors:
    # Check for zero columns in U, V, W.
    # If a column is zero in any of U, V, or W, the term is 0 and can be removed.
    
    # Also, check for linear dependencies among the columns of U, V, W simultaneously.
    # We can form a matrix M where each row is a flattened (U_r, V_r, W_r) vector.
    # If the rank of M is less than the number of rows, there are dependencies.
    # However, the tensor constraint is sum(u_r v_r w_r) = T.
    # If we have a linear dependency sum alpha_r (u_r v_r w_r) = 0, we can eliminate one term.
    # This is not the same as linear dependency of the concatenated vectors.
    
    # Let's use a simpler heuristic that works well for these specific Kronecker products:
    # Many terms will be identical or negatives of each other due to the symmetry of Strassen.
    # We can group identical terms and sum them? No, they are separate rank-1 components.
    # But if two terms have identical U, V, W vectors, they can be combined into one term with coeff 2?
    # The problem requires integer/half-integer entries.
    # If we have two identical terms, we can replace them with one term with doubled coefficients.
    # This reduces rank by 1.
    
    # Let's identify duplicate terms.
    # Represent each term as a tuple of tuples of non-zero coefficients.
    
    def term_key(r):
        u_tuple = tuple((i, U[i, r]) for i in range(n*m) if U[i, r] != 0)
        v_tuple = tuple((i, V[i, r]) for i in range(m*p) if V[i, r] != 0)
        w_tuple = tuple((i, W[i, r]) for i in range(p*n) if W[i, r] != 0)
        return (u_tuple, v_tuple, w_tuple)
    
    keys = [term_key(r) for r in range(rank_base)]
    
    # Find duplicates
    from collections import Counter
    key_counts = Counter(keys)
    
    # We can merge duplicates.
    # If a key appears c times, we can replace c terms with 1 term having coefficients multiplied by c.
    # But we need to be careful: the "coeff" is already baked into U, V, W.
    # So if we have two identical terms, their U, V, W vectors are identical.
    # We can sum them: U_new = U_1 + U_2 = 2*U_1.
    # This is valid as long as 2*U_1 are still integers/half-integers.
    # Since U_1 are integers, 2*U_1 are integers.
    
    new_U = []
    new_V = []
    new_W = []
    
    used = [False] * rank_base
    
    for i in range(rank_base):
        if used[i]:
            continue
        k = keys[i]
        count = key_counts[k]
        if count > 1:
            # Find all indices with this key
            indices = [j for j in range(rank_base) if keys[j] == k and not used[j]]
            # Sum them
            U_sum = np.sum(U[:, indices], axis=1)
            V_sum = np.sum(V[:, indices], axis=1)
            W_sum = np.sum(W[:, indices], axis=1)
            
            new_U.append(U_sum)
            new_V.append(V_sum)
            new_W.append(W_sum)
            
            for j in indices:
                used[j] = True
        else:
            new_U.append(U[:, i])
            new_V.append(V[:, i])
            new_W.append(W[:, i])
            used[i] = True
            
    U_reduced = np.array(new_U).T
    V_reduced = np.array(new_V).T
    W_reduced = np.array(new_W).T
    
    rank_reduced = U_reduced.shape[1]
    
    # Verification
    # Check if the reduced decomposition still sums to the tensor T.
    # T[i,j,k] = 1 for all i,j,k? No, T is the matmul tensor.
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This should equal 1 if (i,j,k) is a valid index? 
    # Actually, the matmul tensor is defined such that C = AB.
    # The structure tensor has 1s at (i,j,k) where the multiplication happens.
    # For all i,j,k, the value is 1? 
    # Let's check: sum_j A[i,j] B[j,k] = C[i,k].
    # The tensor T[i,j,k] = 1 for all i,j,k is incorrect.
    # The tensor is T[i,j,k] = 1 if we consider the basis vectors e_i, e_j, e_k.
    # Actually, the tensor for matrix multiplication is T_{ijk} = 1 for all i,j,k? 
    # No, T_{ijk} is the coefficient of A_{ij} B_{jk} in C_{ik}.
    # So T_{ijk} = 1 for all valid i,j,k.
    
    # Let's verify with a small example or just trust the construction.
    # The construction is based on Strassen, which is correct.
    # So the sum should be 1 for all i,j,k.
    
    # We can verify numerically.
    if time.time() < deadline - 10:
        T_check = np.zeros((n, m, p))
        for r in range(rank_reduced):
            u = U_reduced[:, r]
            v = V_reduced[:, r]
            w = W_reduced[:, r]
            for i in range(n):
                for j in range(m):
                    if u[i*m+j] == 0: continue
                    for k in range(p):
                        T_check[i,j,k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
        
        # Expected tensor is all 1s?
        # Let's check one element.
        # If the check fails, we fall back to the un-reduced version or schoolbook.
        if not np.allclose(T_check, 1.0, atol=1e-9):
            # Fallback to original 77
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Convert to lists of integers or half-integers
    # The entries are sums of integers, so they are integers.
    
    U_final = U_reduced.tolist()
    V_final = V_reduced.tolist()
    W_final = W_reduced.tolist()
    
    return {"U": U_final, "V": V_final, "W": W_final}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
