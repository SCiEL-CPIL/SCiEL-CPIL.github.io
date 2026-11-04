"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # We want T[i, j, k] = 1 for all i,j,k? No, the tensor for matmul has T[i,j,k]=1 if indices match?
    # Wait, the tensor for matrix multiplication C = A B is T_{i,j,k} = A_{i,j} B_{j,k}.
    # In the decomposition sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}, the target tensor is the standard matmul tensor.
    # The entries of the standard matmul tensor T are 1 if (i,j,k) is a valid index triple?
    # Actually, the tensor is defined such that <T, A \otimes B> = C.
    # The basis vectors are e_i \otimes e_j \otimes e_k.
    # The coefficient of e_i \otimes e_j \otimes e_k in the matmul tensor is 1 if we view A_{i,j} and B_{j,k} producing C_{i,k}.
    # So T[i, j, k] = 1 for all i, j, k? No.
    # Let's check the seed code's verify:
    # s = sum U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # if abs(s - 1.0) > 1e-9: return False
    # This implies the target tensor has 1s everywhere?
    # That would mean we are decomposing the tensor of all ones?
    # No, matrix multiplication tensor T_{ijk} is 1 if the indices are consistent?
    # Let's re-read the standard definition.
    # The tensor for matrix multiplication is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k ? No.
    # The multiplication map is mu: A \otimes B -> C.
    # mu(A \otimes B)_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T is defined by T(A, B) = C.
    # T = sum_{i,j,k} e_i \otimes e_j \otimes e_k is the all-ones tensor.
    # The matmul tensor is T_matmul = sum_{i,j,k} e_i \otimes e_j \otimes e_k ?
    # Let's check: (sum_{i,j,k} e_i \otimes e_j \otimes e_k)(A, B) = sum_{i,j,k} A_{ij} B_{jk} e_i ? No, the third index is k in B?
    # Usually indices are i (row of A), j (col of A / row of B), k (col of B).
    # T_{ijk} = 1.
    # Yes, the matrix multiplication tensor has 1s in all entries T[i, j, k].
    # Why? Because A_{ij} B_{jk} contributes to C_{ik}.
    # The tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k.
    # So the verify function checking for 1.0 is correct for the standard matmul tensor.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _rank_reduce(U, V, W, n, m, p, max_iter=200, seed=42):
    """
    Attempt to reduce the rank of the decomposition by finding linear dependencies.
    Strategy:
    1. Randomly mix columns of U, V, W using invertible matrices P, Q, R.
       U' = U P, V' = V Q, W' = W R where P, Q, R are small random perturbations?
       Actually, if we apply a change of basis in the rank space:
       U' = U S, V' = V S^{-1}, W' = W I? No.
       The decomposition is sum_c u_c v_c w_c.
       If we let u'_c = sum_d u_d S_{dc}, v'_c = sum_d v_d (S^{-1})_{cd}, w'_c = w_c?
       Then sum_c u'_c v'_c w_c = sum_c (sum_d u_d S_{dc}) (sum_e v_e (S^{-1})_{ce}) w_c
       = sum_{d,e} u_d v_e w_e (sum_c S_{dc} (S^{-1})_{ce}) = sum_{d,e} u_d v_e w_e delta_{de} = sum_d u_d v_d w_d.
       This preserves the tensor but changes the factors.
       
       Better approach:
       We can add a column from one set to another if it creates a dependency?
       
       Let's use the "column pruning" method.
       1. Start with a valid decomposition (e.g., schoolbook or Strassen).
       2. Apply a random invertible transformation T to the rank space:
          U_new = U T
          V_new = V T^{-1}
          W_new = W
       3. Check if any column of U_new is a linear combination of others?
          If a column of U_new is zero, we can drop it.
          More generally, if the columns of U_new are linearly dependent, we can reduce rank.
          However, we need the whole tensor to remain valid.
          
          A more robust method for rank reduction in tensor decompositions:
          - View the decomposition as a set of rank-1 terms.
          - If we can find a linear relation among the terms, we can eliminate one.
          
          Let's try a simpler heuristic:
          1. Start with schoolbook (rank 96).
          2. Randomly perturb the factors slightly? No, must be exact.
          
          Let's stick to the known Strassen construction (rank 88) as the baseline,
          and then try to improve it.
          
          Actually, let's implement a search for a rank < 88 decomposition.
          We can use the fact that the tensor is small (3x4x8 = 96 entries).
          We can set up a system of linear equations for the entries of U, V, W given a target rank R.
          But R is unknown.
          
          Alternative: Use the "border rank" or "rank" algorithms?
          For small tensors, we can try to solve for a decomposition of rank R by fixing R and solving the bilinear system.
          This is hard.
          
          Let's try to construct a specific lower rank decomposition.
          Known results:
          2x2x2: Rank 7 (Strassen).
          3x3x3: Rank 23 (Landsberg), 20 (unknown?), 19? No, 3x3x3 is 23.
          3x4x8:
          Schoolbook: 96.
          Strassen blocks:
          Split 3x4x8 into blocks.
          i: 3 -> 2+1
          j: 4 -> 2+2
          k: 8 -> 2+2+2+2
          
          Blocks:
          (2,2,2) x 8 blocks?
          i=2, j=2, k=2.
          Number of such blocks:
          i splits: 2 (size 2), 1 (size 1).
          j splits: 2 (size 2), 2 (size 2).
          k splits: 4 (size 2).
          
          Combinations:
          1. i=2, j=2, k=2: 1 * 2 * 4 = 8 blocks of size 2x2x2.
             Each rank 7. Total 8 * 7 = 56.
          2. i=1, j=2, k=2: 1 * 2 * 4 = 8 blocks of size 1x2x2.
             1x2x2 is just 2x2 matrix multiplication? No, 1x2 * 2x2 = 1x2.
             Rank of 1x2x2 tensor?
             T[i, j, k] = 1 for i=0, j in {0,1}, k in {0,1}.
             This is the tensor for multiplying a 1x2 vector by a 2x2 matrix.
             Rank?
             Terms:
             A_{0,0} B_{0,0} -> C_{0,0}
             A_{0,1} B_{1,0} -> C_{0,0}
             A_{0,0} B_{0,1} -> C_{0,1}
             A_{0,1} B_{1,1} -> C_{0,1}
             This is 4 terms. Can we do better?
             It's equivalent to 2x2 matrix multiplication but with one row of A fixed?
             Actually, 1x2x2 tensor rank is 4?
             Let's check.
             U (1*2=2 rows), V (2*2=4 rows), W (2*1=2 rows).
             Rank 4 schoolbook.
             Is rank 3 possible?
             Unlikely for 1x2x2.
             So 8 * 4 = 32.
             
          Total rank = 56 + 32 = 88.
          This matches the current program.
          
          Can we do better?
          Maybe mix the blocks?
          Or use a different splitting?
          
          Let's try to verify if rank 87 or lower is possible.
          Given the time, I will implement the rank 88 solution robustly and add a search for linear dependencies to see if we can drop a rank.
          
          Search strategy:
          1. Construct rank 88 decomposition.
          2. Randomly mix the 88 columns using an invertible matrix T (88x88).
             U' = U T, V' = V T^{-1}, W' = W.
          3. Check if any column of U' is zero. If so, drop it.
          4. If not, check if columns are linearly dependent.
             If U' has rank < 88, we can reduce the rank.
             However, we must ensure the tensor remains valid.
             If U' has rank r < 88, we can pick a basis for the columns of U' and express the others in terms of it.
             Then we can eliminate the dependent columns by adjusting V' and W'?
             This is tricky because V' and W' are coupled.
             
          Actually, if U' = U T, V' = V T^{-1}, W' = W, the tensor is preserved.
          If U' has linearly dependent columns, say u'_c = sum_{d != c} alpha_d u'_d,
          then we can set u'_c = 0 and adjust v'_c, w'_c?
          No, the term u'_c v'_c w'_c must be preserved.
          If u'_c is a combination of others, we can't just drop it unless we can absorb it.
          
          Let's try a different approach:
          Use the known optimal ranks for sub-blocks if available.
          For 1x2x2, rank is 4.
          For 2x2x2, rank is 7.
          
          Is there a better decomposition for 3x4x8?
          Maybe 3x4x8 can be viewed as 3x4x4 x 2?
          Or 3x2x8 x 2?
          
          Let's try to construct a rank 87 decomposition by finding a dependency.
          I will implement a simple search:
          1. Start with rank 88.
          2. For 100 iterations:
             a. Generate random T.
             b. U' = U T, V' = V T^{-1}, W' = W.
             c. Compute rank of U'.
             d. If rank(U') < 88, try to reduce.
                To reduce, we need to find a null vector in the column space of U'.
                Let x be such that U' x = 0.
                Then sum_c U'_c x_c = 0.
                This implies sum_c (U T)_c x_c = 0.
                This doesn't directly help drop a column unless x has a 1 in some position and 0 elsewhere?
                
          Given the complexity and time limit, I will stick to the robust rank 88 construction and ensure it is correct.
          If I can quickly find a rank 87, I will use it.
          
          Let's double check the 1x2x2 rank.
          Tensor T_{j,k} for i=0.
          T_{0,0} = A_{0,0} B_{0,0}
          T_{0,1} = A_{0,1} B_{1,0}
          T_{1,0} = A_{0,0} B_{0,1}
          T_{1,1} = A_{0,1} B_{1,1}
          
          This is just 4 independent terms.
          U: (2, R), V: (4, R), W: (2, R).
          If R=3, can we do it?
          Likely not.
          
          So 88 seems to be the barrier for this block decomposition.
          
          I will return the rank 88 solution.
          """
    return None

def _construct_strassen_base():
    n, m, p = 3, 4, 8
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
        ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
        ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
        ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
        ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
        ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
    ]
    
    def add_block(i0, j0, k0, i_size, j_size, k_size):
        nonlocal col
        if i_size == 2 and j_size == 2 and k_size == 2:
            for t in range(7):
                a_coeffs = terms[t][0]
                b_coeffs = terms[t][1]
                c_coeffs = terms[t][2]
                
                for di in range(2):
                    for dj in range(2):
                        U[(i0+di)*m + (j0+dj)][col] = a_coeffs[di*2+dj]
                
                for dj in range(2):
                    for dk in range(2):
                        V[(j0+dj)*p + (k0+dk)][col] = b_coeffs[dj*2+dk]
                
                for di in range(2):
                    for dk in range(2):
                        W[(k0+dk)*n + (i0+di)][col] = c_coeffs[di*2+dk]
                
                col += 1
        else:
            for di in range(i_size):
                for dj in range(j_size):
                    for dk in range(k_size):
                        U[(i0+di)*m + (j0+dj)][col] = 1
                        V[(j0+dj)*p + (k0+dk)][col] = 1
                        W[(k0+dk)*n + (i0+di)][col] = 1
                        col += 1

    # 8 blocks of 2x2x2 (Strassen)
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                add_block(i0, j0, k0, 2, 2, 2)
                
    # 8 blocks of 1x2x2 (Schoolbook)
    for i0 in [2]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                add_block(i0, j0, k0, 1, 2, 2)
                
    return U, V, W, col

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Try rank reduction search?
    # For now, use the robust rank 88 construction.
    U, V, W, rank = _construct_strassen_base()
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
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
