import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strategy:
    # We want to minimize the rank of the tensor decomposition for C = A * B
    # where A is 2x3 and B is 3x7.
    # C is 2x7.
    #
    # Let's look at the structure.
    # C[i, j] = sum_{k=0}^{2} A[i, k] * B[k, j]
    #
    # We can view this as a sum of rank-1 terms.
    # The trivial decomposition has rank n*m*p = 2*3*7 = 42.
    #
    # Can we do better?
    # Let's try to exploit linearity.
    #
    # Consider splitting the summation index k.
    #
    # Let's try a "block" approach similar to Strassen but for rectangular matrices.
    #
    # Let's look at the specific dimensions: 2x3 and 3x7.
    #
    # One idea: Group the columns of B.
    # B has 7 columns.
    # A has 3 rows.
    #
    # Let's try to reduce the rank by combining terms.
    #
    # Let's use a known algorithm for matrix multiplication.
    # For small matrices, specific decompositions are often better than generic Strassen.
    #
    # Let's try to construct a decomposition with rank < 42.
    #
    # Let's try rank 39 (as in the current program).
    # The current program uses 39.
    #
    # Can we go lower?
    #
    # Let's analyze the tensor T_{ijk} = delta_{ik} (for A) and delta_{kj} (for B)?
    # No, the tensor is T_{i, j, k} = A_{ik} B_{kj}.
    # The multiplication tensor is M_{n,m,p}.
    #
    # Let's try to find a lower rank decomposition.
    #
    # Let's try a random search or a greedy approach to see if we can beat 39.
    # However, we need an exact decomposition.
    #
    # Let's stick to the structural approach.
    #
    # Let's try to optimize the 2x3 * 3x7 case specifically.
    #
    # Let's look at the 2x2 * 2x2 case (Strassen, rank 7).
    # 2x2 * 2x2 -> rank 7.
    #
    # Here we have 2x3 * 3x7.
    #
    # Let's split the 3 rows of B into 2+1.
    # B = [B_top (2x7); B_bottom (1x7)]
    # A = [A_left (2x2) | A_right (2x1)]
    #
    # C = A_left * B_top + A_right * B_bottom
    #
    # Term 1: A_left (2x2) * B_top (2x7).
    # Term 2: A_right (2x1) * B_bottom (1x7).
    #
    # For Term 2:
    # A_right is 2x1. B_bottom is 1x7.
    # C_part = A_right * B_bottom.
    # This is a rank-1 matrix outer product scaled by the single element of B_bottom?
    # No, A_right has 2 elements. B_bottom has 7 elements.
    # C_part[i, j] = A_right[i] * B_bottom[j].
    # This is an outer product of a 2-vector and a 7-vector.
    # The rank of the tensor for 2x1 * 1x7 is 2*7 = 14?
    # Let's check.
    # T_{i, j} = u_i v_j.
    # We need to represent sum_{i,j} u_i v_j e_{ij}.
    # This is just the tensor u \otimes v.
    # The rank of the outer product tensor u \otimes v is 1 if u and v are fixed?
    # No, in matrix multiplication, the inputs are variables.
    # The tensor is \sum_{i,j} E_{ij} \otimes A_{i} \otimes B_{j}.
    # For 2x1 * 1x7:
    # Inputs: A (size 2), B (size 7). Output C (size 14).
    # C_{i,j} = A_i B_j.
    # Tensor T = \sum_{i=0}^1 \sum_{j=0}^6 A_i B_j E_{ij}.
    # This is a sum of 14 rank-1 terms.
    # Can we do better?
    # The rank of the 2x1x1x7 multiplication tensor is 2*7 = 14.
    # Why? Because the 14 outputs are linearly independent in terms of the inputs?
    # Actually, the rank of the multiplication tensor M_{n,m,p} is at least max(n, m, p) * ...?
    # For M_{2,1,7}, the rank is 14.
    # Proof: The 14 entries of C are all distinct bilinear forms A_i B_j.
    # They are linearly independent. So rank >= 14.
    # And we can achieve 14 with the trivial decomposition.
    # So Term 2 costs 14.
    #
    # For Term 1: A_left (2x2) * B_top (2x7).
    # This is 2x2 * 2x7.
    # We can split B_top into 3 blocks of 2 columns and 1 block of 1 column.
    # B_top = [B0 (2x2), B1 (2x2), B2 (2x2), B3 (2x1)]
    #
    # A_left * B0: 2x2 * 2x2. Rank 7 (Strassen).
    # A_left * B1: 2x2 * 2x2. Rank 7 (Strassen).
    # A_left * B2: 2x2 * 2x2. Rank 7 (Strassen).
    # A_left * B3: 2x2 * 2x1.
    #
    # What is the rank of 2x2 * 2x1?
    # C is 2x1.
    # C_0 = A_00 B_0 + A_01 B_1
    # C_1 = A_10 B_0 + A_11 B_1
    #
    # This is 4 bilinear terms.
    # Can we do it in less than 4?
    # The tensor is M_{2,2,1}.
    # Rank of M_{2,2,1}?
    # It is known that rank(M_{2,2,1}) = 4.
    # So Term 1 costs 7 + 7 + 7 + 4 = 25.
    #
    # Total rank = 25 + 14 = 39.
    #
    # This matches the current program.
    #
    # Can we do better?
    #
    # Maybe we can mix the terms?
    #
    # Let's check if rank(M_{2,2,1}) can be reduced if we share structure with the other blocks?
    # Unlikely, as they involve different columns of B.
    #
    # What if we split B differently?
    #
    # Let's try to optimize the 2x2 * 2x1 part.
    # Is it possible to do 2x2 * 2x1 in rank 3?
    # C_0 = A_00 B_0 + A_01 B_1
    # C_1 = A_10 B_0 + A_11 B_1
    #
    # If we use Strassen-like tricks?
    # M1 = (A_00 + A_11)(B_0 + B_1)
    # M2 = (A_10 + A_11) B_0
    # M3 = A_00 (B_1 - B_0)
    #
    # C_0 = M1 - M2 - M3 = (A_00+A_11)(B_0+B_1) - (A_10+A_11)B_0 - A_00(B_1-B_0)
    # = A_00 B_0 + A_00 B_1 + A_11 B_0 + A_11 B_1 - A_10 B_0 - A_11 B_0 - A_00 B_1 + A_00 B_0
    # = 2 A_00 B_0 + A_11 B_1 - A_10 B_0 ... No, this is getting messy.
    #
    # Actually, the rank of M_{2,2,1} is exactly 4.
    # Reference: "The rank of the matrix multiplication tensor"
    # M_{2,2,1} is a 2x2x1 tensor.
    # It is isomorphic to the tensor of multiplying a 2x2 matrix by a 2x1 vector.
    # The rank is 4.
    #
    # So 39 seems to be a hard lower bound for this specific splitting strategy.
    #
    # Can we change the splitting?
    #
    # What if we split A into 1x2 + 1x1?
    # A = [A_top (1x3); A_bottom (1x3)]
    # B = [B_left (3x6) | B_right (3x1)]? No, B is 3x7.
    #
    # Let's try splitting the output C.
    # C is 2x7.
    #
    # Let's try a different approach: Random search for a lower rank decomposition.
    # We can formulate this as a system of polynomial equations.
    # But that is hard.
    #
    # Let's try to verify if 39 is the minimum.
    #
    # Let's try to implement a search for rank 38 or lower.
    # We can use a greedy algorithm to assign terms.
    #
    # However, given the time constraint and the fact that 39 is a known good solution,
    # and the difficulty of proving a lower bound or finding a better one quickly,
    # I will stick with 39 but try to optimize the code for speed and clarity.
    #
    # Wait, is there a better decomposition for 2x2 * 2x7?
    #
    # Let's look at 2x2 * 2x2 again. Rank 7.
    # 2x2 * 2x1. Rank 4.
    #
    # What if we use a decomposition for 2x2 * 2x2 that shares terms with 2x2 * 2x1?
    #
    # Let's try to construct the whole 2x2 * 2x7 tensor.
    #
    # Let's try to use a library or a known result.
    #
    # Actually, let's try to improve the 2x1 * 1x7 part.
    # Rank 14.
    #
    # Is it possible to reduce the rank of the whole system by mixing?
    #
    # Let's try a simple heuristic:
    # Try to find a decomposition of rank R < 39.
    # We can set up a linear system for the coefficients of the rank-1 terms.
    #
    # Let the decomposition be:
    # T = sum_{r=1}^R u_r \otimes v_r \otimes w_r
    #
    # This means:
    # sum_{r=1}^R u_r[i,k] v_r[k,j] w_r[i,j] = delta_{ij} (if we view it as C_ij = sum A_ik B_kj)
    #
    # Actually, the tensor is:
    # T_{i,j,k} = delta_{ik} (for A part) and delta_{kj} (for B part)?
    # No.
    # The multiplication tensor M_{n,m,p} has entries T_{i,j,k} such that
    # C_{i,j} = sum_{k} A_{i,k} B_{k,j}.
    #
    # We want to write M_{n,m,p} = sum_{r=1}^R u_r \otimes v_r \otimes w_r.
    # Where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.
    #
    # This is equivalent to:
    # For all i,j,k:
    # sum_{r=1}^R u_r[i,m+k] v_r[k,p+j] w_r[j*n+i] = delta_{ik} (if k is the summation index)
    #
    # Wait, the indices are:
    # A is n x m. Index (i, k) -> i*m + k.
    # B is m x p. Index (k, j) -> k*p + j.
    # C is n x p. Index (i, j) -> i*p + j. (Or j*n + i as per contract W[k*n+i, r] ?)
    #
    # Contract says: W[k*n+i, r].
    # So C index is k*n + i?
    # "output factor follows the official trace convention W[k*n+i,r]"
    # Usually C is n x p.
    # If C is 2x7, n=2, p=7.
    # Index in W is k*n + i.
    # This implies C is stored in column-major order? Or i is row, k is column?
    # If C is n x p, usually C[i,j].
    # If index is k*n + i, and k goes 0..p-1, i goes 0..n-1.
    # Then W[k*n+i] corresponds to C[i,k].
    # So C is stored column-major.
    #
    # Let's check the current program.
    # c00 = k_start * 2 + 0.
    # Here k_start is the column index of C?
    # In the current program:
    # c00 = k_start * 2 + 0.
    # k_start is 0, 2, 4.
    # So c00 is 0, 4, 8.
    # If C is 2x7, and we are looking at column k_start.
    # Row 0, Col k_start.
    # If index = col * n + row, then index = k_start * 2 + 0.
    # This matches W[k*n+i] if k is column and i is row.
    # So W index = col * n + row.
    #
    # Okay, let's stick to the current program's logic which is known to be valid (score 1.0769).
    #
    # To beat 1.0769, we need rank < 39.
    #
    # Let's try rank 38.
    #
    # We can try to solve the system for rank 38.
    #
    # We have variables:
    # U: 6x38
    # V: 21x38
    # W: 14x38
    #
    # Total variables: 6*38 + 21*38 + 14*38 = 41*38 = 1558.
    #
    # Equations:
    # For each i in 0..1, j in 0..6, k in 0..2:
    # sum_{r=1}^{38} U[i*3+k, r] * V[k*7+j, r] * W[j*2+i, r] = delta_{ik} (if k is the summation index)
    #
    # Wait, the tensor equation is:
    # sum_{r} U_{i,k}^{(r)} V_{k,j}^{(r)} W_{i,j}^{(r)} = delta_{ik} ??
    # No.
    # The tensor T has components T_{i,j,k} = 1 if i==k? No.
    #
    # The multiplication tensor is defined by:
    # T(A, B) = C.
    # C_{i,j} = sum_{k} A_{i,k} B_{k,j}.
    #
    # The tensor T is an element of (R^{n*m})^* \otimes (R^{m*p})^* \otimes (R^{n*p}).
    # Or rather, we want to decompose the trilinear form.
    #
    # The condition for a rank-R decomposition is:
    # sum_{r=1}^R U_r \otimes V_r \otimes W_r = T
    #
    # Where T is the tensor with components:
    # T_{i*m+k, k*p+j, i*p+j} = 1 for all i,j,k.
    # And 0 otherwise.
    #
    # So we need:
    # sum_{r=1}^R U_{i*m+k, r} V_{k*p+j, r} W_{i*p+j, r} = 1 if the indices match the pattern, 0 otherwise.
    #
    # Specifically, for a fixed r, the term U_r \otimes V_r \otimes W_r contributes
    # U_{a,r} V_{b,r} W_{c,r} to T_{a,b,c}.
    #
    # We need the sum to be 1 for (a,b,c) = (i*m+k, k*p+j, i*p+j) and 0 otherwise.
    #
    # Note that a, b, c are not independent.
    # a = i*m+k.
    # b = k*p+j.
    # c = i*p+j.
    #
    # For a given a, b, c, there is at most one (i,j,k) that satisfies these.
    # If such a triple exists, the target is 1. Else 0.
    #
    # So we have a system of equations.
    # Number of non-zero entries in T: n*m*p = 2*3*7 = 42.
    #
    # So we have 42 equations.
    # Variables: R * (n*m + m*p + n*p) = R * (6 + 21 + 14) = 41 R.
    #
    # For R=39, we have 1599 variables and 42 equations.
    # This is a very underdetermined system.
    #
    # The challenge is to find integer/half-integer solutions.
    #
    # The current program finds a solution with R=39.
    #
    # Can we find one with R=38?
    #
    # We can use a random search.
    # Initialize U, V, W with small random integers.
    # Then solve for the residuals?
    #
    # Or, we can use the fact that the system is bilinear.
    #
    # Let's try a simple local search or random restarts.
    #
    # Given the time limit, I will implement a quick random search for R=38.
    # If it finds a solution, great. If not, return the R=39 solution.
    #
    # However, random search for integer solutions to polynomial systems is hard.
    #
    # Let's stick to the R=39 solution but ensure it is correct and fast.
    #
    # Actually, let's try to see if we can reduce the rank of the 2x2*2x1 part.
    #
    # Wait, I recall that the rank of M_{2,2,1} is 4.
    #
    # What about M_{2,3,7}?
    #
    # Let's check if there is a known result for M_{2,3,7}.
    #
    # I will output the R=39 solution as it is reliable.
    #
    # To be safe, I will add a deadline guard.
    
    start_time = time.time()
    deadline = start_time + 100  # 100s limit
    
    # Return the known R=39 solution
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, signs=None):
        nonlocal col
        if signs is None:
            signs = [1] * len(u_indices)
        for idx, s in zip(u_indices, signs):
            U[idx][col] += s
        for idx in v_indices:
            V[idx][col] += 1
        for idx in w_indices:
            W[idx][col] += 1
        col += 1

    def add_strassen_block(k_start):
        # A indices (2x2 part of 2x3 A):
        # A00: 0*3+0 = 0
        # A01: 0*3+1 = 1
        # A10: 1*3+0 = 3
        # A11: 1*3+1 = 4
        
        # B indices (2x2 part of 3x7 B, rows 0-1, cols k_start, k_start+1):
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        # C indices (2x2 part of 2x7 C, rows 0-1, cols k_start, k_start+1):
        # C index = k*2 + i (col * n + row)
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        add_term([0, 4], [b00, b11], [c00, c11])
        
        # M2: (A10+A11)B00 -> C10, -C11
        add_term([3, 4], [b00], [c10, c11])
        W[c11][col-1] = -1
        
        # M3: A00(B01-B11) -> C01, C11
        add_term([0], [b01, b11], [c01, c11])
        V[b11][col-1] = -1
        
        # M4: A11(B10-B00) -> C00, C10
        add_term([4], [b10, b00], [c00, c10])
        V[b00][col-1] = -1
        
        # M5: (A00+A01)B11 -> -C00, C01
        add_term([0, 1], [b11], [c00, c01])
        W[c00][col-1] = -1
        
        # M6: (A10-A00)(B00+B01) -> C11
        add_term([3, 0], [b00, b01], [c11])
        U[0][col-1] = -1
        
        # M7: (A01-A11)(B10+B11) -> C00
        add_term([1, 4], [b10, b11], [c00])
        U[4][col-1] = -1

    # 3 Strassen blocks for cols 0-1, 2-3, 4-5
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Block for col 6 of B (rows 0-1)
    # A00 B06 -> C06
    # A01 B16 -> C06
    # A10 B06 -> C16
    # A11 B16 -> C16
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([0], [b06], [c06])
    add_term([1], [b16], [c06])
    add_term([3], [b06], [c16])
    add_term([4], [b16], [c16])
    
    # Part for A[:, 2] * B[2, :]
    # A02 B2k -> C0k
    # A12 B2k -> C1k
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        add_term([2], [b2k], [c0k])
        add_term([5], [b2k], [c1k])
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
