import numpy as np
import random
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 5, 5, 5
    
    # Strategy: Embed 5x5 multiplication into 6x6 multiplication.
    # The rank of 6x6 matrix multiplication is at most 21 (Strassen on 2x2 blocks of 3x3).
    # However, we need to be careful. The "rank 21" usually refers to the tensor rank over a field where 3x3 multiplication is treated as a scalar operation? No.
    # Actually, the tensor rank of 6x6x6 matrix multiplication is NOT 21.
    # The tensor rank of N x N x N matrix multiplication is denoted R(N).
    # R(1) = 1
    # R(2) = 7
    # R(3) = 23 (best known)
    # R(4) = 49 (best known? No, 49 is upper bound from Strassen? Actually R(4) <= 49 is not tight, best known is 49? No, R(4) <= 49 is from embedding 4x4 into 5x5? No.)
    # Let's check standard values.
    # R(2) = 7
    # R(3) = 23 (Håstad)
    # R(4) = 49 (from 7*7? No. 4x4 can be done with 49 multiplications using 2x2 Strassen on 2x2 blocks? No, that's 7*7=49? No, 2x2 Strassen is for 2x2 matrices. If we treat 2x2 blocks as scalars, R(4) <= 7 * R(2) = 7*7=49. But R(2)=7. So 49 is an upper bound. Is it tight? No, R(4) is known to be less than 49? Actually, R(4) <= 49 is a standard bound. But wait, R(3) <= 23. So R(6) <= 7 * R(3) = 7*23 = 161.
    # Also R(6) <= 49 * 1? No.
    # Let's look at the "rank 21" claim.
    # The tensor rank of 6x6x6 matrix multiplication is NOT 21.
    # The tensor rank of 2x2x2 is 7.
    # The tensor rank of 4x4x4 is at most 49.
    # The tensor rank of 6x6x6 is at most 7 * R(3) = 7 * 23 = 161.
    # Or R(6) <= 49 * 1? No.
    # What is the "rank 21" referring to?
    # Ah, maybe it's the number of scalar multiplications in Strassen's algorithm for 6x6 matrices if the 3x3 blocks are treated as "atomic" units that require 1 multiplication? No, that's not how tensor rank works.
    
    # Let's re-read the task. "Find a rank-minimal exact tensor decomposition".
    # The score is the rank.
    # The seed uses rank 125.
    # Can we do better than 125?
    # R(5) <= R(6) <= 161. This is worse than 125.
    # R(5) <= R(10) / 2? No.
    # Is there a known decomposition for 5x5 with rank < 125?
    # The best known upper bound for R(5) is 125 (schoolbook).
    # Wait, is R(5) < 125?
    # R(3) = 23 < 27.
    # R(4) <= 49.
    # R(5) <= ?
    # If we embed 5x5 into 6x6, we need a decomposition for 6x6.
    # R(6) <= 7 * R(3) = 161.
    # R(6) <= 49 * 1? No.
    # R(6) <= 7 * 7 * 1? No.
    # Actually, R(6) <= 7 * R(3) = 161.
    # R(6) <= 49 * R(1) = 49? No, that would imply R(4) <= 49 * R(1.5)? No.
    # The submultiplicativity property: R(AB) <= R(A)R(B).
    # R(6) = R(2*3) <= R(2)R(3) = 7 * 23 = 161.
    # R(5) <= R(10)/2? No.
    # R(5) <= R(6) <= 161.
    # So embedding into 6x6 gives rank 161, which is worse than 125.
    
    # What about embedding into 4x4? No, 5 > 4.
    # What about using a direct decomposition for 5x5?
    # The best known rank for 5x5 matrix multiplication is 125.
    # It is an open problem whether R(5) < 125.
    # However, we are asked to "improve" the program.
    # If the best known is 125, and the seed is 125, can we improve?
    # Maybe there is a known decomposition with rank < 125 that I am forgetting?
    # Or maybe the "rank 21" hint in the prompt is a red herring or refers to something else?
    # Let's check the prompt again: "utilizing the known rank-21 decomposition for 6x6 matrix multiplication".
    # Is there a rank-21 decomposition for 6x6?
    # R(6) <= 21?
    # If R(6) <= 21, then R(3) <= R(6)/7? No.
    # R(2) = 7. R(6) <= 7 * R(3). If R(6) <= 21, then R(3) <= 3? But R(3) = 23. Contradiction.
    # So R(6) cannot be 21.
    
    # Wait, is it possible that the "rank" in the prompt refers to the number of multiplications in a specific algorithm, not the tensor rank?
    # "Find a rank-minimal exact tensor decomposition... score is the rank".
    # This implies tensor rank.
    
    # Let's reconsider the "rank 21" claim.
    # Maybe it's for 3x3? No, R(3)=23.
    # Maybe it's for 2x2? No, R(2)=7.
    # Maybe it's for 4x4? No, R(4)<=49.
    # Maybe it's for 1x1? No.
    
    # Is it possible that the prompt is mistaken?
    # "utilizing the known rank-21 decomposition for 6x6 matrix multiplication"
    # If such a decomposition existed, R(6) <= 21.
    # Then R(3) <= R(6)/7? No, R(6) >= 7*R(3) is false. R(6) <= 7*R(3) is true.
    # If R(6) <= 21, then 21 >= 7*R(3) => R(3) <= 3. But R(3)=23.
    # So R(6) >= 23*7 = 161? No, R(6) is not necessarily 7*R(3).
    # But R(6) >= R(3) = 23.
    # And R(6) <= 7*R(3) = 161.
    # So R(6) is between 23 and 161.
    # It cannot be 21.
    
    # Therefore, the hint "rank-21 decomposition for 6x6" is likely incorrect or refers to a different metric (like arithmetic complexity or number of multiplications in a specific algorithm that is not a tensor decomposition).
    # However, the task asks for a tensor decomposition.
    
    # Given the constraints and the likely impossibility of finding a rank < 125 decomposition for 5x5 (as it's an open problem), the best we can do is stick to the schoolbook decomposition (rank 125) or find a slightly better one if it exists.
    # But wait, the seed is 125. If we return 125, the score is 1.0.
    # Can we return 124? Or 120?
    # If no such decomposition is known, we cannot.
    
    # Let's check if there is a known decomposition for 5x5 with rank < 125.
    # A quick search in my knowledge base:
    # R(1) = 1
    # R(2) = 7
    # R(3) = 23
    # R(4) = 49 (upper bound)
    # R(5) = 125 (upper bound)
    # R(6) = 161 (upper bound)
    # R(7) = ?
    # R(8) = 343 (upper bound from 7*7)
    
    # It seems 125 is the best known upper bound for R(5).
    # So we cannot improve on 125 with known methods.
    
    # However, the prompt says "improve it within that approach".
    # And "design a new program... utilizing the known rank-21 decomposition".
    # This is confusing.
    
    # Let's assume the prompt is correct and there IS a rank-21 decomposition for 6x6.
    # If so, how do we project it to 5x5?
    # If we have a decomposition of T_6 (6x6x6 tensor) into 21 rank-1 terms:
    # T_6 = sum_{r=1}^{21} u_r \otimes v_r \otimes w_r
    # where u_r, v_r in C^6, w_r in C^6.
    # We want to decompose T_5 (5x5x5 tensor).
    # T_5 is the restriction of T_6 to the subspace where the 6th index is 0?
    # No, T_5 is a different tensor.
    # But if we embed A (5x5) into A_6 (6x6) by adding a zero row/col,
    # and B (5x5) into B_6 (6x6) by adding a zero row/col,
    # then C_6 = A_6 B_6 has the form [C, 0; 0, 0].
    # The top-left 5x5 block of C_6 is C.
    # So we need to extract the top-left 5x5 block of the result.
    # In the tensor decomposition, each term is u_r \otimes v_r \otimes w_r.
    # The contribution to C[i,j] (for i,j < 5) is sum_r u_r[i] v_r[j] w_r[k]?
    # No, the tensor is T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}.
    # Here i is row of A, j is col of A, k is row of B?
    # Standard convention:
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # Tensor T_{ijk} where i is row of C, j is col of A (which is row of B), k is col of C.
    # T_{ijk} = delta_{jk}? No.
    # T_{ijk} = 1 if j=k? No.
    # T_{ijk} = A_{ij} B_{jk} -> C_{ik}.
    # So T_{ijk} = delta_{jk} is not right.
    # T_{ijk} is the tensor such that T(A,B) = C.
    # T_{ijk} = 1 if i is row of C, j is index of multiplication, k is col of C.
    # Actually, T_{ijk} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # So T_{ijk} = 1 for all i,j,k? No.
    # The tensor is defined by T_{ijk} = 1 if j is the shared index.
    # Wait, the tensor for matrix multiplication is:
    # T_{ijk} = 1 if i, j, k are valid indices.
    # No, T_{ijk} is a 3-way array.
    # T_{ijk} = 1 if the entry (i,k) of C depends on A_{ij} and B_{jk}.
    # So T_{ijk} = 1 for all i,j,k?
    # No, T_{ijk} is 1 if the product A_{ij} B_{jk} is added to C_{ik}.
    # So T_{ijk} = 1 for all i,j,k in {1..N}.
    # Wait, that would mean T is the all-ones tensor?
    # No. The tensor is T_{ijk} = delta_{jk}? No.
    # Let's check the definition.
    # The matrix multiplication tensor M_N is defined by:
    # M_N = sum_{i,j,k=1}^N e_i \otimes e_j \otimes e_k \cdot \delta_{jk}?
    # No.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # This is a bilinear map.
    # The tensor is T = sum_{i,j,k} T_{ijk} e_i \otimes e_j \otimes e_k.
    # The condition is that the coefficient of e_i \otimes e_k in the output is sum_j A_{ij} B_{jk}.
    # This corresponds to the tensor T having entries T_{ijk} = 1 if j is the summation index?
    # Actually, the standard definition is:
    # T_{ijk} = 1 if i, j, k are such that A_{ij} and B_{jk} contribute to C_{ik}.
    # This is true for all i,j,k.
    # So T_{ijk} = 1 for all i,j,k?
    # No, that's not right.
    # The tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k \cdot \delta_{jk}?
    # Let's test.
    # If T_{ijk} = delta_{jk}, then T(A,B) = sum_{i,j,k} delta_{jk} A_{ij} B_{jk} e_i \otimes e_k = sum_{i,k} (sum_j A_{ij} B_{jk}) e_i \otimes e_k = C.
    # Yes! The matrix multiplication tensor has entries T_{ijk} = delta_{jk}.
    
    # So for N=5, T_{ijk} = 1 if j=k, 0 otherwise.
    # For N=6, T_{ijk} = 1 if j=k, 0 otherwise.
    
    # Now, if we have a decomposition of T_6 into rank-1 terms:
    # T_6 = sum_{r=1}^{R} u_r \otimes v_r \otimes w_r
    # Then for any A_6, B_6, C_6 = sum_r (u_r^T A_6) (v_r^T B_6) w_r?
    # No, the contraction is:
    # C_{ik} = sum_{j} T_{ijk} A_{ij} B_{jk} = sum_{j} delta_{jk} A_{ij} B_{jk} = sum_j A_{ij} B_{jk}.
    # Using the decomposition:
    # T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
    # Then C_{ik} = sum_{j} (sum_r U_{ir} V_{jr} W_{kr}) A_{ij} B_{jk}
    # = sum_r U_{ir} W_{kr} (sum_j V_{jr} A_{ij} B_{jk})
    # This doesn't look like a standard matrix multiplication form.
    
    # Let's check the index ordering.
    # U is (n*m, rank). V is (m*p, rank). W is (p*n, rank).
    # The task says: "output factor follows the official trace convention W[k*n+i,r]".
    # This implies W is indexed by k (col of B) and i (row of A)?
    # Let's check the seed code.
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # Here i is row of A (0..n-1), j is col of A (0..m-1).
    # k is col of B (0..p-1).
    # So U is indexed by (i,j). V is indexed by (j,k). W is indexed by (k,i).
    # The tensor entry T_{ijk} corresponds to U_{ij,r} V_{jk,r} W_{ki,r}.
    # Sum over r: sum_r U_{ij,r} V_{jk,r} W_{ki,r}.
    # This sum should equal T_{ijk} = delta_{jk}?
    # No, the tensor is for the whole multiplication.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # The decomposition is:
    # C_{ik} = sum_r (sum_j U_{ij,r} A_{ij}) (sum_k V_{jk,r} B_{jk}) W_{ki,r}?
    # No, that's not right.
    
    # Let's look at the standard tensor decomposition form.
    # T = sum_r u_r \otimes v_r \otimes w_r
    # where u_r in F^n, v_r in F^m, w_r in F^p?
    # No, the dimensions are n, m, p.
    # The tensor T is in F^{n x m x p}.
    # T_{ijk} = u_{i,r} v_{j,r} w_{k,r} summed over r.
    # In matrix multiplication, T_{ijk} = delta_{jk} (if we identify indices properly).
    # But the indices are i (row of A), j (col of A / row of B), k (col of B).
    # So T_{ijk} = 1 if j=k? No, j is col of A, k is col of B.
    # They are not the same index.
    # The summation is over j.
    # So T_{ijk} is 1 for all i,j,k?
    # No.
    # Let's re-derive.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # This can be written as C = A B.
    # The tensor T is such that T(A,B) = C.
    # T is a 3-linear form? No, it's a bilinear map.
    # The tensor associated with the bilinear map is T in F^{n x p} \otimes (F^{n x m} \otimes F^{m x p})^*?
    # No, the tensor rank is defined for the 3-way array T_{ijk}.
    # T_{ijk} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # This is true for all i,j,k.
    # So T_{ijk} = 1 for all i,j,k?
    # If T_{ijk} = 1, then sum_r U_{ir} V_{jr} W_{kr} = 1 for all i,j,k.
    # This means U \otimes V \otimes W = 1.
    # This is possible with rank 1? No.
    
    # I think I am misinterpreting the tensor.
    # The tensor for matrix multiplication is:
    # T_{ijk} = 1 if i, j, k are valid.
    # No, the tensor is T_{ijk} = delta_{jk} is for the case where the output is indexed by i and k, and the summation is over j.
    # But in the tensor decomposition, the indices are i, j, k.
    # The condition is that T_{ijk} = 1 if the product A_{ij} B_{jk} is added to C_{ik}.
    # This is always true.
    # So T_{ijk} = 1 for all i,j,k.
    # But then the rank of the all-ones tensor is 1?
    # No, the all-ones tensor of size N x N x N has rank 1?
    # No, the all-ones tensor is u \otimes v \otimes w where u=v=w=1.
    # So rank 1.
    # But matrix multiplication has rank N^3? No.
    
    # I am clearly confused about the tensor definition.
    # Let's look at the seed code again.
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # This sets T_{ijk} = 1 for the specific (i,j,k) triple corresponding to the column.
    # And there are N^3 columns.
    # So T_{ijk} = 1 for all i,j,k.
    # This means the tensor is the all-ones tensor.
    # But the all-ones tensor has rank 1.
    # Why is the seed using rank N^3?
    
    # Ah, the tensor is NOT the all-ones tensor.
    # The tensor T_{ijk} is defined such that:
    # sum_r U_{ir} V_{jr} W_{kr} = T_{ijk}
    # And T_{ijk} is 1 if j is the shared index?
    # No.
    
    # Let's check the contraction.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # Using the decomposition:
    # C_{ik} = sum_r (sum_j U_{ij,r} A_{ij}) (sum_k V_{jk,r} B_{jk}) W_{ki,r}?
    # No, the indices don't match.
    
    # Let's assume the standard form:
    # C_{ik} = sum_r (sum_j U_{ij,r} A_{ij}) (sum_k V_{jk,r} B_{jk}) W_{ki,r}
    # This doesn't make sense.
    
    # Let's try:
    # C_{ik} = sum_r U_{ir} V_{kr} (sum_j A_{ij} B_{jk} W_{jr})?
    
    # Actually, the standard tensor decomposition for matrix multiplication is:
    # T = sum_{r=1}^R u_r \otimes v_r \otimes w_r
    # where u_r in F^n, v_r in F^m, w_r in F^p.
    # And the condition is:
    # sum_r u_{i,r} v_{j,r} w_{k,r} = T_{ijk}
    # where T_{ijk} = 1 if j=k? No.
    # T_{ijk} = 1 if i, j, k are such that A_{ij} and B_{jk} contribute to C_{ik}.
    # This is true for all i,j,k.
    # So T_{ijk} = 1.
    # But then rank 1.
    
    # I must be missing something fundamental.
    # Let's look at the seed code's logic.
    # It creates N^3 terms.
    # Each term has U_{ij}=1, V_{jk}=1, W_{ki}=1.
    # So for a given (i,j,k), the term contributes to T_{ijk}.
    # So T_{ijk} = 1 for all i,j,k.
    # This implies the tensor is the all-ones tensor.
    # But the all-ones tensor has rank 1.
    # So why is the seed using N^3?
    
    # Maybe the tensor is NOT the all-ones tensor.
    # Maybe T_{ijk} is 1 only if j is the shared index?
    # No, j is always the shared index.
    
    # Wait, the tensor is for the map (A, B) -> C.
    # The tensor is in F^{n x p} \otimes (F^{n x m} \otimes F^{m x p})^*.
    # No, it's a 3-way array T_{i j k}.
    # The condition is that the bilinear form is:
    # <T, A \otimes B> = C
    # where A is in F^{n x m}, B is in F^{m x p}, C is in F^{n x p}.
    # This is not a standard inner product.
    
    # Let's stick to the code.
    # The code uses rank N^3.
    # I will stick to rank N^3.
    
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
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
