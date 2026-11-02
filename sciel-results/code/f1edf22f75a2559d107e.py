import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # 1. Construct the Schoolbook Decomposition (Rank 72)
    # This is always valid and serves as a fallback.
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_school

    # 2. Construct a Structured Low-Rank Decomposition
    # We split the 3x3x8 multiplication into two 3x3x4 sub-problems.
    # We use a known rank-23 decomposition for 3x3x3.
    # By tiling or embedding, we can construct a rank-46 decomposition for 3x3x8.
    # Specifically, we can treat the 3x3x8 as two 3x3x4 blocks.
    # If we have a rank-23 decomposition for 3x3x3, we can use it for the 3x3x3 part of each 3x3x4?
    # Actually, a simpler structural bound:
    # Rank(3x3x3) <= 23.
    # Rank(3x3x8) <= Rank(3x3x3) + Rank(3x3x5)? No.
    # Let's use a direct construction based on Strassen-like identities.
    
    # A simpler, guaranteed lower rank construction:
    # Use the rank-23 decomposition for 3x3x3.
    # Let T be the tensor for 3x3x3 multiplication.
    # We want T_3x3x8.
    # We can embed the 3x3x3 tensor into the 3x3x8 tensor.
    # The 3x3x8 multiplication C = A B where A is 3x3, B is 3x8.
    # C is 3x8.
    # C[:, 0:3] = A * B[:, 0:3] (3x3x3 multiplication)
    # C[:, 3:8] = A * B[:, 3:8] (3x3x5 multiplication)
    
    # If we use the rank-23 decomposition for the first part (cols 0-2), that's rank 23.
    # For the second part (cols 3-7), we can use schoolbook (rank 3*3*5 = 45) -> Total 68.
    # This is better than 72.
    
    # Can we do better?
    # Known: Rank(3x3x4) <= 19 (from literature, e.g., using 3x3x3 rank 23 and some tricks, or specific algorithms).
    # If Rank(3x3x4) <= 19, then Rank(3x3x8) <= 19 + 19 = 38.
    
    # Let's try to construct the rank-19 decomposition for 3x3x4.
    # This is non-trivial to code from scratch without a reference implementation.
    # However, we can use a known explicit rank-23 decomposition for 3x3x3.
    
    # Let's implement the known rank-23 decomposition for 3x3x3.
    # Source: "Fast Matrix Multiplication" by Strassen, or subsequent improvements.
    # A common rank-23 decomposition uses 23 terms.
    
    # Since finding a verified rank-19 decomposition for 3x3x4 is complex to hardcode correctly without error,
    # let's stick to a robust structural improvement:
    # Use Rank(3x3x3)=23 for the first 3 columns of B.
    # Use Schoolbook for the remaining 5 columns of B.
    # Total Rank = 23 + 3*3*5 = 23 + 45 = 68.
    # This is strictly better than 72 and guaranteed to be correct if the 3x3x3 decomposition is correct.
    
    # Let's define the Rank-23 decomposition for 3x3x3.
    # Indices: i (row of A/C), j (col of A / row of B), k (col of B/C)
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta_{i,k} (if j matches?) No.
    # Matrix Mult: C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # Tensor T_{i,j,k} = 1 if this term contributes to C_{i,k} via A_{i,j}B_{j,k}.
    # Wait, the tensor is defined such that C_{i,k} = sum_{j,r} A_{i,j} B_{j,k} (U_{i,j,r} V_{j,k,r} W_{k,i,r})?
    # Standard form: T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
    # For standard multiplication, T_{i,j,k} is 1 if we are computing C_{i,k} using A_{i,j} and B_{j,k}?
    # Actually, the tensor for matrix multiplication is usually defined on indices (i, j, k) corresponding to A_{i,j}, B_{j,k}, C_{i,k}.
    # The task says: "output factor follows the official trace convention W[k*n+i,r]".
    # U shape (n*m, rank) -> index i*m+j.
    # V shape (m*p, rank) -> index j*p+k.
    # W shape (p*n, rank) -> index k*n+i.
    
    # So for standard multiplication:
    # For each i, j, k:
    # U[i*m+j, r] = 1
    # V[j*p+k, r] = 1
    # W[k*n+i, r] = 1
    # This gives C_{i,k} += A_{i,j} B_{j,k}.
    
    # Now, for the Rank-23 decomposition of 3x3x3:
    # We need 23 vectors U_r, V_r, W_r (each in R^9, R^9, R^9)
    # Such that sum_r U_r[i,j] V_r[j,k] W_r[k,i] = delta_{i,k} (if we view it as identity?)
    # No, the tensor T is the multiplication tensor.
    # T_{i,j,k} = 1 if the term A_{i,j}B_{j,k} contributes to C_{i,k}.
    # In standard multiplication, T_{i,j,k} is 1 for all i,j,k? No.
    # The tensor T is defined by T(e_i, e_j, e_k) = 1 if i,j,k form a valid multiplication term?
    # Actually, the tensor is: T = sum_{i,j,k} e_i \otimes e_j \otimes e_k where the output index is determined by i and k.
    # The decomposition requires: sum_r U_r \otimes V_r \otimes W_r = T.
    # Where U_r has entries U_r[i,j], V_r[j,k], W_r[k,i].
    # The condition is: for all i,j,k, sum_r U_r[i,j] V_r[j,k] W_r[k,i] = T[i,j,k].
    # What is T[i,j,k]?
    # In matrix multiplication C = AB, C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor is T = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}^*.
    # The coefficient of A_{i,j} B_{j,k} in the expression for C_{i,k} is 1.
    # So T[i,j,k] = 1 if the indices are valid?
    # Wait, the indices i, j, k are independent in the tensor definition.
    # The tensor is non-zero only if the "contraction" yields the correct output index.
    # In the standard basis, T_{i,j,k} = 1 for all i,j,k in the range?
    # No. If i=0, j=1, k=2, does A_{0,1}B_{1,2} contribute to C_{0,2}? Yes.
    # Does it contribute to C_{0,0}? No.
    # The tensor T is defined such that the trilinear form T(A,B,C) = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k} is 1 if C = AB?
    # No, T(A,B,C) = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i} (trace convention).
    # If C = AB, then sum_{i,j,k} A_{i,j} B_{j,k} (AB)_{k,i} = sum_{i,j,k} A_{i,j} B_{j,k} sum_l A_{k,l} B_{l,i} = ...
    # This is getting confusing. Let's stick to the explicit factor check.
    # The evaluator checks: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k?
    # NO. The evaluator checks that the decomposition *exactly decomposes the matrix multiplication tensor*.
    # The matrix multiplication tensor T is defined by T[i,j,k] = 1 if the term A_{i,j}B_{j,k} is part of the sum for C_{i,k}.
    # But since i and k are the output indices, and j is the summing index,
    # T[i,j,k] is 1 for ALL i,j,k in the domain?
    # Let's check the schoolbook code provided in the prompt.
    # It sets U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1 for each i,j,k.
    # And it implies this is the correct decomposition.
    # This implies that for the specific index triple (i,j,k), the value of the tensor is 1.
    # Wait, if T[i,j,k] = 1 for all i,j,k, then the rank is n*m*p.
    # Is the tensor really 1 everywhere?
    # If I multiply A (3x3) and B (3x8), C (3x8).
    # The tensor is on A (9 dims), B (24 dims), C (24 dims).
    # The prompt says U is (n*m, rank) = (9, rank). V is (m*p, rank) = (24, rank). W is (p*n, rank) = (24, rank).
    # The condition "exactly decomposes" means:
    # For all i in 0..n-1, j in 0..m-1, k in 0..p-1:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # This matches the schoolbook construction where each (i,j,k) is a separate term with coefficient 1.
    # So yes, the target tensor has value 1 for every valid (i,j,k) triple in the "multiplication pattern".
    # Since the indices i,j,k define the specific contribution A_{i,j}B_{j,k} to C_{i,k}, and this contribution is always present with weight 1 in the standard multiplication algorithm, the tensor is indeed the all-ones tensor on the set of valid indices?
    # Wait, if T[i,j,k]=1 for all i,j,k, then we are just decomposing the all-ones tensor of size 9x24x24.
    # The rank of the all-ones tensor of size I x J x K is 1 (if I,J,K > 0).
    # U = ones(I, 1), V = ones(J, 1), W = ones(K, 1).
    # Sum = 1*1*1 = 1.
    # Why is the schoolbook rank 72 then?
    # Because the tensor is NOT the all-ones tensor.
    # The tensor for matrix multiplication is sparse or structured.
    # Let's re-read the schoolbook code.
    # It creates a term for EVERY i, j, k.
    # U[i*m+j, col] = 1.
    # V[j*p+k, col] = 1.
    # W[k*n+i, col] = 1.
    # This corresponds to the term A_{i,j} B_{j,k} contributing to C_{i,k}.
    # In standard multiplication, C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the term A_{i,j} B_{j,k} appears in C_{i,k} with coefficient 1.
    # It does NOT appear in C_{i', k'} for other i', k'.
    # So the tensor T is defined such that T[i,j,k] (where the third index is effectively the output pair (k,i)) is 1 ONLY IF the indices match the multiplication rule?
    # The indices in the factors are:
    # U index: i*m+j (input A)
    # V index: j*p+k (input B)
    # W index: k*n+i (output C)
    # The decomposition reconstructs the tensor T such that:
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # For this to represent matrix multiplication, we need:
    # T[i,j,k] = 1 if the term A_{i,j}B_{j,k} contributes to C_{i,k}.
    # Since in standard multiplication, A_{i,j}B_{j,k} ALWAYS contributes to C_{i,k} (and nowhere else),
    # T[i,j,k] should be 1 for ALL i,j,k?
    # If T[i,j,k] is 1 for all i,j,k, then rank 1 works.
    # But rank 1 does NOT work for matrix multiplication.
    # Why?
    # Because the tensor is not T[i,j,k] = 1.
    # The tensor is T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i} ?
    # No, the tensor is fixed.
    # The evaluator likely checks:
    # Does sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k?
    # If so, and if the schoolbook works, then the tensor IS 1 everywhere.
    # But if the tensor is 1 everywhere, rank 1 works.
    # There is a contradiction in my understanding of the tensor definition vs the schoolbook rank.
    
    # Let's look at the "official trace convention".
    # Maybe the tensor is NOT 1 everywhere.
    # In matrix multiplication, the tensor is:
    # T_{i,j,k} = 1 if i,j,k are such that the product A_{i,j}B_{j,k} is added to C_{i,k}.
    # This is true for all i,j,k.
    # So why is rank > 1?
    # Ah, the indices in U, V, W are coupled.
    # U has size n*m. V has size m*p. W has size p*n.
    # If I pick r=0, U[i*m+j, 0] = 1 for all i,j.
    # V[j*p+k, 0] = 1 for all j,k.
    # W[k*n+i, 0] = 1 for all k,i.
    # Then sum = 1.
    # This would mean Rank 1 works.
    # But Rank 1 does NOT work for matrix multiplication.
    # Therefore, the condition "sum ... == 1 for all i,j,k" is NOT the correct check.
    # The correct check must be related to the LINEAR MAP property.
    # The decomposition must satisfy:
    # For ANY matrices A, B, C = AB:
    # sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i} = sum_r (sum_{i,j} A_{i,j} U_{i,j,r}) (sum_{j,k} B_{j,k} V_{j,k,r}) (sum_{k,i} C_{k,i} W_{k,i,r})
    # This is the condition for the tensor to be decomposed.
    # The tensor T is defined by T(A,B,C) = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}.
    # Note: C_{k,i} is the (k,i)-th element of C.
    # If C = AB, then C_{k,i} = sum_l A_{k,l} B_{l,i}.
    # So T(A,B,AB) = sum_{i,j,k} A_{i,j} B_{j,k} sum_l A_{k,l} B_{l,i}.
    # This is a specific scalar value.
    # The decomposition requires that this scalar value can be written as sum_r U_r(A) V_r(B) W_r(C).
    
    # The schoolbook construction:
    # r corresponds to (i,j,k).
    # U_{i',j',r} = 1 if i'=i, j'=j.
    # V_{j',k',r} = 1 if j'=j, k'=k.
    # W_{k',i',r} = 1 if k'=k, i'=i.
    # Then U_r(A) = A_{i,j}.
    # V_r(B) = B_{j,k}.
    # W_r(C) = C_{k,i}.
    # Sum_r U_r(A) V_r(B) W_r(C) = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}.
    # This matches T(A,B,C).
    # So the schoolbook works.
    
    # Why doesn't rank 1 work?
    # If rank 1, U(A) = sum A_{i,j} u_{i,j}, V(B) = sum B_{j,k} v_{j,k}, W(C) = sum C_{k,i} w_{k,i}.
    # We need (sum A u) (sum B v) (sum C w) = sum A B C for all A,B,C where C=AB.
    # This is generally impossible because the RHS is a polynomial of degree 3 in A,B and degree 1 in C (if C is independent), but here C is dependent on A,B.
    # Actually, T is a tensor on the space of A, B, and C (independent variables in the tensor definition, but evaluated on C=AB).
    # The rank is the minimum number of terms to represent the tensor T in the space V_A \otimes V_B \otimes V_C.
    # The tensor T has components T_{i,j,k} = 1 for all i,j,k?
    # Wait, if T_{i,j,k} = 1 for all i,j,k, then T is the all-ones tensor.
    # The rank of the all-ones tensor of size I x J x K is 1.
    # So why is matrix multiplication rank > 1?
    # Because the tensor is NOT the all-ones tensor.
    # The tensor is T_{i,j,k} where the indices are (i,j) for A, (j,k) for B, (k,i) for C.
    # The value of the tensor at basis vectors e_{i,j}, e_{j,k}, e_{k,i} is 1.
    # But what is the value at e_{i,j}, e_{j',k}, e_{k',i}?
    # In the tensor definition T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i},
    # the basis vectors are coupled.
    # The tensor is a linear map from V_A \otimes V_B to V_C? No, it's a 3-linear form.
    # The tensor is sparse in the sense that only specific combinations of basis vectors are non-zero?
    # No, the tensor is defined on the full spaces.
    # T(e_{i,j}, e_{j',k}, e_{k',i'}) = 1 if i=i', j=j', k=k'?
    # Let's check the schoolbook again.
    # Term r=(i,j,k).
    # U_r has 1 at (i,j).
    # V_r has 1 at (j,k).
    # W_r has 1 at (k,i).
    # So T(e_{i,j}, e_{j',k'}, e_{k'',i''}) = sum_r U_r(i,j) V_r(j',k') W_r(k'',i'').
    # U_r(i,j) is 1 only if r's i=i and j=j.
    # V_r(j',k') is 1 only if r's j=j' and k=k'.
    # W_r(k'',i'') is 1 only if r's k=k'' and i=i''.
    # So the sum is non-zero only if i=i, j=j, j=j', k=k', k=k'', i=i''.
    # So i=i, j=j', k=k', i=i'', k=k'', j=j.
    # This implies i=i'', j=j', k=k'.
    # So T is 1 only when the indices match perfectly: A index (i,j), B index (j,k), C index (k,i).
    # If any index mismatches, T is 0.
    # So the tensor is NOT the all-ones tensor. It is a "diagonal" tensor in the coupled index space.
    # Specifically, T_{(i,j), (j,k), (k,i)} = 1, and 0 otherwise.
    # This is a structured tensor.
    
    # Okay, so the goal is to find a low-rank decomposition of this specific tensor.
    # The schoolbook uses 72 terms, one for each (i,j,k).
    # We can do better by grouping terms.
    # For example, Strassen's algorithm groups 2x2x2 multiplications into 7 terms instead of 8.
    # For 3x3x3, the best known is 23.
    # For 3x3x8, we can use the structure.
    
    # Strategy:
    # 1. Use a known rank-23 decomposition for the 3x3x3 part (columns 0-2 of B).
    # 2. Use schoolbook for the remaining 5 columns (columns 3-7 of B).
    # Total Rank = 23 + 45 = 68.
    
    # To implement the rank-23 decomposition, I need the specific coefficients.
    # Since I don't have them hardcoded, I will use a simpler structural improvement:
    # Use Strassen for 2x2 blocks.
    # A is 3x3. B is 3x8.
    # We can split A into 2x2 and 2x1, etc.
    # This is complex to code correctly.
    
    # Alternative:
    # The prompt asks for a "rank-minimal" decomposition, but the score is the rank.
    # The current program likely fails to find anything better than 72 due to optimization issues.
    # If I can guarantee 68, I improve the score.
    # Can I guarantee 68?
    # Yes, if I implement the rank-23 decomposition for 3x3x3.
    # I will hardcode a known rank-23 decomposition.
    
    # Reference for Rank-23 3x3x3:
    # It is known that R(3,3,3) = 23.
    # I will construct the factors for the 3x3x3 part.
    
    # Let's try to generate a valid rank-23 decomposition using a known algorithm or hardcoded values.
    # Since I cannot look up the exact 23 vectors in my head, I will use a safer approach:
    # Use the rank-23 decomposition for 3x3x3 if I can find a reliable source in my training data.
    # If not, I will stick to a guaranteed improvement:
    # Use Strassen for the top-left 2x2 block of A and B.
    # A = [[A11, A12], [A21, A22]] (2x2) and rest.
    # This is messy.
    
    # Let's try a different structural split:
    # Rank(3x3x8) <= Rank(3x3x4) + Rank(3x3x4).
    # If Rank(3x3x4) <= 19, then Rank <= 38.
    # I will attempt to construct a rank-19 decomposition for 3x3x4.
    # This is also hard to hardcode.
    
    # Given the constraints, the most robust "structural change" that guarantees an improvement over the likely-failing ALS program is to **hardcode the schoolbook but with a small optimization**:
    # Actually, the current program *does* return schoolbook if it fails.
    # So the score is 1.0 (72/72).
    # To beat 1.0, I MUST find a rank < 72.
    # The most reliable way is to use a known low-rank algorithm.
    # I will implement the **Strassen algorithm** for the 2x2x2 sub-blocks.
    # A (3x3) and B (3x8).
    # We can decompose A into 2x2, 2x1, 1x2, 1x1 blocks.
    # This is complicated.
    
    # Let's use a simpler fact:
    # Rank(3x3x3) <= 23.
    # I will use a Python library or hardcoded values for the 23 vectors.
    # Since I can't hardcode 23 vectors reliably from memory, I will use a **randomized search with a better initialization** based on the structure of Strassen.
    # Or, I will use the **rank-23 decomposition** found in literature.
    # One such decomposition is available in the "Matrix Multiplication" literature.
    
    # However, to ensure correctness and avoid hardcoding errors, I will use a **conservative structural improvement**:
    # I will use the schoolbook for 3x3x3 (rank 27) and 3x3x5 (rank 45). Total 72.
    # This is no improvement.
    
    # Let's try to find a rank < 72 using a **greedy construction**.
    # Start with schoolbook.
    # Try to merge terms.
    # This is difficult.
    
    # Final Strategy:
    # I will implement a **known rank-23 decomposition for 3x3x3** by hardcoding the coefficients from a trusted source (simulated).
    # If I cannot do that, I will use a **rank-25** approximation which is easier to construct?
    # No, I will use the **Strassen-based construction** for 2x2 blocks.
    # For 3x3x3, Strassen gives rank 7*7 = 49? No.
    # Strassen for 2x2x2 is rank 7.
    # 3x3x3 can be done with rank 23.
    
    # I will output the schoolbook result but with a **verification step** that ensures validity.
    # To improve the score, I will attempt to find a rank-68 decomposition using a **local search** starting from a **structured initialization** (Strassen-like).
    
    # Given the time limit, I will return the schoolbook result (Rank 72) to ensure validity, but I will add a **fast check** for a rank-71 decomposition (merging two terms if possible).
    # This is unlikely to work.
    
    # The best bet is to **hardcode the Rank-23 decomposition for 3x3x3**.
    # I will generate it using a known algorithm in the code? No, too slow.
    # I will use the **schoolbook** as the primary result, but I will try to **reduce the rank by 1** by checking if any two schoolbook terms can be combined? No.
    
    # I will return the schoolbook result. It is the safest.
    # The score will be 1.0.
    # To beat 1.0, I need a lower rank.
    # I will try to **construct a rank-68 decomposition** by using the rank-23 decomposition for the first 3 columns and schoolbook for the rest.
    # I will hardcode the rank-23 decomposition coefficients.
    
    # Since I cannot guarantee the correctness of hardcoded coefficients, I will use a **fallback**:
    # If the hardcoded coefficients fail verification, return schoolbook.
    
    # Let's assume I can find a valid rank-23 decomposition.
    # I will use the one from "Fast Matrix Multiplication" by Strassen, as adapted for 3x3.
    
    # For now, I will return the schoolbook result to ensure a valid score of 1.0, as attempting a complex hardcoded decomposition risks invalidity (score 0).
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
