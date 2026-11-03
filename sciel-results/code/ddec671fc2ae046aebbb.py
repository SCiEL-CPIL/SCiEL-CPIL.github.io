def verify_matmul(U, V, W):
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # Reshape to (n, m, r), (m, p, r), (p, n, r)
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    
    # Compute sum_l U[i,j,l] V[j,k,l] W[k,i,l]
    # This gives a tensor of size (n, m, p)
    # But we need to check specific entries.
    # The matmul tensor T has T[i, j, k] = 1 if we consider the "path" i->j->k?
    # No, the tensor is in A* x B* x C.
    # The entry is 1 for (i,j), (j,k), (i,k).
    # The einsum 'ijl,jkl,kil->ijk' computes S[i,j,k] = sum_l U[i,j,l] V[j,k,l] W[k,i,l].
    # For the matmul tensor, we require S[i,j,k] = 1 for ALL i,j,k?
    # NO.
    # The matmul tensor is sparse.
    # However, the indices i,j,k in S are independent.
    # In the matmul tensor, the "indices" are (i,j), (j,k), (i,k).
    # The value is 1.
    # Are there other non-zero entries? No.
    # So, does S[i,j,k] correspond to the tensor entry?
    # Yes, if we identify the tensor entry at (i,j, j,k, i,k) with the value S[i,j,k]?
    # But S is defined for ALL i,j,k.
    # If the tensor is zero elsewhere, then S[i,j,k] should be 0 for "invalid" triples?
    # But there are no "invalid" triples in the index space $n \times m \times p$.
    # Every i, j, k is a valid index for the 3-tensor.
    # So, does the matmul tensor have 1s everywhere?
    # NO. The matmul tensor is $T = \sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ik}$.
    # This sum is over all i,j,k.
    # So $T_{(ij), (jk), (ik)} = 1$.
    # And for any other combination of indices (e.g. $i' \neq i$), it is 0.
    # But the indices are coupled.
    # In the flattened array, the entry at $(a, b, c)$ is 1 ONLY if $a,b,c$ correspond to the SAME $i,j,k$.
    # So, the tensor is sparse.
    # The `einsum` computes $S[i,j,k]$.
    # $S[i,j,k]$ corresponds to the tensor entry at $(Aidx(i,j), Bidx(j,k), Widx(k,i))$.
    # So, for the matmul tensor, $S[i,j,k]$ MUST be 1 for ALL i,j,k.
    # Because for every i,j,k, there IS a term $e_{ij} \otimes e_{jk} \otimes e_{ik}$ in the sum.
    # So the tensor is **NOT** sparse in the $i,j,k$ representation!
    # It is the **All-ones** tensor in the $i,j,k$ coordinates!
    
    # Wait.
    # If $T = \sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ik}$,
    # then the entry at $(a,b,c)$ is 1 if $a=ij, b=jk, c=ki$ for some i,j,k.
    # Since the map $(i,j,k) \to (a,b,c)$ is injective,
    # the tensor has 1s at $n \cdot m \cdot p$ positions and 0s elsewhere.
    # The total size is $nm \cdot mp \cdot pn = n^2 m^2 p^2$.
    # The number of 1s is $nmp$.
    # So it IS sparse.
    
    # Now, look at `S[i,j,k] = \sum_l U_{ij,l} V_{jk,l} W_{ki,l}`.
    # This sum is exactly the tensor entry at $(a,b,c)$ where $a=ij, b=jk, c=ki$.
    # So, for the matmul tensor, $S[i,j,k]$ MUST be 1 for ALL i,j,k.
    # And for any other index combination (which doesn't exist in the $i,j,k$ grid), it's 0.
    # So, the condition `S[i,j,k] == 1 for all i,j,k` IS the correct condition for the Matmul Tensor!
    
    # So `verify_np` IS correct.
    # And the target IS the All-ones tensor in the $i,j,k$ grid.
    # Which is the Matmul Tensor.
    
    # So why is the rank not 1?
    # Because $U, V, W$ are constrained to be low rank?
    # No, we are finding the rank.
    # If the target is All-ones, Rank 1 is possible.
    # $U_{ij,l} = 1, V_{jk,l} = 1, W_{ki,l} = 1$.
    # $S[i,j,k] = 1 \cdot 1 \cdot 1 = 1$.
    # This works.
    
    # So why is the baseline 66?
    # Maybe the "Matmul" task is NOT the All-ones tensor?
    # Maybe it's the **Identity** tensor?
    # Or maybe the indices are different?
    
    # Let's check the score again.
    # If Rank 1 was possible, the score would be 72.
    # The score is 1.09.
    # This implies Rank ~66 is the best found.
    # This implies Rank 1 is NOT a valid solution.
    # Why?
    # Maybe $U, V, W$ must be **Sparse**? No.
    # Maybe the entries must be **0 or 1**?
    # "Integer/half-integer entries".
    
    # If Rank 1 is invalid, there must be a constraint I'm missing.
    # "Output factor follows the official trace convention W[k*n+i,r]."
    
    # Is it possible the task is to decompose the **Matrix Multiplication Map** as a **Bilinear Map** $A \times B \to C$?
    # Yes.
    # And the rank is the tensor rank.
    # For $3 \times 4 \times 6$, the rank is unknown.
    # But it is definitely > 1.
    # So the "All-ones" interpretation must be wrong.
    
    # Where is the error?
    # $T = \sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ik}$.
    # This tensor is the **Matmul Tensor**.
    # Its rank is the rank of matrix multiplication.
    # It is NOT the All-ones tensor in the full space $A \otimes B \otimes C$.
    # It is a sum of $nmp$ rank-1 tensors.
    # Can it be written with fewer?
    # Yes, that's the question.
    # But can it be written with 1?
    # No, because it is not a rank-1 tensor.
    # A rank-1 tensor $u \otimes v \otimes w$ has the property that $T_{a,b,c} = u_a v_b w_c$.
    # The matmul tensor does NOT have this property.
    # $T_{a,b,c}$ is 1 or 0.
    # If it were rank 1, $T_{a,b,c} T_{a',b',c'} = T_{a,b',c'} T_{a',b,c'}$.
    # This fails for matmul.
    
    # So, why did I think `verify_np` checks All-ones?
    # `verify_np` checks if $S[i,j,k] == 1$.
    # $S[i,j,k]$ IS the tensor entry at $(ij, jk, ki)$.
    # So it checks if the tensor entries at those specific positions are 1.
    # It does NOT check if the tensor is 1 EVERYWHERE in the full space.
    # It only checks the $nmp$ positions that are non-zero in the matmul tensor.
    # It does NOT check if the other positions are 0.
    # **THIS IS THE BUG.**
    # `verify_np` only checks that the "1" positions are 1.
    # It does NOT check that the "0" positions are 0.
    # So a Rank 1 solution $U=1, V=1, W=1$ would give $S[i,j,k]=1$ for all $i,j,k$.
    # So it would pass `verify_np`.
    # But it is NOT a valid matmul decomposition because it produces non-zero values at the "0" positions of the true tensor.
    
    # So, to improve, I must use a **Correct Verification** that checks BOTH 1s and 0s.
    # And then find a lower rank decomposition that satisfies it.

def verify_matmul_correct(U, V, W):
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    # Compute S[i,j,k]
    S = np.einsum('ijl,jkl,kil->ijk', U_r, V_r, W_r)
    # Check 1s
    if not np.allclose(S, 1.0, atol=1e-9):
        return False
    # Check 0s
    # The tensor is zero at all other indices.
    # How to check?
    # We can reconstruct the full tensor and check sparsity?
    # Or simply check that the rank is minimal?
    # Actually, if we use ALS, we minimize the error.
    # We need a target tensor.
    
    # Construct Target Tensor T of size (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
                
    # Reconstruct T_est
    # T_est[a,b,c] = sum_l U[a,l] V[b,l] W[c,l]
    T_est = np.einsum('al,bl,cl->abc', U, V, W)
    
    return np.allclose(T_est, T, atol=1e-9)
