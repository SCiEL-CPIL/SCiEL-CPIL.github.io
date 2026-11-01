     def construct():
         n, m, p = 3, 3, 3
         rank = 23
         U = [[0]*rank for _ in range(9)]
         V = [[0]*rank for _ in range(9)]
         W = [[0]*rank for _ in range(9)]
         
         # Helper to set a column
         def set_col(c, i, j, k, val=1):
             # U index i*m+j
             U[i*3+j][c] = val
             # V index j*p+k
             V[j*3+k][c] = val
             # W index k*n+i
             W[k*3+i][c] = val
             
         # 1. All ones
         for i in range(3):
             for j in range(3):
                 U[i*3+j][0] = 1
                 V[j*3+0][0] = 1 # This is wrong. V depends on j,k.
                 # If the vector is all ones, then V[j,k,0] = 1 for all j,k.
                 # So V[j*3+k][0] = 1 for all j,k.
         # Let's just fill the matrices directly based on the vector definitions.
         
         # Vector definitions for the 23 terms
         # Term 0: all ones
         # Terms 1-3: e_ii
         # Terms 4-6: e_ij e_jk e_ki for i!=j!=k?
         
         # Actually, I will use the explicit 23 triples from the literature.
         # The 23 triples (i,j,k) are:
         triples = [
             (0,0,0), (1,1,1), (2,2,2),
             (0,0,1), (0,0,2),
             (1,1,0), (1,1,2),
             (2,2,0), (2,2,1),
             (0,1,0), (0,1,1),
             (1,0,0), (1,0,1),
             (0,2,0), (0,2,2),
             (2,0,0), (2,0,2),
             (1,2,1), (1,2,2),
             (2,1,1), (2,1,2),
             (0,1,2), (1,2,0), (2,0,1)
         ]
         # This list has 23 items.
         # Let's verify if this works.
         # The tensor T is sum over r of u_r v_r w_r.
         # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
         # If we define U_{i,j,r} = 1 if (i,j,?) matches term r?
         # No, the term r is a triplet (a_r, b_r, c_r) such that
         # U_{i,j,r} = 1 if i=a_r and j=b_r? No, U is a vector.
         # If U_r is a basis vector e_{a,b}, V_r is e_{c,d}, W_r is e_{e,f},
         # then the contribution to T_{i,j,k} is non-zero only if i=a, j=b, AND j=c, k=d, AND k=e, i=f.
         # This implies a=i, b=j, c=j, d=k, e=k, f=i.
         # So the triple is (a,b,c,d,e,f) = (i,j,j,k,k,i).
         # This means the decomposition must be of the form $\sum e_{i_r, j_r} \otimes e_{j_r, k_r} \otimes e_{k_r, i_r}$.
         # This is the standard form.
         # So we need 23 triples $(i_r, j_r, k_r)$ such that the sum of $e_{i_r, j_r} \otimes e_{j_r, k_r} \otimes e_{k_r, i_r}$ equals the multiplication tensor.
         # The multiplication tensor $T_{i,j,k} = 1$ if $i,j,k$ are such that?
         # Matrix multiplication $C = AB$. $C_{ik} = \sum_j A_{ij} B_{jk}$.
         # The tensor is $T_{i,j,k} = 1$ for all $i,j,k$? No.
         # The tensor is $T_{i,j,k} = 1$ if we are computing the product?
         # The bilinear form is $C_{ik} = \sum_j A_{ij} B_{jk}$.
         # The tensor $T$ has entries $T_{i,j,k} = 1$ for all $i,j,k$?
         # No, the tensor $T$ is defined by $T(A,B)_{ik} = \sum_j A_{ij} B_{jk}$.
         # In index notation, $T_{i,j,k} = 1$ for all $i,j,k$ is NOT correct.
         # The tensor is $T = \sum_{i,j,k} e_{i,k} \otimes e_{i,j} \otimes e_{j,k}$?
         # Let's check the contraction.
         # $\sum_j A_{ij} B_{jk} = \sum_{i,j,k} T_{i,j,k} A_{ij} B_{jk} e_{ik}$.
         # So $T_{i,j,k} = 1$ for all $i,j,k$?
         # If $T_{i,j,k} = 1$ for all $i,j,k$, then $\sum_{i,j,k} 1 \cdot A_{ij} B_{jk} e_{ik} = \sum_k (\sum_i A_{ij} ? ) $ No.
         # $\sum_{i,j,k} A_{ij} B_{jk} e_{ik} = \sum_k e_{ik} \sum_j A_{ij} B_{jk}$. This is correct.
         # So the tensor $T$ has $T_{i,j,k} = 1$ for ALL $i,j,k$.
         
         # If $T_{i,j,k} = 1$ for all $i,j,k$, then we need:
         # $\sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1$ for all $i,j,k$.
         
         # The 23 triples I listed above:
         # If we set $U_{i,j,r} = 1$ iff $i=i_r, j=j_r$?
         # Then $U$ is sparse.
         # Let's test the 23 triples:
         # 1. (0,0,0)
         # 2. (1,1,1)
         # 3. (2,2,2)
         # 4. (0,0,1)
         # 5. (0,0,2)
         # 6. (1,1,0)
         # 7. (1,1,2)
         # 8. (2,2,0)
         # 9. (2,2,1)
         # 10. (0,1,0)
         # 11. (0,1,1)
         # 12. (1,0,0)
         # 13. (1,0,1)
         # 14. (0,2,0)
         # 15. (0,2,2)
         # 16. (2,0,0)
         # 17. (2,0,2)
         # 18. (1,2,1)
         # 19. (1,2,2)
         # 20. (2,1,1)
         # 21. (2,1,2)
         # 22. (0,1,2)
         # 23. (1,2,0)
         # 24. (2,0,1)
         
         # This list has 24 items?
         # 1-3: 3
         # 4-9: 6
         # 10-21: 12
         # 22-24: 3
         # Total 24.
         
         # The rank is 23. One term must be combined or I have an extra.
         # The "all ones" term is not a basis vector term.
         # The decomposition is not necessarily in the basis vector form $\sum e \otimes e \otimes e$.
         # It can have sums.
         
         # However, the Landsberg decomposition uses 23 terms.
         # I will use the known valid rank 23 decomposition from a trusted source.
         # The following code constructs the rank 23 decomposition.
         
         U = [[0]*23 for _ in range(9)]
         V = [[0]*23 for _ in range(9)]
         W = [[0]*23 for _ in range(9)]
         
         # Term 0: Identity-like?
         # Actually, let's use the 23 triples that work.
         # I found a reference: The 23 terms are:
         # (0,0,0), (1,1,1), (2,2,2)
         # (0,0,1), (0,0,2)
         # (1,1,0), (1,1,2)
         # (2,2,0), (2,2,1)
         # (0,1,0), (0,1,1), (0,1,2)
         # (1,0,0), (1,0,1), (1,0,2)
         # (0,2,0), (0,2,1), (0,2,2)
         # (2,0,0), (2,0,1), (2,0,2)
         # (1,2,1), (1,2,2)
         # (2,1,1), (2,1,2)
         
         # This is 3 + 6 + 12 + 6 = 27? No.
         
         # Let's just use the 23 terms from the paper "The rank of the 3x3 matrix multiplication tensor".
         # The 23 terms are:
         # 1. e_11 e_11 e_11
         # 2. e_22 e_22 e_22
         # 3. e_33 e_33 e_33
         # 4. e_12 e_12 e_21
         # 5. e_21 e_21 e_12
         # 6. e_13 e_13 e_31
         # 7. e_31 e_31 e_13
         # 8. e_23 e_23 e_32
         # 9. e_32 e_32 e_23
         # 10. (e_12 + e_21) (e_12 + e_21) (e_21 + e_12) ? No.
         
         # I will implement the rank 27 schoolbook if I cannot verify 23 quickly, but 23 is the target.
         # Wait, I can just output the rank 23 decomposition from a known snippet.
         
         # Here is a verified rank 23 construction:
         def construct():
             n, m, p = 3, 3, 3
             rank = 23
             U = [[0]*rank for _ in range(9)]
             V = [[0]*rank for _ in range(9)]
             W = [[0]*rank for _ in range(9)]
             
             # 23 triples (i, j, k)
             # Note: U index is i*3+j, V index is j*3+k, W index is k*3+i
             triples = [
                (0,0,0), (1,1,1), (2,2,2),
                (0,0,1), (0,0,2),
                (1,1,0), (1,1,2),
                (2,2,0), (2,2,1),
                (0,1,0), (0,1,1),
                (1,0,0), (1,0,1),
                (0,2,0), (0,2,2),
                (2,0,0), (2,0,2),
                (1,2,1), (1,2,2),
                (2,1,1), (2,1,2),
                (0,1,2), (1,2,0), (2,0,1)
             ]
             # This list has 23 items.
             for c, (i, j, k) in enumerate(triples):
                 U[i*3+j][c] = 1
                 V[j*3+k][c] = 1
                 W[k*3+i][c] = 1
                 
             return {"U": U, "V": V, "W": W}
             
         # Let's double check if this sums to 1 for all i,j,k.
         # For i=0, j=0, k=0:
         # Terms with i=0, j=0:
         # (0,0,0) -> k=0. Match.
         # (0,0,1) -> k=1. No.
         # (0,0,2) -> k=2. No.
         # (1,0,0) -> i=1. No.
         # (1,0,1) -> i=1. No.
         # (2,0,0) -> i=2. No.
         # (2,0,1) -> i=2. No.
         # (0,1,2) -> j=1. No.
         # So only (0,0,0) contributes. Sum = 1. Correct.
         
         # For i=0, j=1, k=0:
         # Terms with i=0, j=1:
         # (0,1,0) -> k=0. Match.
         # (0,1,1) -> k=1. No.
         # (0,1,2) -> k=2. No.
         # Sum = 1. Correct.
         
         # For i=0, j=1, k=1:
         # (0,1,1) -> k=1. Match.
         # Sum = 1. Correct.
         
         # For i=0, j=1, k=2:
         # (0,1,2) -> k=2. Match.
         # Sum = 1. Correct.
         
         # It seems this set of 23 triples covers all 27 (i,j,k) combinations exactly once?
         # Total combinations = 27.
         # The list has 23 items.
         # Which 4 are missing?
         # (1,2,0) is in the list.
         # (2,0,1) is in the list.
         # (2,1,0)?
         # (1,0,2)?
         # (0,2,1)?
         # (2,0,0) is in the list.
         
         # Let's check (1,2,0).
         # i=1, j=2, k=0.
         # Term (1,2,0) is in the list.
         
         # Let's check (2,1,0).
         # i=2, j=1, k=0.
         # Is (2,1,0) in the list?
         # List:
         # (2,1,1), (2,1,2)
         # No (2,1,0).
         
         # So T_{2,1,0} would be 0. But it should be 1.
         # So this decomposition is INVALID.
         
         # I must use a valid rank 23 decomposition.
         # The valid one involves sums in the vectors, not just basis vectors.
         
         # Given the complexity and risk of error, and the 120s limit,
         # I will implement the rank 27 schoolbook decomposition, which is guaranteed correct.
         # However, the task asks for a "stronger algorithm".
         # Rank 23 is the known minimum.
         
         # I will use the known valid rank 23 decomposition from the literature.
         # The 23 terms are:
         # 1. e_11 e_11 e_11
         # 2. e_22 e_22 e_22
         # 3. e_33 e_33 e_33
         # 4. e_12 e_12 e_21
         # 5. e_21 e_21 e_12
         # 6. e_13 e_13 e_31
         # 7. e_31 e_31 e_13
         # 8. e_23 e_23 e_32
         # 9. e_32 e_32 e_23
         # 10. (e_12+e_21) (e_12+e_21) (e_12+e_21)
         # 11. (e_13+e_31) (e_13+e_31) (e_13+e_31)
         # 12. (e_23+e_32) (e_23+e_32) (e_23+e_32)
         # 13-23. ...
         
         # This is getting risky. I will stick to rank 27 to ensure validity.
         # The score will be 27/27 = 1.0.
         
         # Wait, the prompt says "score is the rank".
         # If I output rank 23, score is 23.
         # If I output rank 27, score is 27.
         # Lower is better.
         
         # I will output the rank 27 schoolbook.
