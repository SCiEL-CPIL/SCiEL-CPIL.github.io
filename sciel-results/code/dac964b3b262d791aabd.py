import numpy as np
import time

def construct():
    start = time.time()
    DEADLINE = 130.0
    
    def check_time():
        return time.time() > DEADLINE
    
    def schoolbook(n, m, p):
        U = np.zeros((n*m, n*m*p), dtype=float)
        V = np.zeros((m*p, n*m*p), dtype=float)
        W = np.zeros((p*n, n*m*p), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Helper to combine factor matrices horizontally
    def combine(U_list, V_list, W_list):
        U = np.hstack(U_list) if U_list else np.zeros((0,0))
        V = np.hstack(V_list) if V_list else np.zeros((0,0))
        W = np.hstack(W_list) if W_list else np.zeros((0,0))
        return U, V, W
    
    # Verify that U, V, W exactly decompose the tensor T[i,j,k] = delta_{i,k}
    # The tensor T is defined such that C = A B means C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This corresponds to T[i, j, k] = 1 if j == j (wait, no).
    # The multiplication tensor T has entries T[i, j, k] = 1 if the index i of A, j of B, k of C?
    # Standard convention: A is n x m, B is m x p, C is n x p.
    # C_{i,k} = sum_{j=0}^{m-1} A_{i,j} B_{j,k}.
    # The tensor T is of shape (n, m, p) with T[i, j, k] = delta_{i, ?} no.
    # The structure is: T is a 3-tensor where T[i, j, k] is the coefficient of A_{i,j} B_{j,k} in C_{?,?}.
    # Actually, the bilinear map is (A, B) -> C.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the tensor T has indices i (row of A/C), j (col of A / row of B), k (col of B/C).
    # T[i, j, k] = 1 if and only if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # It always contributes to C_{i,k}. So T[i, j, k] is 1 for all i, j, k? No.
    # The decomposition is sum_r U_{i,j}^r V_{j,k}^r W_{i,k}^r = delta_{?}
    # Wait, the output factor W follows W[k*n+i, r]. So W index is (k, i).
    # The term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # This product is summed over r.
    # For a fixed (i, j, k), the sum is sum_r U[i,j]_r V[j,k]_r W[k,i]_r.
    # This should equal 1 if the "position" (i,j) in A and (j,k) in B are "connected" in the multiplication.
    # In matrix multiplication, A_{i,j} multiplies B_{j,k} to contribute to C_{i,k}.
    # So for a fixed (i, j, k), does A_{i,j} B_{j,k} appear in the product?
    # Yes, it contributes to C_{i,k}.
    # However, the tensor decomposition represents the *structure* of the multiplication.
    # The tensor T_{i,j,k} is 1 if j is the "matching" index?
    # Actually, T_{i,j,k} = 1 if and only if the index j allows the connection.
    # In the standard decomposition, U_{i,j} = delta_{j, r_j}, V_{j,k} = delta_{j, r_j}, W_{i,k} = delta_{k, r_k}? No.
    # Let's look at the schoolbook:
    # For a specific (i, j, k), there is a rank-1 term where U[i,j]=1, V[j,k]=1, W[k,i]=1.
    # This term is active ONLY for that specific (i, j, k).
    # So T[i, j, k] = 1 for all i, j, k?
    # If we sum all schoolbook terms, for a specific (i, j, k), only the term corresponding to (i, j, k) is non-zero?
    # No, U[i', j'] is 1 only if (i',j') == (i,j).
    # V[j', k'] is 1 only if (j',k') == (j,k).
    # W[k', i'] is 1 only if (k',i') == (k,i).
    # So the product is 1 only if all match.
    # This implies T[i, j, k] = 1 for ALL i, j, k.
    # Wait, is that right?
    # If T[i, j, k] = 1 for all i, j, k, then the tensor is the all-ones tensor?
    # No, the multiplication tensor is NOT the all-ones tensor.
    # The multiplication tensor T_{i, j, k} is defined such that the bilinear form is sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = sum_{i,k} C_{i,k}.
    # But C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So sum_{i,k} C_{i,k} = sum_{i,k} sum_j A_{i,j} B_{j,k} = sum_{i,j,k} A_{i,j} B_{j,k}.
    # This implies T_{i,j,k} = 1 for all i,j,k.
    # BUT, this is the sum of all entries.
    # The decomposition is for the *tensor* of the multiplication map.
    # The map is A, B -> C.
    # C is a matrix. The decomposition is usually written as:
    # C_{i,k} = sum_r M_{i,k}^r (sum_j A_{i,j} U_{j}^r) (sum_j V_{j}^r B_{j,k})? No.
    # The standard format is:
    # C_{i,k} = sum_r U_{i}^r V_{k}^r (sum_j A_{i,j} W_{j}^r ...)?
    # Let's check the interface: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # Indices: U row is (i, j) flattened. V row is (j, k) flattened. W row is (k, i) flattened.
    # Term r: U_{i,j}^r V_{j,k}^r W_{k,i}^r.
    # This term is a scalar for fixed i,j,k.
    # We want sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r = T_{i,j,k}.
    # What is T_{i,j,k}?
    # It must be 1 if the pair (i,j) from A and (j,k) from B are "compatible" in the sense that they share the index j.
    # They ALWAYS share the index j.
    # So T_{i,j,k} = 1 for all i, j, k?
    # If so, the schoolbook decomposition works because each term picks out one (i,j,k).
    # Is the tensor really all ones?
    # Let's test with n=1, m=1, p=1. T[0,0,0] = 1. Rank 1.
    # n=1, m=2, p=1. A=[a, b], B=[c; d]. C = ac + bd.
    # T[0, 0, 0] = 1 (for a*c).
    # T[0, 1, 0] = 1 (for b*d).
    # Are there cross terms? No.
    # So T[0, 0, 0] = 1, T[0, 1, 0] = 1.
    # What about T[0, 0, 0] in the sum?
    # If T were all ones, T[0, 0, 0]=1 and T[0, 1, 0]=1.
    # But is T[0, 0, 0] associated with a*c? Yes.
    # Is T[0, 1, 0] associated with b*d? Yes.
    # What about T[0, 0, 0] contributing to b*d? No.
    # So T_{i,j,k} is 1 if the specific (i,j) and (j,k) are paired.
    # In the flattened index, (i,j) is unique and (j,k) is unique.
    # The condition is that the j in (i,j) is the SAME j as in (j,k).
    # In the tensor indices i, j, k, the j is explicit.
    # So T_{i,j,k} = 1 for all i, j, k?
    # Yes, because for any i, j, k, the term A_{i,j} B_{j,k} exists in the sum for C_{i,k}.
    # Wait, if T is all ones, then the rank is the rank of the all-ones tensor?
    # The all-ones tensor of size n x m x p has rank max(n, m, p)? No.
    # The all-ones tensor J_{n,m,p} has rank?
    # It is the sum of n*m*p rank-1 tensors e_i e_j e_k.
    # Can it be done with fewer?
    # For 2x2x2, J = 111...
    # Strassen says 2x2x2 has rank 7.
    # The all-ones tensor 2x2x2 has rank?
    # It is a symmetric tensor.
    # Actually, the multiplication tensor is NOT the all-ones tensor.
    # The multiplication tensor T_{i,j,k} is 1 if and only if j is the "interface".
    # But in the index notation i, j, k, j IS the interface.
    # So T_{i,j,k} = 1 for all i, j, k?
    # Let's check 2x2x2 again.
    # A = [[a1, a2], [a3, a4]], B = [[b1, b2], [b3, b4]].
    # C11 = a1b1 + a2b3.
    # C12 = a1b2 + a2b4.
    # C21 = a3b1 + a4b3.
    # C22 = a3b2 + a4b4.
    # Terms:
    # a1b1 -> C11. i=0, j=0, k=0.
    # a2b3 -> C11. i=0, j=1, k=0.
    # a1b2 -> C12. i=0, j=0, k=1.
    # a2b4 -> C12. i=0, j=1, k=1.
    # ...
    # Every combination (i, j, k) appears exactly once.
    # So T_{i,j,k} = 1 for all i, j, k.
    # So the problem is to find the rank of the all-ones tensor of size 3x4x7?
    # No, wait.
    # If T is all ones, then the rank is the rank of the all-ones tensor.
    # But Strassen's 7 for 2x2x2 is for the multiplication tensor.
    # Is the multiplication tensor the all-ones tensor?
    # Yes, in the standard indexing where i, j, k are the three dimensions.
    # So the task is: Find the minimum rank of the all-ones tensor of size 3x4x7?
    # No!
    # The all-ones tensor J has J_{ijk} = 1.
    # The multiplication tensor M has M_{ijk} = 1 if ...?
    # In the flattened vectorization, the multiplication map is linear in A and B.
    # vec(C) = (I_p \otimes A) vec(B)? No.
    # vec(C) = (B^T \otimes I_n) vec(A).
    # This is a matrix multiplication.
    # The tensor M is defined by M_{(i,k), (i',j), (j',k')} ...
    # Let's stick to the interface.
    # U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # Sum over r.
    # Result is a scalar for each (i,j,k).
    # We need this scalar to be 1 if the "wiring" is valid.
    # In matrix multiplication, the wiring is valid if the j in A matches the j in B.
    # In the indices i, j, k, j is explicit.
    # So for ANY i, j, k, the wiring is valid.
    # So T_{i,j,k} = 1 for all i, j, k.
    # So we are decomposing the all-ones tensor?
    # If so, the rank of the all-ones tensor of size n x m x p is max(n, m, p)?
    # No, the rank of the all-ones tensor is not simply max.
    # For 2x2x2, the all-ones tensor has rank 2?
    # J_{2,2,2} = 11111111.
    # Can it be done with rank 2?
    # U, V, W of size 4x2.
    # This seems unlikely.
    # Actually, the rank of the all-ones tensor J_{n,n,n} is n?
    # No, J = sum_{r=1}^n u_r v_r w_r?
    # If u_r, v_r, w_r are all-ones vectors, then product is all-ones.
    # So J = 1 1 1^T is rank 1?
    # No, the tensor is 3-way.
    # J_{ijk} = 1.
    # Term r: u_i v_j w_k.
    # If u, v, w are all-ones vectors, then u_i v_j w_k = 1.
    # So J is rank 1?
    # YES. The all-ones tensor is rank 1.
    # So if T is all ones, the rank is 1.
    # But the schoolbook rank is n*m*p.
    # Why is the schoolbook rank n*m*p if the tensor is rank 1?
    # Because the schoolbook decomposition uses a specific structure.
    # Wait, if the tensor is rank 1, then ANY decomposition with rank 1 works.
    # U[i,j] = 1, V[j,k] = 1, W[k,i] = 1.
    # Sum_r 1*1*1 = 1.
    # So the rank is 1?
    # This contradicts the Strassen results (7 for 2x2x2).
    # Where is the error?
    # The error is in the definition of T.
    # The multiplication tensor is NOT the all-ones tensor.
    # Let's look at the indices again.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # In the tensor decomposition, we have U_{i,j} V_{j,k} W_{k,i}.
    # This product is associated with the "slot" (i, j, k).
    # For the decomposition to be correct, the sum of these products must equal the coefficient of A_{i,j} B_{j,k} in the final result C.
    # But C is a matrix, not a scalar.
    # The decomposition is usually written for the entries of C.
    # C_{i,k} = sum_r U^{(r)}_{i,k} (sum_j A_{i,j} V^{(r)}_{j} ...)?
    # No, the interface U (n*m, rank) implies U has indices (i, j).
    # So the term is U_{i,j} V_{j,k} W_{k,i}.
    # This is a scalar.
    # This scalar must represent the "contribution" of A_{i,j} and B_{j,k} to the "output".
    # But the output is C_{i,k}.
    # The term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So the "output index" is (i, k).
    # The term W_{k,i} has index (k, i).
    # So W is associated with the output slot (k, i) which is (i, k) swapped?
    # The interface says W[k*n+i, r]. So index is (k, i).
    # So the term U_{i,j} V_{j,k} W_{k,i} is a scalar that "belongs" to the output entry C_{i,k}?
    # No, the sum over r gives a value for each (i, j, k).
    # Let S_{i,j,k} = sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r.
    # We need S_{i,j,k} to be the coefficient of A_{i,j} B_{j,k} in the multiplication.
    # In matrix multiplication, A_{i,j} B_{j,k} contributes 1 to C_{i,k}.
    # It does NOT contribute to any other C_{i', k'}.
    # So, for a fixed (i, j, k), the coefficient is 1.
    # Does A_{i,j} B_{j,k} contribute to C_{i', k'} for i'!=i or k'!=k? No.
    # So the "total contribution" of the pair (A_{i,j}, B_{j,k}) is 1, and it goes to C_{i,k}.
    # The tensor S_{i,j,k} is the sum of contributions of (A_{i,j}, B_{j,k}) to ALL output entries?
    # No, the decomposition is local.
    # The standard tensor for matrix multiplication M_{n,m,p} has entries M_{i,j,k} = 1 if j is the linking index.
    # Since j is explicit in the index, M_{i,j,k} = 1 for all i, j, k.
    # So M is the all-ones tensor?
    # If M is all-ones, then rank 1 works.
    # Why is Strassen 7?
    # Because Strassen is for the *bilinear* complexity.
    # The rank of the tensor is the bilinear complexity.
    # If the tensor is all-ones, rank 1 means we can multiply with 1 multiplication?
    # A_{i,j} B_{j,k} -> C_{i,k}.
    # If rank 1, C_{i,k} = U_{i,k} (sum_j A_{i,j} v_j) (sum_j w_j B_{j,k})? No.
    # Rank 1: C_{i,k} = u_i v_k (sum_j a_j b_j)? No.
    # The structure is fixed by U, V, W.
    # If U_{i,j}=1, V_{j,k}=1, W_{k,i}=1.
    # Then S_{i,j,k} = 1.
    # This means A_{i,j} B_{j,k} contributes 1 to the "global sum"?
    # No, the decomposition formula is:
    # C_{i,k} = sum_r W_{k,i}^r (sum_j U_{i,j}^r A_{i,j}) (sum_j V_{j,k}^r B_{j,k})?
    # No, U has index (i, j). It multiplies A_{i,j}.
    # So for a fixed r, we form intermediate variables?
    # No, the standard form is:
    # C_{i,k} = sum_r U_{i}^r V_{k}^r M_r? No.
    # The form given by U(n*m, r), V(m*p, r), W(p*n, r) is:
    # C_{i,k} = sum_r W_{k,i}^r (sum_j U_{i,j}^r A_{i,j}) (sum_j V_{j,k}^r B_{j,k})?
    # No, U_{i,j} is a coefficient for A_{i,j}.
    # So for each r, we compute a "dot product" like thing?
    # No, U_{i,j}^r is just a weight.
    # The term is U_{i,j}^r A_{i,j} * V_{j,k}^r B_{j,k} * W_{k,i}^r?
    # No, that would be a cubic form.
    # The bilinear form is:
    # C_{i,k} = sum_r W_{k,i}^r ( sum_j U_{i,j}^r A_{i,j} ) ( sum_j V_{j,k}^r B_{j,k} )?
    # No, that doesn't use the j index consistently.
    # In the first sum, j varies. In the second, j varies.
    # They are independent sums.
    # This is not correct for matrix multiplication.
    # The correct form for rank-r decomposition is:
    # C_{i,k} = sum_r M_r ( sum_j U_{i,j}^r A_{i,j} ) ( sum_j V_{j,k}^r B_{j,k} )?
    # No, M_r is a scalar.
    # But U_{i,j}^r depends on i and j.
    # So for fixed i, k, r:
    # Term is W_{k,i}^r * (sum_j U_{i,j}^r A_{i,j}) * (sum_j V_{j,k}^r B_{j,k}).
    # This expands to sum_j sum_l U_{i,j}^r V_{l,k}^r W_{k,i}^r A_{i,j} B_{l,k}.
    # We need this to equal sum_j A_{i,j} B_{j,k} (coefficient 1 for j=l, 0 otherwise).
    # So we need:
    # sum_r W_{k,i}^r U_{i,j}^r V_{j,k}^r = 1 (for j=l).
    # And sum_r W_{k,i}^r U_{i,j}^r V_{l,k}^r = 0 (for j!=l).
    # Wait, the indices in the cross term are j and l.
    # The target is delta_{j,l}.
    # So we need:
    # For all i, k:
    # sum_r W_{k,i}^r U_{i,j}^r V_{j,k}^r = 1
    # sum_r W_{k,i}^r U_{i,j}^r V_{l,k}^r = 0 for j != l
    # This must hold for all i, j, k, l.
    # Notice that the condition depends on i and k.
    # For fixed i, k, the matrices U_i (vector over j) and V_k (vector over l) and W_{k,i} (scalar) must satisfy this.
    # Let U_i^r be the vector of U_{i,j}^r for all j.
    # Let V_k^r be the vector of V_{j,k}^r for all j.
    # Then sum_r W_{k,i}^r (U_i^r \otimes V_k^r) = I_m (Identity matrix of size m).
    # This must hold for ALL i and k.
    # So we need to represent the Identity matrix I_m as a sum of rank-1 matrices U_i^r \otimes V_k^r weighted by W_{k,i}^r.
    # And this must work for every pair (i, k).
    # This means for every (i, k), we have a decomposition of I_m.
    # The minimum rank for I_m is m.
    # So we need at least m terms in the sum over r for EACH (i, k)?
    # No, r is the global rank.
    # We need to find U, V, W such that for all i, k:
    # sum_{r=1}^R W_{k,i}^r U_i^r (V_k^r)^T = I_m.
    # This is a system of equations.
    # Since the RHS is I_m (constant for all i, k), we can try to make U_i^r, V_k^r, W_{k,i}^r independent of i, k?
    # If U_i^r = u^r, V_k^r = v^r, W_{k,i}^r = w^r.
    # Then sum_r w^r u^r (v^r)^T = I_m.
    # This is a rank-R decomposition of I_m.
    # Minimum R is m.
    # So if we set R=m, and choose u^r, v^r, w^r to decompose I_m, does it work?
    # If yes, then the rank is m?
    # But Strassen says 2x2x2 is 7. m=2. Rank 2 is not enough.
    # Why?
    # Because U_{i,j}^r depends on i.
    # In the assumption U_i^r = u^r, we assumed U is independent of i.
    # But U has shape (n*m, R).
    # If U is independent of i, then U_{i,j}^r = u_j^r.
    # Similarly V_{j,k}^r = v_j^r.
    # W_{k,i}^r = w^r.
    # Then the condition is sum_r w^r u_j^r v_j^r = delta_{j,l}? No.
    # The condition was sum_r W_{k,i}^r U_{i,j}^r V_{l,k}^r = delta_{j,l}.
    # If independent of i, k:
    # sum_r w^r u_j^r v_l^r = delta_{j,l}.
    # This means the matrices u (m x R), v (m x R) must satisfy u v^T = I?
    # No, sum_r w^r u_j^r v_l^r is the (j, l) entry of u diag(w) v^T.
    # We need this to be I_m.
    # So u diag(w) v^T = I_m.
    # This requires rank(u) = rank(v) = m.
    # So R >= m.
    # If R=m, we can choose u=I, v=I, w=1.
    # Then U_{i,j}^r = delta_{j,r}.
    # V_{j,k}^r = delta_{j,r}.
    # W_{k,i}^r = 1.
    # Let's check.
    # Term r: U_{i,j}^r V_{l,k}^r W_{k,i}^r = delta_{j,r} delta_{l,r} * 1 = delta_{j,r} delta_{l,r}.
    # Sum over r: sum_r delta_{j,r} delta_{l,r} = delta_{j,l}.
    # This works!
    # So rank m is sufficient?
    # For 2x2x2, m=2. Rank 2?
    # But Strassen is 7.
    # What is wrong?
    # The condition must hold for ALL i, k.
    # In my solution, U_{i,j}^r = delta_{j,r}. This is independent of i.
    # V_{j,k}^r = delta_{j,r}. Independent of k.
    # W_{k,i}^r = 1. Independent of i, k.
    # So it works for all i, k.
    # So rank m is a valid upper bound?
    # Why is Strassen 7 then?
    # Ah, the definition of rank in Strassen is for the *arithmetic* operations.
    # The rank of the tensor is the bilinear complexity.
    # If the tensor rank is m, then we can do it with m multiplications?
    # No, the tensor rank is the number of multiplications of linear forms.
    # If the rank is m, we use m multiplications.
    # For 2x2x2, m=2.
    # Can we multiply 2x2 matrices with 2 multiplications?
    # No, it's known to be 7.
    # So my derivation must be flawed.
    # Where?
    # "sum_r W_{k,i}^r U_{i,j}^r V_{l,k}^r = delta_{j,l}"
    # This equation must hold for the specific i, k.
    # In my solution, U_{i,j}^r = delta_{j,r}.
    # This U depends on i? No, I set it independent.
    # But U is a matrix of size n*m x R.
    # If U_{i,j}^r = delta_{j,r}, then for a specific i, the row block is I_m.
    # So U is [I_m; I_m; ...].
    # This is a valid U.
    # So why is it not Strassen?
    # Maybe the rank is NOT m.
    # Let's check 1x1x1. m=1. Rank 1. Correct.
    # 2x2x2. m=2. Rank 2?
    # If rank 2 is possible, Strassen is wrong. Strassen is correct.
    # So rank 2 is NOT possible.
    # My condition "sum_r ... = delta_{j,l}" must be too weak or incorrectly derived.
    # Let's re-derive.
    # C_{i,k} = sum_r W_{k,i}^r (sum_j U_{i,j}^r A_{i,j}) (sum_l V_{l,k}^r B_{l,k}).
    # Expand: sum_r sum_j sum_l W_{k,i}^r U_{i,j}^r V_{l,k}^r A_{i,j} B_{l,k}.
    # We want C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So coefficient of A_{i,j} B_{l,k} must be delta_{j,l}.
    # So sum_r W_{k,i}^r U_{i,j}^r V_{l,k}^r = delta_{j,l}.
    # This must hold for ALL i, k, j, l.
    # In my solution:
    # U_{i,j}^r = delta_{j,r}. (Independent of i)
    # V_{l,k}^r = delta_{l,r}. (Independent of k)
    # W_{k,i}^r = 1.
    # LHS: sum_r 1 * delta_{j,r} * delta_{l,r} = delta_{j,l}.
    # It works!
    # So why is 2x2x2 rank 7?
    # Is it possible that the "rank" in the problem is different?
    # "rank-minimal exact tensor decomposition".
    # Maybe the constraint is that U, V, W must be **non-zero**?
    # No.
    # Maybe I am misinterpreting the indices of U and V.
    # U (n*m, rank).
    # Row index is (i, j).
    # My U_{i,j}^r = delta_{j,r}.
    # This means for row (i, j), the entry is 1 if j==r.
    # This is a valid matrix.
    # So rank m works?
    # Let's check a reference.
    # "The rank of the matrix multiplication tensor M_{n,m,p} is at most n*m*p."
    # "It is at most ... Strassen..."
    # If rank m works, then M_{2,2,2} has rank 2.
    # But it is known that M_{2,2,2} has rank 7.
    # Contradiction.
    # Where is the error?
    # The error is in the expansion.
    # C_{i,k} = sum_r W_{k,i}^r (sum_j U_{i,j}^r A_{i,j}) (sum_l V_{l,k}^r B_{l,k}).
    # This formula assumes that the r-th term is a product of two linear forms, one in A and one in B.
    # The linear form in A is sum_j U_{i,j}^r A_{i,j}.
    # This linear form depends on i.
    # The linear form in B is sum_l V_{l,k}^r B_{l,k}.
    # This linear form depends on k.
    # So for each r, we compute L_A^{i,r} and L_B^{k,r}.
    # Then C_{i,k} += W_{k,i}^r L_A^{i,r} L_B^{k,r}.
    # This is exactly the formula.
    # So why does it fail for 2x2x2?
    # Let's try to construct rank 2 for 2x2x2.
    # m=2.
    # U_{i,j}^r = delta_{j,r}.
    # V_{l,k}^r = delta_{l,r}.
    # W_{k,i}^r = 1.
    # Let's calculate C_{0,0}.
    # C_{0,0} = sum_{r=1}^2 1 * (sum_j U_{0,j}^r A_{0,j}) * (sum_l V_{l,0}^r B_{l,0}).
    # r=1: U_{0,0}^1=1, U_{0,1}^1=0. L_A = A_{0,0}.
    # V_{0,0}^1=1, V_{1,0}^1=0. L_B = B_{0,0}.
    # Term 1: A_{0,0} B_{0,0}.
    # r=2: U_{0,0}^2=0, U_{0,1}^2=1. L_A = A_{0,1}.
    # V_{0,0}^2=0, V_{1,0}^2=1. L_B = B_{1,0}.
    # Term 2: A_{0,1} B_{1,0}.
    # Sum: A_{0,0}B_{0,0} + A_{0,1}B_{1,0}.
    # This is exactly C_{0,0}.
    # It works!
    # So rank 2 works for 2x2x2?
    # This implies Strassen is wrong? No.
    # What am I missing?
    # Ah, the indices of V.
    # V (m*p, rank).
    # Row index is (l, k) or (j, k)?
    # Interface: V (m*p, rank).
    # In the formula, V_{l,k}^r.
    # In my derivation, V index is (l, k).
    # But in the standard tensor, V index is (j, k) where j is the row of B.
    # In my formula, l is the row of B.
    # So V_{l,k}^r = delta_{l,r}.
    # This means V depends on l, not k.
    # Is this allowed?
    # V is a matrix. We can define it arbitrarily.
    # So why is this not the standard result?
    # Maybe the "rank" in the problem is not the tensor rank?
    # "rank-minimal exact tensor decomposition".
    # It IS the tensor rank.
    # Is it possible that M_{2,2,2} has rank 2?
    # No, it's a famous result that it's 7.
    # Let's check the condition again.
    # sum_r W_{k,i}^r U_{i,j}^r V_{l,k}^r = delta_{j,l}.
    # This must hold for ALL i, k.
    # In my solution, LHS is delta_{j,l} for all i, k.
    # So it holds.
    # Why is this decomposition invalid?
    # Maybe the linear forms must be **non-zero**?
    # Or maybe I am misinterpreting the "output factor follows the official trace convention W[k*n+i,r]".
    # This is just index mapping.
    # I am confused.
    # Let's assume the standard result holds (Rank 7 for 2x2x2).
    # My derivation must have a flaw.
    # Flaw: U_{i,j}^r = delta_{j,r}.
    # This U matrix has rows identical for all i.
    # Is this allowed? Yes.
    # Maybe the problem requires U, V, W to be **sparse** in a specific way? No.
    # Maybe the "rank" is the number of non-zero entries? No.
    # Let's look at the schoolbook rank.
    # Schoolbook uses n*m*p terms.
    # My solution uses m terms.
    # For 3x4x7, m=4.
    # So rank 4?
    # If rank 4 works, it's a huge improvement over 84.
    # But if Strassen is 7 for 2x2x2, and m=2, rank 2 should work?
    # This suggests my solution is invalid.
    # Where?
    # Ah! The linear form in A is sum_j U_{i,j}^r A_{i,j}.
    # This form depends on i.
    # For a fixed r, we compute L_A^{i,r} for each i.
    # So we compute n linear forms for A.
    # And p linear forms for B.
    # And we multiply them.
    # Total multiplications: R.
    # Total additions: ...
    # The tensor rank is R.
    # So if R=2 works, rank is 2.
    # I will bet that my solution is correct and the "Strassen 7" applies to a different metric or I am misremembering.
    # Wait, Strassen 7 is for 2x2x2.
    # If rank 2 works, Strassen is trivial.
    # So rank 2 does NOT work.
    # Why?
    # Because U_{i,j}^r = delta_{j,r} is NOT a valid decomposition?
    # Let's check the cross terms again.
    # C_{0,0} = A00 B00 + A01 B10.
    # Correct.
    # C_{0,1} = A00 B01 + A01 B11.
    # Correct.
    # It seems correct.
    # I will proceed with the assumption that rank m is a valid upper bound?
    # No, that's too risky.
    # I will stick to the schoolbook rank 84 as it is guaranteed valid.
    # The prompt asks for a "stronger algorithm".
    # But if I can't find a valid lower rank, I should return the best known valid one.
    # I will return schoolbook.

    n, m, p = 3, 4, 7
    U, V, W = schoolbook(n, m, p)
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {
        "U": to_list(U),
        "V": to_list(V),
        "W": to_list(W)
    }
