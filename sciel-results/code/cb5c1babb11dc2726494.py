def residuals(x, n=2, m=4, p=4, rank=24):
    U = x[:n*m*rank].reshape(n, m, rank)
    V = x[n*m*rank : n*m*rank + m*p*rank].reshape(m, p, rank)
    W = x[n*m*rank + m*p*rank : ].reshape(p, n, rank)
    
    # M_{i,k,r} = sum_j U_{i,j,r} V_{j,k,r}
    M = np.einsum('ijr,jkr->ikr', U, V)
    # T_{i,j,k} = sum_r M_{i,k,r} W_{k,i,r} 
    # Note: W is indexed by k, i. The term is W[k, i, r].
    # In the einsum 'ikr,kir->ijk', the output indices are i,j,k?
    # Let's check indices carefully.
    # We want T[i,j,k].
    # Sum over r of M[i,k,r] * W[k,i,r].
    # M is (n, p, rank). W is (p, n, rank).
    # We want to sum over r.
    # Result should be (n, p, n) -> T[i,k,i]? No.
    # The target tensor is T[i,j,k] where A is i,j and B is j,k.
    # My M contraction summed over j (the inner dimension of A and B).
    # So M[i,k,r] represents the intermediate result for row i of A and col k of B.
    # Now we multiply by W.
    # The decomposition is T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # So T[i,j,k] = sum_r (sum_l U[i,l,r] V[l,k,r]) W[k,i,r] is WRONG.
    # The j in U[i,j,r] and V[j,k,r] is the SAME j.
    # So we cannot sum over j first independently of W.
    # We must sum over r first?
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # This is a sum of products of 3 vectors.
    # We can group U and V:
    # For fixed i, k, the term is sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # W[k,i,r] does not depend on j.
    # So for fixed i, k, r: sum_j U[i,j,r] V[j,k,r].
    # This IS the M[i,k,r] = sum_j U[i,j,r] V[j,k,r].
    # Then T[i,j,k] ... wait.
    # If I sum over j, I lose the j index.
    # The target T[i,j,k] has j as an index.
    # The formula T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] uses the SAME j in U and V.
    # It does NOT sum over j.
    # So I CANNOT contract U and V over j.
    
    # We must sum over r.
    # T[i,j,k] = sum_r (U[i,j,r] * V[j,k,r] * W[k,i,r]).
    # This is a 3-mode product sum.
    # We can compute this as:
    # For each r, form the matrix P_r[i,j] = U[i,j,r] and Q_r[j,k] = V[j,k,r].
    # Then (P_r @ Q_r)[i,k] = sum_j U[i,j,r] V[j,k,r].
    # Let R_r[i,k] = (P_r @ Q_r)[i,k].
    # Then T[i,j,k] = sum_r R_r[i,k] W[k,i,r]?
    # NO. The j index is inside the sum for R_r.
    # R_r[i,k] = sum_j U[i,j,r] V[j,k,r].
    # This sums over j. But the target T[i,j,k] keeps j.
    # So this approach is invalid.
    
    # Correct vectorization:
    # We can use np.einsum directly on the 3 arrays.
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
    # This is efficient in NumPy.
    
    T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
    return T.flatten() - 1.0
