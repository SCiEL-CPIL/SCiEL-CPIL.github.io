    def verify(U, V, W):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        R = Ua.shape[1]
        
        # Reshape
        # Ua: (n*m, R) -> (n, m, R)
        U3 = Ua.reshape(n, m, R)
        # Va: (m*p, R) -> (m, p, R)
        V3 = Va.reshape(m, p, R)
        # Wa: (p*n, R) -> (p, n, R) ? 
        # Contract says W[k*n+i, r]. Index is k*n + i.
        # So Wa is (p*n, R). Reshape to (p, n, R).
        W3 = Wa.reshape(p, n, R)
        
        # We need to check:
        # sum_r W_{ik, r} U_{ij, r} V_{j'k, r} == delta_{j, j'}
        # Note: W3 is indexed [k, i, r]. So W3[k, i, r] = W_{ik, r}.
        # U3 is indexed [i, j, r].
        # V3 is indexed [j', k, r].
        
        # LHS[i, j, j', k] = sum_r W3[k, i, r] * U3[i, j, r] * V3[j', k, r]
        # This is a 4D tensor.
        # We can compute this using einsum.
        # 'kir, ijr, jkr -> ijjk'? 
        # Indices:
        # W: k, i, r
        # U: i, j, r
        # V: j', k, r (using j' as the first index of V)
        
        # Let's name indices carefully.
        # U3: i, j, r
        # V3: jp, k, r  (where jp is the row index of V, corresponding to j in A and k in B? No.)
        # V is (m*p, R). Row index is j*p + k.
        # So V3[j, k, r] corresponds to V[j*p+k, r].
        # This V term is multiplied by B[j, k].
        # In the product $C_{ik} = \sum_j A_{ij} B_{jk}$, the B index is $j, k$.
        # So the V tensor should align with $B_{jk}$.
        # So V3[j, k, r] is the coefficient for $B_{jk}$.
        
        # The term in the sum is $U_{ij, r} A_{ij} V_{jk, r} B_{jk} W_{ik, r}$.
        # Wait, the previous analysis said:
        # $C_{ik} = \sum_r W_{ik, r} (\sum_j U_{ij, r} A_{ij}) (\sum_{j'} V_{j'k, r} B_{j'k})$.
        # This expands to $\sum_{j, j'} W_{ik, r} U_{ij, r} V_{j'k, r} A_{ij} B_{j'k}$.
        # We need this to equal $\sum_j A_{ij} B_{jk}$.
        # So we need $\sum_r W_{ik, r} U_{ij, r} V_{j'k, r} = \delta_{j, j'}$.
        
        # So we compute T[i, j, j', k] = sum_r W3[k, i, r] * U3[i, j, r] * V3[j', k, r].
        # And check if T[i, j, j', k] == delta_{j, j'} for all i, k.
        
        # Einsum: 'kir, ijr, jkr -> ijjk'
        # W3: k, i, r
        # U3: i, j, r
        # V3: j', k, r  (Note: V3 second index is k, first is j')
        
        Tt = np.einsum('kir,ijr,jkr->ijjk', W3, U3, V3)
        
        # Target: delta_{j, j'}
        # Create target tensor of shape (n, m, m, p)
        # target[i, j, j', k] = 1 if j == j' else 0
        # It is independent of i and k.
        
        # Check
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        expected = 1.0 if j == jp else 0.0
                        if abs(Tt[i, j, jp, k] - expected) > 1e-6:
                            return False
        return True
    