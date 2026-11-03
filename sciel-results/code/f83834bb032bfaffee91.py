    def calc_swap_delta(i, j, p, B_p, a, b):
        u = p[i]
        v = p[j]
        
        # Terms involving i and j
        # Sum over k != i, j
        # A[i, :] dot (B_p[v, p] - B_p[u, p])
        # This vector is B_p[v, p] - B_p[u, p].
        # B_p is b[:, p]. So B_p[k, l] = b[k, p[l]].
        # B_p[v, p] is a vector where index k is b[v, p[k]].
        # So we need dot(A[i, :], b[v, p] - b[u, p]).
        
        vec_v = b[v, p]
        vec_u = b[u, p]
        
        # Sum 1: sum_k A[i, k] * (vec_v[k] - vec_u[k])
        s1 = np.dot(a[i, :], vec_v - vec_u)
        
        # Sum 2: sum_k A[j, k] * (vec_u[k] - vec_v[k])
        s2 = np.dot(a[j, :], vec_u - vec_v)
        
        # Sum 3: sum_k A[k, i] * (b[p[k], v] - b[p[k], u])
        # b[p[k], v] is column v of b indexed by p[k].
        # This is b[:, v][p] - b[:, u][p]
        col_v = b[:, v]
        col_u = b[:, u]
        vec_cv = col_v[p]
        vec_cu = col_u[p]
        
        s3 = np.dot(a[:, i], vec_cv - vec_cu)
        
        # Sum 4: sum_k A[k, j] * (vec_cu[k] - vec_cv[k])
        s4 = np.dot(a[:, j], vec_cu - vec_cv)
        
        # Total sum over all k
        total = s1 + s2 + s3 + s4
        
        # Subtract terms where k=i and k=j
        # For k=i:
        # s1 term: A[i, i] * (b[v, p[i]] - b[u, p[i]]) = A[i, i] * (b[v, u] - b[u, u])
        # s2 term: A[j, i] * (b[u, p[i]] - b[v, p[i]]) = A[j, i] * (b[u, u] - b[v, u])
        # s3 term: A[i, i] * (b[p[i], v] - b[p[i], u]) = A[i, i] * (b[u, v] - b[u, u])
        # s4 term: A[i, j] * (b[p[i], u] - b[p[i], v]) = A[i, j] * (b[u, u] - b[u, v])
        
        # For k=j:
        # s1 term: A[i, j] * (b[v, p[j]] - b[u, p[j]]) = A[i, j] * (b[v, v] - b[u, v])
        # s2 term: A[j, j] * (b[u, p[j]] - b[v, p[j]]) = A[j, j] * (b[u, v] - b[v, v])
        # s3 term: A[j, i] * (b[p[j], v] - b[p[j], u]) = A[j, i] * (b[v, v] - b[v, u])
        # s4 term: A[j, j] * (b[p[j], u] - b[p[j], v]) = A[j, j] * (b[v, u] - b[v, v])
        
        # Wait, the derivation earlier included diagonal terms in the "diff" sums.
        # The "total" variable above includes k=i and k=j.
        # We need to subtract the contributions of k=i and k=j from the total sum,
        # and then add the correct direct contributions for the swap of i and j.
        
        # Actually, it's easier to just compute the full sum and correct.
        # The formula derived earlier:
        # Delta = Sum_{k!=i,j} ... + Direct Terms
        
        # Let's stick to the full sum and subtract i, j parts, then add direct.
        
        # Subtractions for k=i:
        sub_i = (a[i, i] * (b[v, u] - b[u, u]) + 
                 a[j, i] * (b[u, u] - b[v, u]) + 
                 a[i, i] * (b[u, v] - b[u, u]) + 
                 a[i, j] * (b[u, u] - b[u, v]))
                 
        # Subtractions for k=j:
        sub_j = (a[i, j] * (b[v, v] - b[u, v]) + 
                 a[j, j] * (b[u, v] - b[v, v]) + 
                 a[j, i] * (b[v, v] - b[v, u]) + 
                 a[j, j] * (b[v, u] - b[v, v]))
        
        # Direct terms (i,j and j,i interactions)
        # From earlier derivation:
        # a_ii(2b_vv - 2b_uu) + a_jj(2b_uu - 2b_vv) + 2(a_ij - a_ji)(b_vu - b_uv)
        # Wait, let's re-verify the direct terms.
        # Original cost terms involving i,j:
        # a_ii b_uu + a_ij b_uv + a_ji b_vu + a_jj b_vv
        # New cost terms involving i,j (p_i=v, p_j=u):
        # a_ii b_vv + a_ij b_vu + a_ji b_uv + a_jj b_uu
        
        # Direct Delta = (a_ii b_vv + a_ij b_vu + a_ji b_uv + a_jj b_uu) - (a_ii b_uu + a_ij b_uv + a_ji b_vu + a_jj b_vv)
        # = a_ii(b_vv - b_uu) + a_jj(b_uu - b_vv) + a_ij(b_vu - b_uv) + a_ji(b_uv - b_vu)
        # = (a_ii - a_jj)(b_vv - b_uu) + (a_ij - a_ji)(b_vu - b_uv)
        
        direct = (a[i, i] - a[j, j]) * (b[v, v] - b[u, u]) + (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
        
        return int(total - sub_i - sub_j + direct)
    