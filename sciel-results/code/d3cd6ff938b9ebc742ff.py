def _swap_delta(a, b, p, i, j, R, C):
    """Compute cost delta for swapping p[i] and p[j]. O(n)."""
    u, v = p[i], p[j]
    # Delta from row i and row j perspective
    # Cost = sum_{i,j} a[i,j]*b[p[i],p[j]]
    # Terms involving i or j change
    # delta = sum_k a[i,k]*(b[v,p'[k]]-b[u,p[k]]) + sum_k a[k,i]*(b[p'[k],v]-b[p[k],u])
    #         + sum_k a[j,k]*(b[u,p'[k]]-b[v,p[k]]) + sum_k a[k,j]*(b[p'[k],u]-b[p[k],v])
    # where p' has p'[i]=v, p'[j]=u, else same
    # For k=i: a[i,i]*(b[v,v]-b[u,u]) + a[i,i]*(b[v,v]-b[u,u]) ... 
    # This is O(n) if we precompute properly
    
    # Simpler: 
    # delta = (R[i,v] - R[i,u]) + (R[j,u] - R[j,v]) 
    #         + sum_{k≠i,j} (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
    # But R[i,v] = sum_k a[i,k]*b[v,p[k]] which uses OLD p[k] for k≠i,j
    # and for k=j, b[v,p[j]]=b[v,u], for k=i, b[v,p[i]]=b[v,u]... 
    # Actually R is computed with current p, so R[i,v] = sum_k a[i,k]*b[v,p[k]]
    # When we swap, for k≠i,j, p[k] doesn't change.
    # The issue is that R[i,v] includes the term a[i,j]*b[v,p[j]]=a[i,j]*b[v,u] which is correct for new state
    # and a[i,i]*b[v,p[i]]=a[i,i]*b[v,u] which should be a[i,i]*b[v,v] in new state
    # So R[i,v] is NOT exactly what we need.
    
    # Let me just compute it directly:
    # delta = 0
    # for k in range(n):
    #     if k == i: pk_new = v
    #     elif k == j: pk_new = u  
    #     else: pk_new = p[k]
    #     delta += a[i,k]*(b[v,pk_new] - b[u,p[k]])
    #     delta += a[k,i]*(b[pk_new,v] - b[p[k],u])
    #     delta += a[j,k]*(b[u,pk_new] - b[v,p[k]])
    #     delta += a[k,j]*(b[pk_new,u] - b[p[k],v])
    # But this counts some terms twice. Let me be more careful.
    
    # Cost = sum_{x,y} a[x,y]*b[p[x],p[y]]
    # After swap, p'[i]=v, p'[j]=u, p'[k]=p[k] for k≠i,j
    # delta = sum_{x,y} a[x,y]*(b[p'[x],p'[y]] - b[p[x],p[y]])
    # Only terms where x or y is i or j change
    # = sum_y a[i,y]*(b[v,p'[y]]-b[u,p[y]]) + sum_x a[x,i]*(b[p'[x],v]-b[p[x],u])
    #   + sum_y a[j,y]*(b[u,p'[y]]-b[v,p[y]]) + sum_x a[x,j]*(b[p'[x],u]-b[p[x],v])
    #   - a[i,i]*(b[v,v]-b[u,u]) - a[i,j]*(b[v,u]-b[u,v])  # subtract double-counted
    #   - a[j,i]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])  # etc.
    # This is getting too complex. Let me just do the O(n) computation directly.
    
    pk = p.copy()
    pk[i], pk[j] = v, u
    # delta = sum over all x,y where x or y in {i,j}
    # Vectorized:
    # row_i_delta = a[i,:] * (b[v, pk] - b[u, p])
    # col_i_delta = a[:,i] * (b[pk, v] - b[p, u])
    # row_j_delta = a[j,:] * (b[u, pk] - b[v, p])
    # col_j_delta = a[:,j] * (b[pk, u] - b[p, v])
    # But this overcounts the (i,i), (i,j), (j,i), (j,j) terms
    # Actually each term a[x,y]*b[p[x],p[y]] appears exactly once in the sum
    # row_i_delta covers x=i, all y
    # col_i_delta covers y=i, all x -> but (i,i) is counted twice
    # So: delta = row_i_delta + col_i_delta + row_j_delta + col_j_delta - a[i,i]*(b[v,v]-b[u,u]) - a[i,j]*(b[v,u]-b[u,v]) - a[j,i]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
    # Wait no. row_i_delta = sum_y a[i,y]*(b[v,pk[y]]-b[u,p[y]])
    # col_i_delta = sum_x a[x,i]*(b[pk[x],v]-b[p[x],u])
    # The (i,i) term appears in both: a[i,i]*(b[v,v]-b[u,u]) in row_i_delta AND a[i,i]*(b[v,v]-b[u,u]) in col_i_delta
    # So we need to subtract it once.
    # Similarly (i,j) appears in row_i_delta as a[i,j]*(b[v,u]-b[u,v]) and in col_j_delta as a[i,j]*(b[u,u]-b[v,u])... 
    # Hmm, col_j_delta = sum_x a[x,j]*(b[pk[x],u]-b[p[x],v])
    # For x=i: a[i,j]*(b[v,u]-b[u,v])
    # And row_i_delta for y=j: a[i,j]*(b[v,u]-b[u,v])
    # So (i,j) term is counted twice with the same value, should be counted once.
    
    # This is a mess. Let me just compute it as:
    # delta = sum_x sum_y a[x,y]*(b[p'[x],p'[y]]-b[p[x],p[y]])
    # = (sum over x,y where x=i or y=i) + (sum over x,y where x=j or y=j, excluding already counted)
    # 
    # Cleanest: just compute the full delta in O(n) with 4 vectorized ops:
    # d1 = a[i,:] @ (b[v, pk] - b[u, p])  # x=i contribution
    # d2 = a[:,i] @ (b[pk, v] - b[p, u])  # y=i contribution  
    # d3 = a[j,:] @ (b[u, pk] - b[v, p])  # x=j contribution
    # d4 = a[:,j] @ (b[pk, u] - b[p, v])  # y=j contribution
    # delta = d1 + d2 + d3 + d4 - (a[i,i]*(b[v,v]-b[u,u]) + a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u]) + a[j,j]*(b[u,u]-b[v,v]))
    # Wait, I think the overcounting is exactly these 4 corner terms, each counted twice.
    
    d1 = np.dot(a[i, :], b[v, pk] - b[u, p])
    d2 = np.dot(a[:, i], b[pk, v] - b[p, u])
    d3 = np.dot(a[j, :], b[u, pk] - b[v, p])
    d4 = np.dot(a[:, j], b[pk, u] - b[p, v])
    # Subtract the 4 corner terms that are double-counted
    corner = (a[i,i]*(b[v,v]-b[u,u]) + a[i,j]*(b[v,u]-b[u,v]) 
             + a[j,i]*(b[u,v]-b[v,u]) + a[j,j]*(b[u,u]-b[v,v]))
    return d1 + d2 + d3 + d4 - corner
