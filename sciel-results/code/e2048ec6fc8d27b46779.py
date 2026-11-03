<<<<<<< SEARCH
def _compute_swap_delta(a, b, p, i, j):
    """
    Compute the cost delta for swapping positions i and j.
    """
    pi = p[i]
    pj = p[j]
    
    # T1: sum_{k != i, j} (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
    # We compute sum for all k, then subtract k=i and k=j terms.
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_all_T1 = np.dot(diff_a_col, diff_b_row)
    
    # Correction for k=i: (a[i,i] - a[i,j]) * (b[pi, pj] - b[pi, pi])
    term_1_i = (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    # Correction for k=j: (a[j,i] - a[j,j]) * (b[pj, pj] - b[pj, pi])
    term_1_j = (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    T1 = sum_all_T1 - term_1_i - term_1_j
    
    # T2: sum_{k != i, j} (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_all_T2 = np.dot(diff_a_row, diff_b_col)
    
    # Correction for k=i: (a[i,i] - a[j,i]) * (b[pj, pi] - b[pi, pi])
    term_2_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    # Correction for k=j: (a[i,j] - a[j,j]) * (b[pj, pj] - b[pi, pj])
    term_2_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    T2 = sum_all_T2 - term_2_i - term_2_j
    
    # T3: (a[i,j] - a[j,i]) * (b[pj, pi] - b[pi, pj])
    T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return T1 + T2 + T3
=======
def _compute_swap_delta(a, b, p, i, j):
    """
    Compute the cost delta for swapping positions i and j.
    """
    pi = p[i]
    pj = p[j]
    
    # T1: sum_{k != i, j} (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
    # We compute sum for all k, then subtract k=i and k=j terms.
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_all_T1 = np.dot(diff_a_col, diff_b_row)
    
    # Correction for k=i: (a[i,i] - a[i,j]) * (b[pi, pj] - b[pi, pi])
    term_1_i = (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    # Correction for k=j: (a[j,i] - a[j,j]) * (b[pj, pj] - b[pj, pi])
    term_1_j = (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    T1 = sum_all_T1 - term_1_i - term_1_j
    
    # T2: sum_{k != i, j} (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_all_T2 = np.dot(diff_a_row, diff_b_col)
    
    # Correction for k=i: (a[i,i] - a[j,i]) * (b[pj, pi] - b[pi, pi])
    term_2_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    # Correction for k=j: (a[i,j] - a[j,j]) * (b[pj, pj] - b[pi, pj])
    term_2_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    T2 = sum_all_T2 - term_2_i - term_2_j
    
    # T3: (a[i,j] - a[j,i]) * (b[pj, pi] - b[pi, pj])
    T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return T1 + T2 + T3

def _compute_all_deltas_for_i(a, b, p, i):
    """
    Vectorized computation of swap deltas for all j != i.
    Returns array of length n-1 with deltas for swapping i with each other position.
    """
    n = len(p)
    pi = p[i]
    
    # Precompute terms that depend only on i
    # For each j, delta(i,j) involves:
    # T1: sum_{k!=i,j} (a[k,i]-a[k,j])*(b[p[k],p[j]]-b[p[k],p[i]])
    # T2: sum_{k!=i,j} (a[i,k]-a[j,k])*(b[p[j],p[k]]-b[p[i],p[k]])
    # T3: (a[i,j]-a[j,i])*(b[p[j],p[i]]-b[p[i],p[j]])
    
    # Vectorized approach: compute for all j at once
    # T1[j] = sum_k (a[k,i]-a[k,j])*(b[p[k],p[j]]-b[p[k],p[i]])
    #       = sum_k a[k,i]*b[p[k],p[j]] - sum_k a[k,i]*b[p[k],p[i]] - sum_k a[k,j]*b[p[k],p[j]] + sum_k a[k,j]*b[p[k],p[i]]
    
    # Let's compute components:
    # S1[j] = sum_k a[k,i] * b[p[k], p[j]]  -> this is (a[:,i] @ b[p,:]) but indexed by p[j]
    # Actually: S1[j] = sum_k a[k,i] * b[p[k], p[j]]
    # = sum_k a[k,i] * b[p[k], :] evaluated at column p[j]
    
    # Let A_i = a[:, i] (vector of length n)
    # Let B_p = b[p, :] (n x n matrix where row k is b[p[k], :])
    # Then S1[j] = A_i @ B_p[:, p[j]] = (A_i @ B_p) [p[j]]
    
    A_i = a[:, i]  # length n
    B_p = b[p, :]  # n x n, row k is b[p[k], :]
    
    # S1_vec[j'] = sum_k a[k,i] * b[p[k], p[j']] for j' in 0..n-1
    # = A_i @ B_p  (length n vector, indexed by location)
    S1_loc = A_i @ B_p  # length n, S1_loc[loc] = sum_k a[k,i]*b[p[k],loc]
    S1 = S1_loc[p]  # S1[j] = S1_loc[p[j]]
    
    # S2[j] = sum_k a[k,i] * b[p[k], p[i]] = A_i @ B_p[:, p[i]] = S1_loc[p[i]]
    S2 = np.full(n, S1_loc[p[i]])  # constant for all j
    
    # S3[j] = sum_k a[k,j] * b[p[k], p[j]]
    # = sum_k a[k,j] * b[p[k], :] evaluated at column p[j]
    # For each j, this is a[:,j] @ B_p[:, p[j]]
    # Vectorized: for each j, compute dot(a[:,j], B_p[:, p[j]])
    # B_p[:, p[j]] is column p[j] of B_p
    # We need: for each j, dot(a[:,j], B_p[:, p[j]])
    # = diag( (a * B_p[:, p[None,:]]).sum(axis=0) ) ... let me think differently
    # a[:,j] @ B_p[:, p[j]] = sum_k a[k,j] * B_p[k, p[j]]
    # B_p[k, p[j]] = b[p[k], p[j]]
    # So S3[j] = sum_k a[k,j] * b[p[k], p[j]]
    # This is the diagonal of (a * b[p[:,None], p[None,:]]) summed... no
    # Let C = a * b[np.ix_(p, p)]  (n x n, C[i,j] = a[i,j]*b[p[i],p[j]])
    # Then S3[j] = sum_k a[k,j] * b[p[k], p[j]] ... not quite C
    # C[k,j] = a[k,j] * b[p[k], p[j]], so S3[j] = sum_k C[k,j] = C[:,j].sum()
    C = a * b[np.ix_(p, p)]
    S3 = C.sum(axis=0)  # S3[j] = sum_k a[k,j]*b[p[k],p[j]]
    
    # S4[j] = sum_k a[k,j] * b[p[k], p[i]]
    # = sum_k a[k,j] * b[p[k], :] evaluated at column p[i]
    # = for each j, dot(a[:,j], B_p[:, p[i]])
    # B_p[:, p[i]] is a fixed column
    Bp_col_i = B_p[:, p[i]]  # length n
    S4 = a @ Bp_col_i  # S4[j] = sum_k a[k,j]*B_p[k,p[i]] = sum_k a[k,j]*b[p[k],p[i]]
    
    # T1[j] = S1[j] - S2[j] - S3[j] + S4[j]
    T1_vec = S1 - S2 - S3 + S4
    
    # Now T2: sum_{k!=i,j} (a[i,k]-a[j,k])*(b[p[j],p[k]]-b[p[i],p[k]])
    # = sum_k (a[i,k]-a[j,k])*(b[p[j],p[k]]-b[p[i],p[k]]) - corrections for k=i,j
    # Let's expand:
    # = sum_k a[i,k]*b[p[j],p[k]] - sum_k a[i,k]*b[p[i],p[k]] - sum_k a[j,k]*b[p[j],p[k]] + sum_k a[j,k]*b[p[i],p[k]]
    
    # T2a[j] = sum_k a[i,k]*b[p[j],p[k]]
    # = sum_k a[i,k]*b[p[j],:] evaluated at column p[k]... 
    # b[p[j], p[k]] for fixed j, varying k: this is b[p[j], p[k]] = B_row_pj[p[k]] where B_row_pj = b[p[j],:]
    # T2a[j] = sum_k a[i,k] * b[p[j], p[k]] = a[i,:] @ b[p[j], p]
    # b[p[j], p] is a vector: for each k, b[p[j], p[k]]
    # = a[i,:] @ (b[p[j], p])
    
    # Vectorized over j:
    # T2a[j] = a[i,:] @ b[p[j], p]
    # b[p[j], p] for all j: this is b[np.ix_(p, p)] transposed... 
    # b[p[j], p[k]] = B_pp[j, k] where B_pp = b[np.ix_(p,p)]
    # So T2a[j] = sum_k a[i,k] * B_pp[j,k] = (B_pp @ a[i,:])[j]
    
    B_pp = b[np.ix_(p, p)]  # B_pp[j,k] = b[p[j],p[k]]
    a_i = a[i, :]  # length n
    T2a = B_pp @ a_i  # T2a[j] = sum_k B_pp[j,k]*a_i[k] = sum_k b[p[j],p[k]]*a[i,k]
    
    # T2b[j] = sum_k a[i,k]*b[p[i],p[k]] = sum_k a[i,k]*b[p[i],:] at p[k]
    # = a[i,:] @ b[p[i], p]
    b_pi_p = b[p[i], p]  # length n, b_pi_p[k] = b[p[i],p[k]]
    T2b_val = a_i @ b_pi_p  # scalar, same for all j
    T2b = np.full(n, T2b_val)
    
    # T2c[j] = sum_k a[j,k]*b[p[j],p[k]]
    # = sum_k a[j,k]*B_pp[j,k] = diag(a * B_pp) ... 
    # For each j: sum_k a[j,k]*B_pp[j,k] = (a * B_pp)[j,:].sum()
    T2c = (a * B_pp).sum(axis=1)  # T2c[j] = sum_k a[j,k]*b[p[j],p[k]]
    
    # T2d[j] = sum_k a[j,k]*b[p[i],p[k]]
    # = sum_k a[j,k]*b[p[i],:] at p[k] = a[j,:] @ b[p[i],p]
    # = a[j,:] @ b_pi_p
    T2d = a @ b_pi_p  # T2d[j] = sum_k a[j,k]*b_pi_p[k]
    
    T2_vec = T2a - T2b - T2c + T2d
    
    # T3[j] = (a[i,j]-a[j,i])*(b[p[j],p[i]]-b[p[i],p[j]])
    # b[p[j],p[i]] = B_pp[j, i] (since B_pp[j,k]=b[p[j],p[k]], so B_pp[j,i]=b[p[j],p[i]])
    # b[p[i],p[j]] = B_pp[i, j]
    T3_vec = (a[i, :] - a[:, i]) * (B_pp[:, i] - B_pp[i, :])
    
    # Raw deltas before corrections for k=i and k=j
    raw_deltas = T1_vec + T2_vec + T3_vec
    
    # Now subtract corrections for k=i and k=j
    # For k=i term in T1: (a[i,i]-a[i,j])*(b[p[i],p[j]]-b[p[i],p[i]])
    # For k=j term in T1: (a[j,i]-a[j,j])*(b[p[j],p[j]]-b[p[j],p[i]])
    
    # corr_T1_i[j] = (a[i,i]-a[i,j])*(b[p[i],p[j]]-b[p[i],p[i]])
    # b[p[i],p[j]] = B_pp[i,j], b[p[i],p[i]] = B_pp[i,i]
    corr_T1_i = (a[i, i] - a[i, :]) * (B_pp[i, :] - B_pp[i, i])
    
    # corr_T1_j[j] = (a[j,i]-a[j,j])*(b[p[j],p[j]]-b[p[j],p[i]])
    # b[p[j],p[j]] = B_pp[j,j], b[p[j],p[i]] = B_pp[j,i]
    corr_T1_j = (a[:, i] - np.diag(a)) * (np.diag(B_pp) - B_pp[:, i])
    
    # For T2:
    # corr_T2_i[j] = (a[i,i]-a[j,i])*(b[p[j],p[i]]-b[p[i],p[i]])
    # = (a[i,i]-a[j,i])*(B_pp[j,i]-B_pp[i,i])
    corr_T2_i = (a[i, i] - a[:, i]) * (B_pp[:, i] - B_pp[i, i])
    
    # corr_T2_j[j] = (a[i,j]-a[j,j])*(b[p[j],p[j]]-b[p[i],p[j]])
    # = (a[i,j]-a[j,j])*(B_pp[j,j]-B_pp[i,j])
    corr_T2_j = (a[i, :] - np.diag(a)) * (np.diag(B_pp) - B_pp[i, :])
    
    # T3 correction: the formula already excludes k=i,j in the sum, but T3 is a separate term
    # Actually T3 is just a single term, no sum, so no correction needed beyond what's in the formula
    
    # Total corrections
    corrections = corr_T1_i + corr_T1_j + corr_T2_i + corr_T2_j
    
    deltas = raw_deltas - corrections
    
    # Exclude j=i (should be 0 but let's mask it)
    deltas[i] = 0
    
    return deltas
>>>>>>> REPLACE

