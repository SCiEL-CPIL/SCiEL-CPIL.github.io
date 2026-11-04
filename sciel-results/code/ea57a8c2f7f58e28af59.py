<<<<<<< SEARCH
    # Schoolbook for j = 2
    for i in range(n):
        for k in range(p):
            U[u_idx(i, 2)][idx] += 1
            V[v_idx(2, k)][idx] += 1
            W[w_idx(k, i)][idx] += 1
            idx += 1

    return {'U': U, 'V': V, 'W': W}
=======
    # Rank-1 decomposition for j = 2 (outer product)
    # U[i, 2] = 1 for all i
    # V[2, k] = 1 for all k
    # W[k, i] = 1 for all k, i
    U[u_idx(0, 2)][idx] += 1
    U[u_idx(1, 2)][idx] += 1
    
    for k in range(p):
        V[v_idx(2, k)][idx] += 1
        for i in range(n):
            W[w_idx(k, i)][idx] += 1
            
    idx += 1

    return {'U': U, 'V': V, 'W': W}
>>>>>>> REPLACE
