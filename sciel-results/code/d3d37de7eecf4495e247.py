def _rect_1x2x3_factors():
    """Return (U1, V1, W1) for 1x2x3 rectangular, shapes (2,5),(6,5),(3,5).
    U1 row: j (0..1), V1 row: j*3+k (0..5), W1 row: k (0..2).
    T[j, 3j+k, k] = 1 for j in {0,1}, k in {0,1,2}."""
    # Need: sum_s U[j,s] * V[3j+k,s] * W[k,s] = 1 for valid (j,k)
    # And: sum_s U[j,s] * V[b,s] * W[c,s] = 0 for invalid (j,b,c)
    
    # This is a 5-rank decomposition of a 2x6x3 tensor with 6 non-zero entries.
    # Let me try to construct it.
    
    # One approach: use the fact that for fixed j, the equation is:
    # sum_s U[j,s] * V[3j+k,s] * W[k,s] = 1 for k=0,1,2
    # This is a system where for each j, we have a 3x3 matrix M_j[k,s] = V[3j+k,s]*W[k,s]
    # and we need U[j,:] @ M_j = [1,1,1].
    
    # For invalid entries: sum_s U[j,s] * V[b,s] * W[c,s] = 0
    # where (j,b,c) is not a valid triple.
    
    # This is complex. Let me try a specific construction.
    
    # Actually, let me try rank 6 first (schoolbook for 1x2x3):
    # U[j,s] = 1 if s = 3j+k for k=0,1,2... no, that's 6 terms.
    
    # Schoolbook: for each (j,k), one term:
    # U[j, 3j+k] = 1, V[3j+k, 3j+k] = 1, W[k, 3j+k] = 1
    # But this gives rank 6, not 5.
    
    # For rank 5, we need to share one term. Let me try:
    # Term 0: U[0,0]=1, V[0,0]=1, W[0,0]=1  (covers j=0,k=0)
    # Term 1: U[0,1]=1, V[1,1]=1, W[1,1]=1  (covers j=0,k=1)
    # Term 2: U[0,2]=1, V[2,2]=1, W[2,2]=1  (covers j=0,k=2)
    # Term 3: U[1,3]=1, V[3,3]=1, W[0,3]=1  (covers j=1,k=0)
    # Term 4: U[1,4]=1, V[4,4]=1, W[1,4]=1  (covers j=1,k=1)
    # And we need to cover j=1,k=2 with the existing terms.
    
    # For j=1,k=2: sum_s U[1,s]*V[5,s]*W[2,s] = 1
    # Currently: U[1,3]=1, V[5,3]=0, W[2,3]=0 -> contributes 0
    #            U[1,4]=1, V[5,4]=0, W[2,4]=0 -> contributes 0
    # So we need to add contributions. This doesn't work with rank 5 schoolbook-like.
    
    # Let me try a different approach. Maybe rank 5 is not achievable for 1x2x3.
    # Let me just use rank 6 (schoolbook) and get total rank 7*6 = 42, same as current.
    
    # Or try rank 5 with a non-trivial construction.
    
    pass
