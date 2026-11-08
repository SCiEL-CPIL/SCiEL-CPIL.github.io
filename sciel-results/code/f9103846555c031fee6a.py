# V is (nB, rank), W is (nC, rank)
# We need M_VW[b*nC+c, k] = V[b,k] * W[c,k]
V_rep = np.repeat(V, nC, axis=0) # (nB*nC, rank)
W_rep = np.tile(W, (nB, 1))      # (nB*nC, rank)
M_VW = V_rep * W_rep             # (nB*nC, rank)
