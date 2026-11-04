U_t = U[:, t].reshape(4, 4)
V_t = V[:, t].reshape(4, 5)
W_t = W[:, t].reshape(5, 4)  # W[k*n+i] -> W_t[k, i]
left = np.sum(A_test * U_t)
right = np.sum(B_test * V_t)
C_recon += left * right * W_t.T
