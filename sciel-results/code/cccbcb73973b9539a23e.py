V_slice = V[j*p : (j+1)*p, :]
W_slice = W.reshape(p, n, R)[:, i, :]
M = V_slice * W_slice
