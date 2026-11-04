import numpy as np
N = 50
w = 0.5 / N  # 0.01
b = -0.25 + np.arange(N+1) * w  # boundaries

def autoconv_peak(h):
    h = np.asarray(h, dtype=float)
    best = 0.0
    for m in range(-N, N+1):
        # interval for i: [b_i, b_{i+1}]
        # interval for j shifted by m*w: [m*w - b_{j+1}, m*w - b_j]
        A = b[:-1]            # b_i, shape (N,)
        B = b[1:]             # b_{i+1}, shape (N,)
        C = m*w - b[1:]       # m*w - b_{j+1}, shape (N,)
        D = m*w - b[:-1]      # m*w - b_j, shape (N,)
        # overlap of [A_i, B_i] and [C_j, D_j] for all i,j
        lo = np.maximum(A[:,None], C[None,:])
        hi = np.minimum(B[:,None], D[None,:])
        ov = np.maximum(0.0, hi - lo)  # (N,N)
        val = ov @ (h[None,:] * h[:,None])  # sum_{i,j} ov_{i,j} h_i h_j
        # actually val = sum_{i,j} ov_{i,j} * h_i * h_j
        val = (ov * (h[:,None] * h[None,:])).sum()
        if val > best:
            best = val
    return best
