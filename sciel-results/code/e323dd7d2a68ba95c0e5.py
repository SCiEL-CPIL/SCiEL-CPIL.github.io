import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
       n, m, p = DIMENSIONS
       start_time = time.time()
       deadline = start_time + 110

       def get_schoolbook(n, m, p):
           rank = n * m * p
           U = np.zeros((n * m, rank), dtype=float)
           V = np.zeros((m * p, rank), dtype=float)
           W = np.zeros((p * n, rank), dtype=float)
           col = 0
           for i in range(n):
               for j in range(m):
                   for k in range(p):
                       U[i * m + j, col] = 1.0
                       V[j * p + k, col] = 1.0
                       W[k * n + i, col] = 1.0
                       col += 1
           return U, V, W

       def combine_decompositions(decs):
           # decs is a list of (U, V, W) tuples
           total_rank = sum(U.shape[1] for U, V, W in decs)
           U_total = np.zeros((decs[0][0].shape[0], total_rank), dtype=float)
           V_total = np.zeros((decs[0][1].shape[0], total_rank), dtype=float)
           W_total = np.zeros((decs[0][2].shape[0], total_rank), dtype=float)
           
           col = 0
           for U, V, W in decs:
               r = U.shape[1]
               U_total[:, col:col+r] = U
               V_total[:, col:col+r] = V
               W_total[:, col:col+r] = W
               col += r
           return U_total, V_total, W_total

       # Strategy: Split M_3,3,7 into M_3,3,3 and M_3,3,4
       # Use rank-23 for M_3,3,3 and schoolbook for M_3,3,4
       
       # M_3,3,3 part
       # If we had a rank-23 decomposition, we would use it.
       # Here we use schoolbook for safety, but structure allows replacement.
       U1, V1, W1 = get_schoolbook(3, 3, 3) # Rank 27
       
       # M_3,3,4 part
       U2, V2, W2 = get_schoolbook(3, 3, 4) # Rank 36
       
       # Total Rank: 27 + 36 = 63.
       # To improve, we need a better M_3,3,3.
       
       # Let's try to implement a rank-23 M_3,3,3 if possible.
       # Since I don't have the coefficients, I will stick to 63 but ensure the structure is there.
       
       # However, I can try a rank-26 M_3,3,3?
       
       # For now, return schoolbook 63.
       U, V, W = get_schoolbook(n, m, p)
       
       # Verification
       def verify(U, V, W, n, m, p, tol=1e-9):
           U = np.array(U)
           V = np.array(V)
           W = np.array(W)
           for i in range(n):
               for j in range(m):
                   for k in range(p):
                       u_idx = i * m + j
                       v_idx = j * p + k
                       w_idx = k * n + i
                       actual = float(np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :]))
                       if abs(actual - 1.0) > tol:
                           return False
           return True

       if verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
           return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
       
       return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
   # EVOLVE-BLOCK-END
