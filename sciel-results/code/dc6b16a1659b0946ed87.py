import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
       n, m, p = DIMENSIONS
       # Schoolbook rank
       rank_school = n * m * p
       
       # Build target tensor
       # U shape (n*m, 1), V shape (m*p, 1), W shape (p*n, 1)
       # T[u, v, w] = 1 if u, v, w form a path
       u_size = n * m
       v_size = m * p
       w_size = p * n
       
       T = np.zeros((u_size, v_size, w_size))
       for i in range(n):
           for j in range(m):
               for k in range(p):
                   u = i * m + j
                   v = j * p + k
                   w = k * n + i
                   T[u, v, w] = 1.0

       # Try to find rank 47 decomposition
       target_rank = rank_school - 1
       
       start_time = time.time()
       time_limit = 140.0 # Safety margin for 150s limit
       
       best_result = None
       best_error = np.inf
       
       # We will try a few random restarts
       # Variables: U (u_size, r), V (v_size, r), W (w_size, r)
       r = target_rank
       
       # Gradient Descent
       # We can use a simple manual implementation or scipy.
       # scipy.optimize.minimize might be slow to start.
       # Let's use a simple gradient step with momentum.
       
       def compute_residual(U, V, W):
           # Sum of outer products
           # Reshape for efficient computation?
           # T is sparse, but sum is dense.
           # We can compute the tensor product sum.
           # Since r is small, we can loop.
           S = np.zeros_like(T)
           for c in range(r):
               S += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
           return T - S

       def compute_grad(U, V, W, Res):
           # d/dU Res^2 = 2 * Res * (V^T W^T)
           # d/dV Res^2 = 2 * Res * (U^T W^T)
           # d/dW Res^2 = 2 * Res * (U^T V^T)
           # We need to contract Res with V and W etc.
           # Res is (u, v, w)
           # dU[i, c] = sum_{v,w} Res[i, v, w] * V[v, c] * W[w, c]
           # This is einsum('ivw,vc,wc->ic', Res, V, W)
           dU = np.einsum('ivw,vc,wc->ic', Res, V, W)
           dV = np.einsum('ivw,uc,wc->vc', Res, U, W)
           dW = np.einsum('ivw,uc,vc->wc', Res, U, V)
           return 2 * dU, 2 * dV, 2 * dW

       # Random restarts
       num_restarts = 5
       steps_per_restart = 200
       
       for restart in range(num_restarts):
           if time.time() - start_time > time_limit - 10:
               break
               
           # Initialize
           # Scale 1/sqrt(rank)
           U = np.random.randn(u_size, r) / np.sqrt(r)
           V = np.random.randn(v_size, r) / np.sqrt(r)
           W = np.random.randn(w_size, r) / np.sqrt(r)
           
           lr = 0.01
           momentum = 0.9
           vU = np.zeros_like(U)
           vV = np.zeros_like(V)
           vW = np.zeros_like(W)
           
           for step in range(steps_per_restart):
               if time.time() - start_time > time_limit - 5:
                   break
                   
               Res = compute_residual(U, V, W)
               err = np.linalg.norm(Res)
               if err < 1e-6:
                   # Found a solution!
                   # Convert to list of lists
                   U_list = U.tolist()
                   V_list = V.tolist()
                   W_list = W.tolist()
                   return {"U": U_list, "V": V_list, "W": W_list}
               
               dU, dV, dW = compute_grad(U, V, W, Res)
               
               vU = momentum * vU + (1 - momentum) * dU
               vV = momentum * vV + (1 - momentum) * dV
               vW = momentum * vW + (1 - momentum) * dW
               
               U -= lr * vU
               V -= lr * vV
               W -= lr * vW
               
               # Adaptive learning rate?
               # If error increases, reduce lr
               # Simple: just keep going.
           
           # After restart, check final error
           Res = compute_residual(U, V, W)
           err = np.linalg.norm(Res)
           if err < best_error:
               best_error = err
               best_result = (U.tolist(), V.tolist(), W.tolist())

       if best_error < 1e-6:
           return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
       
       # Fallback to schoolbook
       U = [[0] * rank_school for _ in range(u_size)]
       V = [[0] * rank_school for _ in range(v_size)]
       W = [[0] * rank_school for _ in range(w_size)]
       column = 0
       for i in range(n):
           for j in range(m):
               for k in range(p):
                   U[i * m + j][column] = 1
                   V[j * p + k][column] = 1
                   W[k * n + i][column] = 1
                   column += 1
       return {"U": U, "V": V, "W": W}
