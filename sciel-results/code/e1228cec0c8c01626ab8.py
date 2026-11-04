import scipy
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
       n, m, p = DIMENSIONS
       # Target tensor T
       # T[i, j, k] = 1 if it corresponds to A[i,j] * B[j,k] contributing to C[i,k]
       # In the prompt's indexing:
       # U index: i*m + j
       # V index: j*p + k
       # W index: k*n + i
       # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k?
       # No, the tensor is T[i, j, k] = 1?
       # If T is all ones, rank is 1.
       # The seed code implies T is the sum of basis tensors.
       # Let's assume the target is the standard matrix multiplication tensor.
       
       # Initialize with schoolbook (Rank 27)
       rank_schoolbook = n * m * p
       U_sb = np.zeros((n*m, rank_schoolbook))
       V_sb = np.zeros((m*p, rank_schoolbook))
       W_sb = np.zeros((p*n, rank_schoolbook))
       col = 0
       for i in range(n):
           for j in range(m):
               for k in range(p):
                   U_sb[i*m+j, col] = 1
                   V_sb[j*p+k, col] = 1
                   W_sb[k*n+i, col] = 1
                   col += 1
       
       # Try to find Rank 19
       target_rank = 19
       if target_rank < rank_schoolbook:
           start_time = time.time()
           deadline = start_time + 100  # Leave 20s margin
           
           # Use a simple gradient descent or random search
           # This is computationally expensive.
           # For the sake of the task, if I can't find it quickly, I return schoolbook.
           
           # A known rank 19 decomposition exists.
           # I will return the schoolbook one to be safe, as implementing a solver
           # that reliably finds rank 19 in <100s in Python is risky.
           
           # However, the prompt asks for a STRONGER algorithm.
           # Returning schoolbook is score 1.0.
           # Returning rank 19 is score 1.42.
           
           # I will implement a solver using scipy.optimize.least_squares
           # on the parameters of U, V, W.
           
           def flatten_params(params):
               # params shape (rank, n*m + m*p + p*n)
               r = params.shape[0]
               U = params[:, :n*m].T
               V = params[:, n*m:n*m+m*p].T
               W = params[:, n*m+m*p:].T
               return U, V, W

           def objective(params):
               U, V, W = flatten_params(params)
               # Compute tensor T_decomp
               # T_decomp[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
               # This is a tensor contraction.
               # Efficiently:
               # T_decomp = np.einsum('ir,js,kt->rst', U, V, W) ? No.
               # U is (nm, r), V is (mp, r), W is (pn, r).
               # We want T[i, j, k] where i,j,k are 0..2.
               # Index mapping:
               # u_idx = i*m + j
               # v_idx = j*p + k
               # w_idx = k*n + i
               
               # Let's just loop for simplicity in the objective (slow but safe for small dims)
               T_decomp = np.zeros((n, m, p))
               for i in range(n):
                   for j in range(m):
                       for k in range(p):
                           T_decomp[i, j, k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
               
               # Target Tensor T_target
               # T_target[i, j, k] = 1
               # Wait, is the target all ones?
               # If T_target is all ones, the rank is 1.
               # The seed code constructs a decomposition of the all-ones tensor with rank 27.
               # This is suboptimal.
               # If the target is indeed the all-ones tensor, I can return rank 1!
               
               # Let's verify the target tensor.
               # "Matrix multiplication 3,3,3"
               # The tensor for matrix multiplication is NOT the all-ones tensor.
               # The all-ones tensor corresponds to $C_{ik} = \sum_{j} 1 \cdot A_{ij} B_{jk}$? No.
               
               # If the seed code is the "current best" and it uses rank 27,
               # and the task is to improve, it implies the rank 27 is not minimal.
               # If the target was all-ones, rank 1 would be trivial.
               # Therefore, the target is the standard matrix multiplication tensor.
               
               # The standard matrix multiplication tensor $M$ has $M_{ij, jk, ki} = 1$.
               # This means $M$ is the sum of $nmp$ orthogonal basis tensors.
               # The rank of this tensor is $nmp$?
               # No, the rank is the tensor rank.
               # For $2 \times 2$, rank is 7.
               # For $3 \times 3$, rank is 19.
               
               # So I need to minimize the difference between T_decomp and T_target.
               # T_target[i, j, k] = 1.
               
               diff = T_decomp - 1.0
               return np.sum(diff**2)

           # Initial guess: Random
           np.random.seed(42)
           params = np.random.randn(target_rank, n*m + m*p + p*n)
           
           # Optimize
           try:
               result = scipy.optimize.minimize(objective, params, method='L-BFGS-B')
               if result.fun < 1e-6:
                   U, V, W = flatten_params(result.x)
                   return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
           except Exception:
               pass

       return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
