import numpy as np

N = 130
M = 20000 # Test points

def construct():
       # 1. Generate test points on sphere (Fibonacci spiral)
       i = np.arange(M) + 0.5
       z = 1.0 - 2.0 * i / M
       phi = np.pi * (1.0 + np.sqrt(5.0)) * i
       r = np.sqrt(1.0 - z * z)
       T = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
       
       # 2. Initialize P (Fibonacci spiral)
       i_p = np.arange(N) + 0.5
       z_p = 1.0 - 2.0 * i_p / N
       phi_p = np.pi * (1.0 + np.sqrt(5.0)) * i_p
       r_p = np.sqrt(1.0 - z_p * z_p)
       P = np.column_stack([r_p * np.cos(phi_p), r_p * np.sin(phi_p), z_p])
       
       # 3. CVT Iterations
       for _ in range(100):
           # Compute distances from all test points to all config points
           # T shape (M, 3), P shape (N, 3)
           # dist_sq shape (M, N)
           dist_sq = np.sum(T[:, np.newaxis, :] ** 2) + np.sum(P[np.newaxis, :, :] ** 2) - 2 * T @ P.T
           # Clip to avoid negative due to floating point
           dist_sq = np.maximum(dist_sq, 0)
           
           # Assign each test point to nearest config point
           assignments = np.argmin(dist_sq, axis=1)
           
           # Compute new positions (centroids)
           new_P = np.zeros_like(P)
           counts = np.zeros(N)
           
           # Efficient accumulation
           # We can use np.add.at or bincount-like logic
           # Since M is large, looping over M is slow.
           # Vectorized approach:
           # Create indices 0..M-1
           idx = np.arange(M)
           # For each config point i, we need sum of T[j] where assignments[j] == i
           # We can use np.bincount with weights? No, weights are vectors.
           # We can use a loop over N (130 is small)
           for i in range(N):
               mask = (assignments == i)
               if np.any(mask):
                   new_P[i] = np.sum(T[mask], axis=0)
                   counts[i] = np.sum(mask)
               else:
                   new_P[i] = P[i] # Fallback
                   counts[i] = 1.0
                   
           # Normalize
           norms = np.linalg.norm(new_P, axis=1)
           # Avoid division by zero
           norms[norms == 0] = 1.0
           new_P = new_P / norms[:, np.newaxis]
           
           # Check convergence
           if np.max(np.linalg.norm(new_P - P, axis=1)) < 1e-6:
               P = new_P
               break
               
           P = new_P
           
       return P
