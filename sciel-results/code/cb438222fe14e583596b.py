import numpy as np
from scipy.optimize import minimize
import time

def construct():
       start_time = time.time()
       deadline = start_time + 120 # 30s margin
       
       n = 8
       # Objective: minimize -log_score
       # log_score = sum(log(d_ij)) - 56 * log(max_d)
       
       def objective(x_flat):
           x = x_flat.reshape(n, 2)
           # Distances
           diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
           dists = np.sqrt(np.sum(diff**2, axis=2))
           # Mask diagonal
           np.fill_diagonal(dists, 1.0) # avoid log(0)
           # Actually, we need to exclude diagonal from sum
           mask = ~np.eye(n, dtype=bool)
           d_valid = dists[mask]
           
           if np.any(d_valid == 0):
               return 1e100
               
           max_d = np.max(d_valid)
           if max_d < 1e-9:
               return 1e100
               
           log_prod = np.sum(np.log(d_valid))
           log_obj = log_prod - 56 * np.log(max_d)
           return -log_obj
           
       best_score = 0
       best_x = None
       
       # Initial points: Regular Octagon
       angles = np.linspace(0, 2*np.pi, n, endpoint=False)
       x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
       
       # List of initial guesses
       inits = [x0]
       
       # Add random perturbations of x0
       rng = np.random.default_rng(42)
       for i in range(10):
           noise = rng.normal(0, 0.1, size=x0.shape)
           inits.append(x0 + noise)
           
       # Add random points
       for i in range(5):
           x_rand = rng.uniform(-1, 1, size=(n, 2)).flatten()
           inits.append(x_rand)
           
       # Optimization loop
       for x_init in inits:
           if time.time() > deadline:
               break
               
           try:
               res = minimize(objective, x_init, method='Powell', 
                              options={'maxiter': 1000, 'xtol': 1e-8, 'ftol': 1e-8})
               if res.success:
                   x_opt = res.x
                   score_val = -res.fun
                   # Convert log score to linear score
                   # score = exp(score_val) / 8^8 ? No.
                   # log_score = log(Delta) - 56 log(D).
                   # Score = Delta / D^56 / 8^8 ?
                   # Wait, the task score is Delta / n^n after scaling to D=2.
                   # If we scale to D=2, Delta scales by (2/D)^56.
                   # Delta_scaled = Delta_orig * (2/D)^56.
                   # Score = Delta_scaled / 8^8 = Delta_orig * 2^56 / D^56 / 8^8.
                   # log(Score) = log(Delta_orig) - 56 log(D) + 56 log(2) - 8 log(8).
                   # log(Score) = log_obj_original + 56 log(2) - 8 log(8).
                   # My objective was minimizing -(log(Delta) - 56 log(D)).
                   # So -res.fun = log(Delta) - 56 log(D).
                   # log(Score) = -res.fun + 56 log(2) - 8 log(8).
                   # 8 log(8) = 8 * 3 log(2) = 24 log(2).
                   # log(Score) = -res.fun + 56 log(2) - 24 log(2) = -res.fun + 32 log(2).
                   
                   log_score = -res.fun + 32 * np.log(2)
                   score = np.exp(log_score)
                   
                   if score > best_score:
                       best_score = score
                       best_x = x_opt
                       
           except Exception as e:
               pass
               
       if best_x is None:
           # Fallback to regular octagon
           return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
           
       # Scale best_x to diameter 2
       x_best = best_x.reshape(n, 2)
       diff = x_best[:, np.newaxis, :] - x_best[np.newaxis, :, :]
       dists = np.sqrt(np.sum(diff**2, axis=2))
       D = np.max(dists)
       x_scaled = x_best / (D/2.0)
       
       # Ensure distinctness and bounds
       # Check if points are distinct
       # If not, return regular octagon
       
       return x_scaled.tolist()
