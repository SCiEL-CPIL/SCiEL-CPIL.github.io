import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
       start_time = time.time()
       deadline = start_time + 110.0 # 10s margin
       
       # Candidate 1: Constant
       h_const = np.full(N, 0.5)
       
       # Candidate 2: Linear Ramp
       h_ramp = np.linspace(0, 1, N)
       # Adjust sum to N/2
       # Sum is (N-1)/2. Need N/2.
       # Add 0.5/N to each?
       # h_ramp += 0.5/N
       # This makes h_0 = 0.005, h_{N-1} = 1.005 (clip to 1)
       # Better: Scale? No, shape matters.
       # Just use optimizer.
       
       def calc_overlaps(h):
           h = np.clip(h, 0, 1)
           s = np.zeros(N-1)
           for k in range(1, N):
               s[k-1] = np.sum(h[:N-k] * (1 - h[k:]))
           return s
           
       def score(h):
           return np.max(calc_overlaps(h))
           
       best_h = h_ramp
       best_score = score(h_ramp)
       
       # Try to optimize
       def objective(h):
           if time.time() > deadline:
               return 1e9
           s = calc_overlaps(h)
           alpha = 100.0
           return np.log(np.sum(np.exp(alpha * s))) / alpha
           
       constraints = [
           {'type': 'eq', 'fun': lambda h: np.sum(h) - N/2}
       ]
       bounds = [(0, 1) for _ in range(N)]
       
       try:
           res = minimize(objective, best_h, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-8})
           if res.success and time.time() < deadline:
               cand = res.x
               cand_score = score(cand)
               if cand_score < best_score:
                   best_h = cand
                   best_score = cand_score
       except:
           pass
           
       return best_h
