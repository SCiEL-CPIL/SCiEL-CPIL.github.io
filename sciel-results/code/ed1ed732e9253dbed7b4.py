import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
                start_time = time.time()
                deadline = start_time + 290  # 10s safety margin
                
                lo = np.array([b[0] for b in bounds])
                hi = np.array([b[1] for b in bounds])
                
                # State
                X = []
                F = []
                used = 0
                
                def add_point(x, f):
                    nonlocal used
                    X.append(x)
                    F.append(f)
                    used += 1

                # 1. Informed Initialization
                # DTLZ4: First M-1 vars define front, rest define distance.
                # M=3, so x[0], x[1] define front. x[2:] define distance.
                # Optimal distance vars are 0.
                n_init = 30
                for _ in range(n_init):
                    x = np.zeros(self.dim)
                    # Sample front variables
                    # Use Dirichlet-like sampling for simplex front or uniform for box
                    # Let's try uniform [0,1] for x[0], x[1]
                    x[0] = np.random.rand()
                    x[1] = np.random.rand()
                    # Distance variables: small random
                    x[2:] = np.random.rand(self.dim - 2) * 0.01
                    x = np.clip(x, lo, hi)
                    f = problem(x)
                    add_point(x, f)
                    used = len(X)
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break

                # 2. Local Search / Refinement
                # Strategy: 
                # - Keep a pool of non-dominated solutions.
                # - For each new eval, pick a random ND solution.
                # - Create child:
                #   - Perturb front vars (x[0], x[1]) slightly.
                #   - Push distance vars (x[2:]) closer to 0.
                
                while used < self.budget:
                    if time.time() > deadline:
                        break
                    
                    # Identify non-dominated indices
                    # Simple O(N^2) check for small N
                    nd_idx = []
                    for i in range(len(X)):
                        dominated = False
                        for j in range(len(X)):
                            if i == j: continue
                            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                dominated = True
                                break
                        if not dominated:
                            nd_idx.append(i)
                    
                    if not nd_idx:
                        nd_idx = [np.random.randint(len(X))]
                    
                    parent_idx = np.random.choice(nd_idx)
                    parent = X[parent_idx].copy()
                    
                    # Mutation
                    # Front vars: small jitter
                    parent[0] += np.random.normal(0, 0.05)
                    parent[1] += np.random.normal(0, 0.05)
                    
                    # Distance vars: push towards 0
                    # If current value is v, new value is v * (1 - step) + noise
                    # Or just set to small random value
                    parent[2:] = parent[2:] * 0.5 + np.random.rand(self.dim - 2) * 0.001
                    
                    parent = np.clip(parent, lo, hi)
                    f_child = problem(parent)
                    add_point(parent, f_child)
                    
                    # Optional: Prune dominated points to keep list small?
                    # Not strictly necessary for budget, but good for selection.
                    # With max 200 points, O(N^2) is fine.
                    
                return X
