"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """PSO with constriction for f19 (Griewank-Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        n = self.dim
        budget = self.budget
        
        # PSO Parameters
        # Constriction coefficient formula: (2|eta|) / (|eta| - sqrt(|eta|^2 - 4|eta|))
        # For eta = c1 + c2 + w, with c1=c2=2.0, w=0.729 -> eta = 4.729
        # kappa = 2 / (4.729 - sqrt(4.729^2 - 4*4.729))
        # sqrt(22.36 - 18.91) = sqrt(3.45) = 1.85
        # kappa = 2 / (4.729 - 1.85) = 2 / 2.879 = 0.694
        # Let's use standard values: w=0.729, c1=c2=1.4944 (eta=3.729)
        # eta = 3.729
        # eta^2 - 4eta = 13.9 - 14.9 = -1 (complex, wait)
        # Standard constriction requires c1+c2 > 4 and |c1+c2 - 2 - sqrt((c1+c2-2)^2 - 4(c1+c2))| < 2
        # Let's stick to w=0.729, c1=2.05, c2=2.05 -> eta = 4.829
        # eta^2 - 4eta = 23.3 - 19.3 = 4.0
        # sqrt(4.0) = 2.0
        # kappa = 2 / (4.829 - 2.0) = 2 / 2.829 = 0.707
        # This is stable.
        
        w = 0.729
        c1 = 2.05
        c2 = 2.05
        kappa = 0.707 # Approximate constriction factor
        
        # Swarm size
        # With 500 budget, we want ~10-20 generations.
        # If swarm size is 25, we have 20 generations.
        # If swarm size is 50, we have 10 generations.
        # Larger swarm = better coverage, fewer iterations.
        # Smaller swarm = more iterations, potentially faster convergence per eval?
        # For AOCC, early progress is key. 
        # Let's try swarm size 30. 500/30 = 16 generations.
        m = 30
        
        # Initialize swarm
        # Box is [-5, 5]^5
        # Initialize positions uniformly or Gaussian? Uniform covers space better initially.
        # But f19 is smooth. 
        # Let's use uniform initialization in [-4, 4] to stay away from boundaries initially?
        # Or just [-5, 5].
        
        # We need to be careful with function calls.
        # Init: m calls.
        # Loop: m calls per generation.
        
        # Total calls: m + m * G
        # We have 500 calls.
        # G = (500 - m) // m
        
        # Initialize
        X = np.random.uniform(-5, 5, (m, n))
        V = np.zeros((m, n))
        
        F = np.zeros(m)
        calls = 0
        
        # Evaluate initial swarm
        for i in range(m):
            if calls >= budget: break
            F[i] = func(X[i])
            calls += 1
            
        if calls == 0:
             return np.zeros(n), func(np.zeros(n))
             
        # Track best
        # Global best
        idx_best = np.argmin(F)
        pbest_idx = idx_best
        pbest_F = F[idx_best]
        pbest_X = X[idx_best].copy()
        
        # Personal bests
        pbest_Fs = F.copy()
        pbest_Xs = X.copy()
        
        best_F_overall = pbest_F
        best_X_overall = pbest_X.copy()
        
        # Main loop
        # We need to ensure we don't exceed budget.
        # Each generation consumes m calls.
        # We can do partial generations if budget allows? 
        # Yes, we can evaluate one particle at a time and update if budget is tight.
        # But standard PSO updates velocities based on current pbest/gbest.
        # If we stop mid-generation, the velocities for the rest of the swarm are stale.
        # It's better to complete generations.
        
        # However, to maximize early progress, maybe a smaller swarm with more generations?
        # Let's stick to m=25 for 20 generations.
        # Or m=20 for 25 generations.
        # Let's try m=25.
        
        m = 25
        # Re-initialize if m changed? 
        # The code above used m=30. Let's restart the logic properly.
        
        # --- Reset for m=25 ---
        X = np.random.uniform(-5, 5, (m, n))
        V = np.zeros((m, n))
        F = np.zeros(m)
        calls = 0
        
        for i in range(m):
            if calls >= budget: break
            F[i] = func(X[i])
            calls += 1
            
        if calls == 0:
             return np.zeros(n), func(np.zeros(n))
             
        idx_best = np.argmin(F)
        pbest_Fs = F.copy()
        pbest_Xs = X.copy()
        gbest_F = F[idx_best]
        gbest_X = X[idx_best].copy()
        
        best_F_overall = gbest_F
        best_X_overall = gbest_X.copy()
        
        # Time check
        if time.time() > deadline:
            return best_X_overall, best_F_overall
            
        # Generations
        # We have budget - calls remaining.
        # Each gen uses m calls.
        
        while calls < budget and time.time() < deadline:
            # Check if we can afford a full generation
            if budget - calls < m:
                # Do partial generation? 
                # If we do partial, we update positions for those particles.
                # Let's just break to keep it simple and safe.
                # Or do 1 particle at a time?
                # If we do 1 particle at a time, we can squeeze out the last few calls.
                # But PSO is population based.
                # Let's do a single particle update loop for the remainder.
                
                # Update velocity for particle 0 only?
                # r1, r2 = np.random.rand(2)
                # V[0] = w*V[0] + c1*r1*(pbest_Xs[0]-X[0]) + c2*r2*(gbest_X-X[0])
                # X[0] = X[0] + V[0]
                # ...
                # This is messy. Let's just break.
                break
                
            # Update velocities and positions
            r1 = np.random.rand(m, n)
            r2 = np.random.rand(m, n)
            
            # V = w*V + c1*r1*(pbest - X) + c2*r2*(gbest - X)
            # Apply constriction factor kappa to the whole update?
            # The constriction formula usually replaces w, c1, c2 with kappa*w, kappa*c1, kappa*c2
            # Or it's derived such that the update is stable.
            # Standard: V_new = kappa * (w*V_old + c1*r1*(pbest-X) + c2*r2*(gbest-X))
            # Wait, the constriction coefficient kappa is derived from w, c1, c2.
            # The update rule becomes:
            # V_new = w*V_old + c1*r1*(pbest-X) + c2*r2*(gbest-X)
            # But with constriction, we effectively scale the velocity to prevent divergence.
            # Actually, the formula V_new = kappa * ( ... ) is not standard.
            # The standard PSO with constriction uses:
            # V_new = w*V_old + c1*r1*(pbest-X) + c2*r2*(gbest-X)
            # Where w, c1, c2 are chosen such that the system is stable.
            # The "constriction factor" kappa is often used in the formula:
            # V_new = kappa * ( w*V_old + c1*r1*(pbest-X) + c2*r2*(gbest-X) ) 
            # is INCORRECT.
            # The correct interpretation is that w, c1, c2 are adjusted.
            # Specifically, if we use the standard inertia weight w=0.729, c1=c2=1.4944, it is stable.
            # If we use w=0.729, c1=c2=2.05, we might need to scale.
            # Let's use the standard stable parameters: w=0.729, c1=1.4944, c2=1.4944.
            # This is known to be stable without an explicit kappa multiplier in the code, 
            # as the stability is inherent in the parameter choice.
            
            w_eff = 0.729
            c1_eff = 1.4944
            c2_eff = 1.4944
            
            V = w_eff * V + c1_eff * r1 * (pbest_Xs - X) + c2_eff * r2 * (gbest_X - X)
            
            # Update positions
            X_new = X + V
            # Clip to box
            X_new = np.clip(X_new, -5, 5)
            
            # Evaluate
            for i in range(m):
                if calls >= budget: break
                F[i] = func(X_new[i])
                calls += 1
                
                # Update personal best
                if F[i] < pbest_Fs[i]:
                    pbest_Fs[i] = F[i]
                    pbest_Xs[i] = X_new[i].copy()
                    if F[i] < gbest_F:
                        gbest_F = F[i]
                        gbest_X = X_new[i].copy()
                        best_F_overall = gbest_F
                        best_X_overall = gbest_X.copy()
                        
            X = X_new
            
            # Early stopping check?
            # If no improvement for many generations, break?
            # With 25 gens, it's short.
            
        # Final refinement: (1+1)-ES from best
        if best_X_overall is not None and calls < budget and time.time() < deadline - 5:
            x = best_X_overall.copy()
            fx = best_F_overall
            sigma = 0.5
            # Use remaining budget
            remaining = budget - calls
            # Limit steps to avoid infinite loop if sigma doesn't shrink fast enough
            max_steps = min(remaining, 50)
            
            for _ in range(max_steps):
                if calls >= budget or time.time() > deadline:
                    break
                y = np.clip(x + sigma * np.random.randn(n), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_F_overall:
                        best_F_overall = fx
                        best_X_overall = x.copy()
                else:
                    sigma *= 0.85
                    if sigma < 1e-8:
                        break

        if best_X_overall is None:
            best_X_overall = np.zeros(n)
            best_F_overall = func(best_X_overall)

        return best_X_overall, best_F_overall
# EVOLVE-BLOCK-END
