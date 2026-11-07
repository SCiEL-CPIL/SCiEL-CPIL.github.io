"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Population-based Differential Evolution (DE/rand/1) with Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Time guard: return best if we exceed 310s (kill is at 330s)
        start_time = time.time()
        deadline = start_time + 310.0

        # DE Parameters
        pop_size = 10
        # 5000 function evals is safe for 300s limit. 
        # With pop_size 10, that's 500 generations.
        # Budget is 300, so we might hit budget limit before time limit usually, 
        # but if func is slow, time limit triggers.
        # Actually, budget is 300 calls total. 
        # Wait, the contract says "budget=300". 
        # "Every call is recorded... More than 300 calls... makes result invalid".
        # So I am strictly limited to 300 function evaluations.
        
        # With 300 evals and pop_size 10, I can do 30 generations.
        # This is very few for DE. 
        # Maybe smaller population? 
        # If pop_size=5, 60 generations. 
        # If pop_size=10, 30 generations.
        # DE usually needs lambda >= 5*dim for good performance, so 25+ for dim=5.
        # But budget is hard capped at 300.
        
        # Let's try a hybrid:
        # Phase 1: Multi-start local search? No, DE is better for global.
        # Let's use a smaller population to get more generations, or accept fewer generations.
        # Actually, for dim=5, pop_size=10 is on the low side but acceptable for 300 evals.
        # Let's stick to pop_size=10.
        
        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fpop = np.array([func(x) for x in pop])
        
        best_idx = np.argmin(fpop)
        best_x = pop[best_idx].copy()
        best_fx = fpop[best_idx]
        
        calls_made = pop_size
        
        # Adaptive Mutation Factor F and Crossover Rate CR
        # Start with F=0.8, CR=0.9 (standard for small populations)
        F = 0.8
        CR = 0.9
        
        # Track stagnation
        no_improve_gens = 0
        
        while calls_made < budget:
            # Time check
            if time.time() > deadline:
                break
            
            gen_best_fx = best_fx
            gen_best_x = best_x
            
            # Generate offspring
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Select 3 distinct random individuals
                idxs = rng.choice(pop_size, 4, replace=False)
                a, b, c, j = idxs
                if j == i:
                    # Ensure j is not i for standard DE, though some variants allow it
                    j = idxs[3]
                    if j == i:
                        j = (i + 1) % pop_size
                        if j == a: j = (j+1)%pop_size
                        # This logic is getting messy, let's just pick randomly until distinct
                        while j == i or j == a or j == b or j == c:
                            j = rng.integers(0, pop_size)
                            
                # Mutation: v = a + F * (b - c)
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Crossover
                u = np.zeros(dim)
                # Random crossover
                for k in range(dim):
                    if rng.random() < CR or k == j:
                        u[k] = v[k]
                    else:
                        u[k] = pop[i][k]
                        
                # Clip to bounds
                u = np.clip(u, -5, 5)
                
                # Evaluate
                fu = func(u)
                calls_made += 1
                
                # Selection
                if fu <= fpop[i]:
                    pop[i] = u
                    fpop[i] = fu
                    
                    if fu < gen_best_fx:
                        gen_best_fx = fu
                        gen_best_x = u.copy()
                        
            # Update global best
            if gen_best_fx < best_fx:
                best_fx = gen_best_fx
                best_x = gen_best_x
                no_improve_gens = 0
            else:
                no_improve_gens += 1
                
            # Adaptive strategy:
            # If stuck, increase F (larger steps) and decrease CR (more mutation dominance)
            # Or restart population around best?
            if no_improve_gens > 5:
                # Restart population: keep best, re-initialize others
                pop[0] = best_x
                fpop[0] = best_fx
                for i in range(1, pop_size):
                    if calls_made >= budget:
                        break
                    # Local restart around best
                    pop[i] = np.clip(best_x + rng.normal(0, 1.0, dim), -5, 5)
                    fpop[i] = func(pop[i])
                    calls_made += 1
                no_improve_gens = 0
                # Reset F and CR
                F = 0.8
                CR = 0.9
            else:
                # Slight adaptation
                # If improving, decrease F slightly to refine?
                # Standard DE doesn't adapt much without more complex schemes.
                # Let's keep F constant for simplicity and robustness, 
                # or use a simple schedule.
                pass

        # Final Polish with Nelder-Mead
        # Use remaining budget for local refinement
        remaining = budget - calls_made
        if remaining > 10:
            # Limit Nelder-Mead iterations to avoid timeout or budget overrun
            # NM in 5D is fast, but func calls count.
            # maxiter=100 is usually safe for 5D if func is cheap.
            # We have up to ~300 calls total.
            # If calls_made is low, we can use more.
            
            # Simple NM wrapper
            def nm_func(x):
                return func(np.clip(x, -5, 5))
                
            try:
                # options: maxiter, xatol, fatol
                # We want to use up to 'remaining' calls roughly.
                # NM calls func multiple times per iteration.
                # In 5D, one iteration is ~6 calls.
                # So maxiter = remaining // 6
                max_iter = max(1, remaining // 6)
                
                res = scipy.optimize.minimize(
                    nm_func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                
                if res.fun < best_fx:
                    best_x = np.clip(res.x, -5, 5)
                    best_fx = res.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
