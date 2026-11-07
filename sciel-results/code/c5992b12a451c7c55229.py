"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using Differential Evolution."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self._ref = np.array([3000.0, 0.06])
        self._eval_count = 0
        self._deadline = None
        self._problem = None
        self._lo = None
        self._hi = None

    def __call__(self, problem, bounds):
        start_time = time.time()
        self._deadline = start_time + 320.0  # Safety margin before 330s kill
        self._problem = problem
        self._lo = np.array([b[0] for b in bounds])
        self._hi = np.array([b[1] for b in bounds])
        self._eval_count = 0

        dim = self.dim
        lo = self._lo
        hi = self._hi
        budget = self.budget
        
        # DE Parameters
        # Population size should be reasonable for 4D. 
        # If budget is 200, and we want maybe 10-20 generations, pop_size ~ 10-20.
        # Let's aim for a population size that allows for several iterations.
        # Standard DE often uses NP >= 5*D. D=4, so NP >= 20.
        # Let's try NP = 20. 
        # Generations = (budget - NP) / NP = (200 - 20) / 20 = 9 generations.
        # This seems tight. Maybe smaller population for more generations?
        # NP = 10? Generations = 19.
        # Let's stick to NP = 15 or 20. 
        # Actually, for small budgets, smaller populations with more generations might explore better if mutation is good.
        # Let's use NP = 12. (Budget 200 -> ~16 generations).
        
        NP = 12
        # Ensure NP is at least 4 (for DE/rand/1)
        if NP < 4:
            NP = 4
            
        # If budget is very small, adjust NP
        if budget < NP * 2:
             NP = max(4, budget // 2)
             
        # Differential Evolution weights
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        
        rng = np.random.default_rng(42) # Use fixed seed for reproducibility if needed, or global?
        # The task says evaluator calls np.random.seed(seed). 
        # Using np.random.default_rng(42) ignores that. 
        # It's better to use the global numpy random state or derive from it to respect the seed.
        # However, the task says "For seeds 0 and 1 the evaluator calls np.random.seed(seed)".
        # So we should use np.random functions directly to pick up the seed.
        
        # Initialize population
        pop = lo + (hi - lo) * np.random.random((NP, dim))
        
        # Evaluate initial population
        fitness = np.zeros((NP, self.n_obj))
        for i in range(NP):
            if self._eval_count >= budget or time.time() > self._deadline:
                break
            fitness[i] = self._evaluate(pop[i])
            
        # Keep track of all evaluated points to compute hypervolume at the end?
        # No, the evaluator records every returned objective vector.
        # So we just need to make sure we call self._evaluate for every new candidate.
        
        # DE Loop
        while self._eval_count < budget and time.time() < self._deadline:
            # Check if we can afford a full generation
            # Each generation evaluates up to NP candidates (plus some failures)
            # Actually, in standard DE, we generate 1 trial vector per individual.
            # So NP evaluations per generation.
            
            # If remaining budget < NP, we might do a partial generation or stop.
            # Let's just run generation if we have at least 4 evals left (min for DE/rand/1)
            if self._eval_count + 4 > budget:
                # Do random sampling with remaining budget
                remaining = budget - self._eval_count
                for _ in range(remaining):
                    if self._eval_count >= budget or time.time() > self._deadline:
                        break
                    x_rand = lo + (hi - lo) * np.random.random(dim)
                    self._evaluate(x_rand)
                break
                
            # Generate trial vectors
            for i in range(NP):
                if self._eval_count >= budget or time.time() > self._deadline:
                    break
                    
                # Select three distinct random indices
                # We need i, r1, r2, r3 to be distinct
                while True:
                    r1 = np.random.randint(0, NP)
                    r2 = np.random.randint(0, NP)
                    r3 = np.random.randint(0, NP)
                    if r1 != r2 and r1 != r3 and r2 != r3 and i != r1 and i != r2 and i != r3:
                        break
                        
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Clip to bounds
                v = np.clip(v, lo, hi)
                
                # Crossover: u[j] = v[j] if rand() < CR or j == j_rand else x_i[j]
                # Ensure at least one gene is from v
                j_rand = np.random.randint(0, dim)
                u = np.where(np.random.random(dim) < CR, v, pop[i])
                u[j_rand] = v[j_rand]
                
                # Evaluate trial vector
                f_u = self._evaluate(u)
                
                # Selection: replace x_i if u dominates x_i
                # u dominates x_i if all(f_u <= f_i) and any(f_u < f_i)
                if np.all(f_u <= fitness[i]) and np.any(f_u < fitness[i]):
                    pop[i] = u
                    fitness[i] = f_u
                    
        return None

    def _evaluate(self, x):
        """Evaluate a point, tracking budget and time."""
        if self._eval_count >= self.budget:
            raise RuntimeError("Budget exceeded")
        if time.time() > self._deadline:
            raise TimeoutError("Time limit exceeded")
            
        # Clip to bounds
        x = np.clip(x, self._lo, self._hi)
        
        # Call problem
        f = self._problem(x)
        
        self._eval_count += 1
        return f


# EVOLVE-BLOCK-END
