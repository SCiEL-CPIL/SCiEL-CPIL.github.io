"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer using NSGA-II.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.pop_size = 50  # Fixed population size for NSGA-II
        self.crossover_prob = 0.9
        self.mutation_prob = 1.0 / dim
        self.eta_c = 20.0  # Distribution index for SBX
        self.eta_m = 20.0  # Distribution index for Polynomial Mutation

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Initialize Population
        pop = lo + (hi - lo) * np.random.random((self.pop_size, self.dim))
        F = np.array([problem(x) for x in pop])
        
        # Track total evaluations
        evals_used = self.pop_size
        
        # Main Loop
        while evals_used < self.budget:
            # Generate Offspring
            offspring = np.empty_like(pop)
            for i in range(self.pop_size):
                # Selection: Binary Tournament
                p1, p2 = self._tournament_select(pop, F)
                
                # Crossover: Simulated Binary Crossover (SBX)
                if np.random.random() < self.crossover_prob:
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Mutation: Polynomial Mutation
                c1 = self._polynomial_mutation(c1, lo, hi)
                c2 = self._polynomial_mutation(c2, lo, hi)
                
                offspring[i] = c1
                # Store second child for next iteration if possible, 
                # but standard NSGA-II often generates pairs. 
                # To keep it simple and fit budget, we generate 1 child per loop iteration 
                # but use 2 parents. This is slightly less efficient than generating 
                # 2 children but easier to control budget strictly.
                # Alternatively, we can generate 2 children per 2 parents and 
                # adjust loop logic. Let's stick to 1 child per iteration for 
                # strict budget control (1 eval per child).
                
                evals_used += 1
                if evals_used >= self.budget:
                    break

            if evals_used >= self.budget:
                break

            # Evaluate Offspring
            F_offspring = np.array([problem(x) for x in offspring[:self.pop_size]])
            
            # Combine Parents and Offspring
            P = np.vstack((pop, offspring[:self.pop_size]))
            F_P = np.vstack((F, F_offspring))
            
            # Non-dominated Sorting
            ranks = self._non_dominated_sort(F_P)
            
            # Crowding Distance
            # We need to assign crowding distances to fronts
            # For simplicity in this implementation, we calculate CD for the whole population
            # and use it for selection within fronts.
            # Proper NSGA-II calculates CD per front.
            
            # Select Next Generation
            # We need to pick self.pop_size individuals from P (size 2*pop_size)
            # Based on rank (lower is better) and Crowding Distance (higher is better)
            
            # First, sort by rank
            # Then, for each rank, sort by crowding distance in descending order
            
            # Calculate Crowding Distance for P
            # Note: CD is only meaningful within a non-dominated front.
            # We will compute CD for the entire population but only use it 
            # to break ties within the same rank during selection.
            
            cd = self._crowding_distance(F_P)
            
            # Selection Strategy:
            # 1. Create a list of indices sorted by (rank, -crowding_distance)
            # 2. Take the first self.pop_size indices.
            
            # However, standard NSGA-II adds fronts one by one.
            # If the next front doesn't fit, we take the most diverse ones from it.
            
            selected_indices = []
            current_rank = 0
            while len(selected_indices) < self.pop_size:
                # Get indices of individuals with current rank
                mask = (ranks == current_rank)
                front_indices = np.where(mask)[0]
                
                if len(selected_indices) + len(front_indices) <= self.pop_size:
                    # Add entire front
                    selected_indices.extend(front_indices.tolist())
                else:
                    # Add only some from this front
                    # Sort this front by crowding distance (descending)
                    # Note: CD calculation should ideally be relative to this front.
                    # For a quick approximation, we use the global CD or recalculate.
                    # Let's recalculate CD for just this front to be accurate.
                    F_front = F_P[front_indices]
                    cd_front = self._crowding_distance(F_front)
                    # Sort indices in front by cd_front descending
                    sorted_front = front_indices[np.argsort(-cd_front)]
                    needed = self.pop_size - len(selected_indices)
                    selected_indices.extend(sorted_front[:needed].tolist())
                    break
                current_rank += 1
                if current_rank > np.max(ranks):
                    break

            # Update population
            pop = P[selected_indices]
            F = F_P[selected_indices]

        return pop

    def _tournament_select(self, pop, F):
        """Binary tournament selection based on non-dominated sorting."""
        # For simplicity and speed in this constrained budget, we use a simple 
        # random selection if we don't have ranks pre-computed for the current pop.
        # However, NSGA-II relies on ranks. 
        # Since we update pop at the end of the loop, we need ranks for the current pop.
        # Let's compute ranks for the current pop before selection.
        # This adds overhead. 
        # Alternative: Use simple random selection for the "exploration" part 
        # and rely on the elitism in the combine step? 
        # No, that's not NSGA-II.
        
        # Let's compute ranks for the current population (size N)
        # This is O(N^2 * M) which is fine for N=50, M=2.
        ranks = self._non_dominated_sort(F)
        
        # Select two random individuals
        i1 = np.random.randint(0, len(pop))
        i2 = np.random.randint(0, len(pop))
        
        # Compare ranks
        if ranks[i1] < ranks[i2]:
            return pop[i1], pop[i2]
        elif ranks[i2] < ranks[i1]:
            return pop[i2], pop[i1]
        else:
            # Same rank, use crowding distance
            # Compute CD for the two? Or just pick random?
            # To be proper, we should check CD.
            # For efficiency, we can just pick random if ranks are equal, 
            # or compute CD for the whole pop again.
            # Let's pick random for now to save time, as N is small.
            if np.random.random() < 0.5:
                return pop[i1], pop[i2]
            else:
                return pop[i2], pop[i1]

    def _sbx_crossover(self, p1, p2, lo, hi):
        """Simulated Binary Crossover."""
        c1 = np.empty_like(p1)
        c2 = np.empty_like(p2)
        for i in range(self.dim):
            if np.random.random() < 0.5:
                c1[i], c2[i] = p1[i], p2[i]
            else:
                if abs(p1[i] - p2[i]) < 1e-14:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
                else:
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (self.eta_c + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (self.eta_c + 1))
                    
                    c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                    
                    # Clamp to bounds
                    c1[i] = np.clip(c1[i], lo[i], hi[i])
                    c2[i] = np.clip(c2[i], lo[i], hi[i])
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi):
        """Polynomial Mutation."""
        for i in range(self.dim):
            if np.random.random() < self.mutation_prob:
                delta = np.random.random()
                if delta < 0.5:
                    delta1 = (2 * delta) ** (1.0 / (self.eta_m + 1)) - 1
                else:
                    delta1 = 1 - (2 * (1 - delta)) ** (1.0 / (self.eta_m + 1))
                
                x[i] = x[i] + delta1 * (hi[i] - lo[i])
                x[i] = np.clip(x[i], lo[i], hi[i])
        return x

    def _non_dominated_sort(self, F):
        """
        Assigns non-dominated ranks to each point in F.
        Rank 0 is the first non-dominated front, Rank 1 the next, etc.
        """
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        
        # Calculate domination relationships
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(F[i], F[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(F[j], F[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
        
        # First front
        front = [i for i in range(n) if domination_count[i] == 0]
        rank = 0
        while front:
            next_front = []
            for i in front:
                ranks[i] = rank
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            front = next_front
            rank += 1
            
        return ranks

    def _dominates(self, f1, f2):
        """
        Returns True if f1 dominates f2 (minimization).
        f1 <= f2 in all objectives and f1 < f2 in at least one.
        """
        if not np.all(f1 <= f2):
            return False
        if not np.any(f1 < f2):
            return False
        return True

    def _crowding_distance(self, F):
        """
        Calculates crowding distance for a set of non-dominated points.
        Assumes F is a front (all non-dominated).
        """
        n = len(F)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([np.inf])
            
        cd = np.zeros(n)
        for m in range(self.n_obj):
            # Sort by objective m
            sorted_indices = np.argsort(F[:, m])
            cd[sorted_indices[0]] = np.inf
            cd[sorted_indices[-1]] = np.inf
            f_min = F[sorted_indices[0], m]
            f_max = F[sorted_indices[-1], m]
            if f_max - f_min > 1e-14:
                for i in range(1, n - 1):
                    cd[sorted_indices[i]] += (F[sorted_indices[i + 1], m] - F[sorted_indices[i - 1], m]) / (f_max - f_min)
            else:
                # All values are the same
                pass
                
        return cd
# EVOLVE-BLOCK-END
