"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

N = 5000
_DEADLINE = time.time() + 320.0  # 330s wall-clock, 10s safety margin

# EVOLVE-BLOCK-START

def _primes(limit):
    """Return list of primes <= limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]

# Precompute primes up to N
_all_primes = _primes(N)

def _precompute_residue_indices(primes_to_use):
    """
    For each prime p in primes_to_use, precompute the list of indices 
    in [0, N] for each residue class mod p.
    Returns a dict: {p: {r: [indices]}}
    """
    indices_by_prime = {}
    indices = np.arange(N + 1)
    for p in primes_to_use:
        residues_dict = {}
        # Efficient way: group indices by mod p
        # We can do this in a loop or using np.unique, but loop is fine for N=5000
        # Actually, creating lists of lists is memory heavy if not careful, but N is small.
        for r in range(p):
            # Find all x in [0, N] such that x % p == r
            # x = k*p + r
            # 0 <= k*p + r <= N
            # k_min = 0 (if r <= N)
            # k_max = (N - r) // p
            if r > N:
                residues_dict[r] = []
            else:
                k_max = (N - r) // p
                # Generate range
                # Using np.arange is fast
                # But we need to store in a structure accessible by prime
                pass
        
        # Let's just store the masks or indices
        # For fast updates, we need to iterate over indices.
        # Let's store lists of numpy arrays.
        res_arrs = []
        for r in range(p):
            # x = r, r+p, r+2p...
            # Use np.arange(r, N+1, p)
            res_arrs.append(np.arange(r, N + 1, p))
        indices_by_prime[p] = res_arrs
        
    return indices_by_prime

class AdmissibleTupleSolver:
    def __init__(self, k, primes_to_use):
        self.k = k
        self.primes_to_use = primes_to_use
        self.P = len(primes_to_use)
        self.indices_by_prime = _precompute_residue_indices(primes_to_use)
        
        # Initialize residues randomly
        self.residues = [random.randint(0, p - 1) for p in self.primes_to_use]
        
        # Initialize cover_count and alive status
        self.cover_count = np.zeros(N + 1, dtype=np.int32)
        self.alive = np.ones(N + 1, dtype=bool)
        
        for i, p in enumerate(self.primes_to_use):
            r = self.residues[i]
            indices = self.indices_by_prime[p][r]
            if len(indices) > 0:
                self.cover_count[indices] += 1
                
        self.alive = (self.cover_count == 0)
        self.survivor_count = int(np.sum(self.alive))
        
    def _update_residue(self, i, new_r):
        """
        Change residue of prime i to new_r.
        Returns delta in survivor_count.
        """
        p = self.primes_to_use[i]
        old_r = self.residues[i]
        
        if old_r == new_r:
            return 0
            
        delta = 0
        
        # Remove old residue contribution
        old_indices = self.indices_by_prime[p][old_r]
        if len(old_indices) > 0:
            # cover_count decreases by 1 for these indices
            # If cover_count becomes 0, they become alive (+1 to survivor_count)
            # If cover_count was 1, it becomes 0.
            # If cover_count was > 1, it becomes > 0 (still dead).
            
            # We need to check the current cover_count of these indices
            # Before update: cover_count[old_indices]
            # After update: cover_count[old_indices] - 1
            
            # Only those with cover_count == 1 will become alive.
            # count_becoming_alive = np.sum(self.cover_count[old_indices] == 1)
            
            # We also need to update self.cover_count
            self.cover_count[old_indices] -= 1
            
            # Calculate delta
            # Those that were 1 become 0 -> Alive
            # Those that were > 1 become > 0 -> Dead (no change in alive status, just cover count)
            
            # Efficient way:
            # We can't easily get the count of 1s without a pass.
            # But N is small.
            # Let's just do a pass or use numpy.
            # numpy sum is fast.
            count_becoming_alive = np.sum(self.cover_count[old_indices] == 0) # After decrement, 0 means was 1
            delta += count_becoming_alive
            
        # Add new residue contribution
        new_indices = self.indices_by_prime[p][new_r]
        if len(new_indices) > 0:
            # cover_count increases by 1 for these indices
            # If cover_count was 0, it becomes 1 -> Dead (-1 to survivor_count)
            
            # Check how many were 0 before increment
            count_becoming_dead = np.sum(self.cover_count[new_indices] == 0)
            delta -= count_becoming_dead
            
            self.cover_count[new_indices] += 1
            
        self.residues[i] = new_r
        self.survivor_count += delta
        
        return delta

    def simulated_annealing(self, max_steps, start_temp, end_temp):
        """
        Run Simulated Annealing to maximize survivor_count.
        """
        current_temp = start_temp
        step_size = (end_temp - start_temp) / max_steps
        best_survivors = self.survivor_count
        best_residues = self.residues.copy()
        
        # We want to reach survivor_count >= self.k
        # If we reach it, we can stop early? 
        # Or we can just maximize to see how high we can go.
        
        for step in range(max_steps):
            if time.time() > _DEADLINE - 15.0:
                break
                
            # Pick a random prime index
            i = random.randint(0, self.P - 1)
            p = self.primes_to_use[i]
            
            # Pick a random new residue
            new_r = random.randint(0, p - 1)
            
            old_survivors = self.survivor_count
            delta = self._update_residue(i, new_r)
            new_survivors = self.survivor_count
            
            # Calculate acceptance probability
            if new_survivors > old_survivors:
                # Always accept
                pass
            else:
                # Accept with probability exp(-delta / temp) where delta is negative
                # delta_surv = new - old < 0
                # energy_increase = old - new > 0
                energy_increase = old_survivors - new_survivors
                if energy_increase == 0:
                    # Tie, accept?
                    pass
                else:
                    # If temp is very low, reject
                    if current_temp < 1e-9:
                        # Revert
                        self._update_residue(i, self.residues[i]) # Wait, residues[i] is already new_r
                        # Need to revert to old_r
                        # We stored old_r? No.
                        # Let's store old_r before update.
                        # Actually, _update_residue updates self.residues[i].
                        # So to revert, we call _update_residue(i, old_r).
                        # But we didn't store old_r in this scope easily? 
                        # Yes we did: old_r = self.residues[i] was taken BEFORE update?
                        # No, in _update_residue, old_r is local.
                        # In SA loop, we need to know old_r to revert.
                        # Let's modify _update_residue to return old_r? Or just store it.
                        
                        # Let's just re-fetch or store it.
                        # I'll refactor the loop to be cleaner.
                        pass
            
            # This implementation is getting messy with reverts.
            # Let's write a cleaner SA step.
            break # Break to fix logic
            
        return best_survivors, best_residues

    def sa_clean(self, max_steps, start_temp, end_temp):
        """
        Cleaner Simulated Annealing.
        """
        current_temp = start_temp
        step_size = (end_temp - start_temp) / max_steps
        best_survivors = self.survivor_count
        best_residues = self.residues.copy()
        
        for step in range(max_steps):
            if time.time() > _DEADLINE - 15.0:
                break
                
            i = random.randint(0, self.P - 1)
            p = self.primes_to_use[i]
            old_r = self.residues[i]
            new_r = random.randint(0, p - 1)
            
            if old_r == new_r:
                continue
                
            old_survivors = self.survivor_count
            delta = self._update_residue(i, new_r)
            new_survivors = self.survivor_count
            
            accepted = False
            if new_survivors > old_survivors:
                accepted = True
            else:
                energy_increase = old_survivors - new_survivors
                if energy_increase > 0:
                    # Calculate probability
                    # Avoid overflow/underflow
                    if current_temp > 1e-9:
                        prob = math.exp(-energy_increase / current_temp)
                        if random.random() < prob:
                            accepted = True
                    else:
                        accepted = False
                else:
                    accepted = True # Should not happen if new <= old and increase > 0
            
            if not accepted:
                # Revert
                self._update_residue(i, old_r)
            
            if self.survivor_count > best_survivors:
                best_survivors = self.survivor_count
                best_residues = self.residues.copy()
                
            # Update temperature
            current_temp += step_size
            if current_temp < 1e-9:
                current_temp = 1e-9
                
        return best_survivors, best_residues

def construct():
    """
    Constructs the largest admissible tuple in [0, N].
    Strategy:
    1. Binary search for max k.
    2. For each k, use Simulated Annealing to find residues with survivors >= k.
    """
    
    # Primes up to N
    # We need to determine the range of k to search.
    # Max k is N+1.
    # Min k is 1.
    
    # Heuristic upper bound for k:
    # Density of survivors ~ e^-gamma / ln(k) * k ? No.
    # If we choose residues optimally, we can keep more.
    # But generally, k is around N * prod(1-1/p) ?
    # For k=100, prod ~ 0.12. N=5000. 600 survivors.
    # So k=100 is feasible.
    # For k=500, primes up to 500. prod ~ e^-gamma / ln(500) ~ 0.56 / 6.2 ~ 0.09.
    # 5000 * 0.09 = 450.
    # So k=500 might be infeasible? 450 < 500.
    # So max k is likely around 200-300?
    
    # Let's start with a binary search.
    low = 1
    high = 500  # Start with a conservative upper bound, expand if needed?
    # Actually, let's just search up to 500. If 500 is feasible, try higher.
    # But 5000 is the max range.
    
    best_k = 1
    best_residues = None
    best_alive = None
    
    # To save time, we will not do a full binary search over 5000.
    # We will try a few strategic points.
    # Or a binary search with a time limit per check.
    
    # Let's define a helper to check if k is feasible using SA.
    def is_feasible(k, time_budget):
        primes_to_use = [p for p in _all_primes if p <= k]
        if not primes_to_use:
            return True, [], np.ones(N+1, dtype=bool)
            
        solver = AdmissibleTupleSolver(k, primes_to_use)
        
        # Run SA
        # Steps: proportional to time budget?
        # Let's run for a fixed number of steps or until time runs out.
        # Max steps: 50000?
        # Each step is ~ N/p ops.
        
        # We want to maximize survivors.
        # If max_survivors >= k, it's feasible.
        
        # To speed up, we can stop SA early if we reach k.
        # But SA doesn't know k directly in the objective (it maximizes survivors).
        # We can modify SA to stop if survivor_count >= k.
        
        # Let's add an early stop condition to SA.
        # I'll modify sa_clean to take a target.
        
        # For now, let's just run SA for a short time and see the max survivors.
        
        # Time budget per check: 10 seconds?
        # If we have 300s, we can do 30 checks.
        
        start_time = time.time()
        max_steps = 10000 # Adjust based on cost
        # Cost per step is variable. Let's just use a step limit and time limit.
        
        best_survivors, best_res = solver.sa_clean(max_steps, 10.0, 0.1)
        
        # Check time
        elapsed = time.time() - start_time
        
        if best_survivors >= k:
            # Reconstruct alive
            alive = np.ones(N + 1, dtype=bool)
            for p, r in zip(primes_to_use, best_res):
                alive[np.arange(r, N + 1, p)] = False
            return True, best_res, alive
        else:
            return False, best_res, None

    # Search strategy:
    # Try k = 100, 200, 300, 400, 500...
    # Or binary search.
    
    # Let's try a linear search with jumps to find the boundary roughly, then binary search.
    # Or just binary search on [1, 500].
    
    low = 1
    high = 500
    
    while low <= high:
        if time.time() > _DEADLINE - 30.0:
            break
            
        mid = (low + high) // 2
        
        feasible, res, alive = is_feasible(mid, 10.0)
        
        if feasible:
            best_k = mid
            best_residues = res
            best_alive = alive
            low = mid + 1
        else:
            high = mid - 1
            
    # If we found a feasible k, try to push it higher?
    # The binary search should find the max in the range [1, 500].
    # What if max k > 500?
    # Unlikely, but let's check if 500 was feasible.
    if best_k == 500:
        # Try 600, 700...
        for k in range(600, 1000, 100):
            if time.time() > _DEADLINE - 30.0:
                break
            feasible, res, alive = is_feasible(k, 10.0)
            if feasible:
                best_k = k
                best_residues = res
                best_alive = alive
            else:
                break
                
    # Final construction
    if best_residues is None:
        return []
        
    # Verify and construct
    final_primes_to_use = [p for p in _all_primes if p <= best_k]
    alive = np.ones(N + 1, dtype=bool)
    for p, r in zip(final_primes_to_use, best_residues):
        alive[np.arange(r, N + 1, p)] = False
        
    result_indices = np.where(alive)[0]
    
    if len(result_indices) < best_k:
        # Fallback: reduce k to len(result_indices)
        # But we need to return a valid tuple of size k.
        # If we can't get best_k, we should return a smaller valid one.
        # The score is the size. So we want the max size.
        # If our check was wrong, we might have a smaller set.
        # Let's just take what we have.
        pass
        
    final_set = result_indices[:best_k].tolist()
    
    return final_set

# EVOLVE-BLOCK-END
