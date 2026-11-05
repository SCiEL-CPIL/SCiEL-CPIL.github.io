import numpy as np
import time
import random

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Generate all divisors of 51975 = 3^3 * 5^2 * 7 * 11
    divisors = []
    for i in range(3, base + 1, 2):
        if base % i == 0:
            divisors.append(i)
    
    n_divs = len(divisors)
    
    # Precompute residue class indices for each divisor
    # For divisor m, class r covers positions r, r+m, r+2m, ...
    # We'll use numpy slicing for efficiency
    
    def compute_uncovered(residues, divisors):
        """Compute number of uncovered residues given assignment."""
        cover = np.zeros(base, dtype=np.int8)
        for m in divisors:
            r = residues[m]
            cover[r::m] += 1
        return np.sum(cover == 0)
    
    def greedy_init(order=None):
        """Greedy initialization: pick residue for each modulus to maximize new coverage."""
        if order is None:
            order = sorted(divisors, reverse=True)  # Larger moduli first
        residues = {}
        covered = np.zeros(base, dtype=bool)
        for m in order:
            best_r = 0
            best_gain = -1
            for r in range(m):
                gain = np.sum(~covered[r::m])
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            residues[m] = best_r
            covered[best_r::m] = True
        return residues
    
    def local_search(residues, divisors, deadline, max_passes=200):
        """Local search: iteratively improve each residue assignment."""
        cover = np.zeros(base, dtype=np.int8)
        for m in divisors:
            r = residues[m]
            cover[r::m] += 1
        uncovered = np.sum(cover == 0)
        
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            for m in divisors:
                r_old = residues[m]
                cover[r_old::m] -= 1
                zeros_before = np.sum(cover == 0)
                
                best_r = r_old
                best_uncovered_after = uncovered  # We want to minimize uncovered after
                
                # Try all residues for this modulus
                for r in range(m):
                    cover[r::m] += 1
                    unc = np.sum(cover == 0)
                    cover[r::m] -= 1
                    if unc < best_uncovered_after:
                        best_uncovered_after = unc
                        best_r = r
                
                if best_r != r_old:
                    cover[best_r::m] += 1
                    residues[m] = best_r
                    uncovered = best_uncovered_after
                    improved = True
                else:
                    cover[r_old::m] += 1
        
        return residues, uncovered
    
    def simulated_annealing(residues, divisors, deadline, T_start=10.0, T_end=0.1, cooling=0.995):
        """Simulated annealing to escape local optima."""
        cover = np.zeros(base, dtype=np.int8)
        for m in divisors:
            r = residues[m]
            cover[r::m] += 1
        current_uncovered = np.sum(cover == 0)
        best_residues = dict(residues)
        best_uncovered = current_uncovered
        
        T = T_start
        divisors_list = list(divisors)
        
        while T > T_end and time.time() < deadline:
            # Pick a random modulus
            m = random.choice(divisors_list)
            r_old = residues[m]
            r_new = random.randint(0, m - 1)
            if r_new == r_old:
                T *= cooling
                continue
            
            # Compute change in uncovered
            cover[r_old::m] -= 1
            cover[r_new::m] += 1
            new_uncovered = np.sum(cover == 0)
            delta = new_uncovered - current_uncovered
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                current_uncovered = new_uncovered
                residues[m] = r_new
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_residues = dict(residues)
            else:
                # Revert
                cover[r_new::m] -= 1
                cover[r_old::m] += 1
            
            T *= cooling
        
        return best_residues, best_uncovered
    
    def swap_local_search(residues, divisors, deadline):
        """Try swapping residues between pairs of moduli."""
        cover = np.zeros(base, dtype=np.int8)
        for m in divisors:
            r = residues[m]
            cover[r::m] += 1
        current_uncovered = np.sum(cover == 0)
        best_uncovered = current_uncovered
        best_residues = dict(residues)
        
        divisors_list = list(divisors)
        n = len(divisors_list)
        
        # Iterate over pairs
        for i in range(n):
            if time.time() >= deadline:
                break
            m1 = divisors_list[i]
            for j in range(i + 1, n):
                if time.time() >= deadline:
                    break
                m2 = divisors_list[j]
                r1 = residues[m1]
                r2 = residues[m2]
                
                # Try swapping
                cover[r1::m1] -= 1
                cover[r2::m2] -= 1
                cover[r2::m1] += 1
                cover[r1::m2] += 1
                new_uncovered = np.sum(cover == 0)
                
                if new_uncovered < best_uncovered:
                    best_uncovered = new_uncovered
                    best_residues = dict(residues)
                    # Apply swap
                    residues[m1] = r2
                    residues[m2] = r1
                else:
                    # Revert
                    cover[r2::m1] -= 1
                    cover[r1::m2] -= 1
                    cover[r1::m1] += 1
                    cover[r2::m2] += 1
        
        return best_residues, best_uncovered
    
    # Main optimization loop with multiple restarts
    best_residues = None
    best_uncovered = base  # Worst case: nothing covered
    
    # First, try greedy with different orderings
    orderings = [
        sorted(divisors, reverse=True),  # Largest first
        sorted(divisors),  # Smallest first
        sorted(divisors, key=lambda m: -base // m),  # By number of residues covered
    ]
    
    for order in orderings:
        if time.time() >= deadline:
            break
        residues = greedy_init(order)
        residues, uncovered = local_search(residues, divisors, deadline, max_passes=50)
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_residues = dict(residues)
    
    # Now do simulated annealing from the best solution
    if best_residues is not None and time.time() < deadline:
        residues, uncovered = simulated_annealing(best_residues, divisors, deadline, T_start=5.0, T_end=0.05, cooling=0.998)
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_residues = dict(residues)
    
    # More local search passes
    if best_residues is not None and time.time() < deadline:
        residues, uncovered = local_search(best_residues, divisors, deadline, max_passes=100)
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_residues = dict(residues)
    
    # Swap local search
    if best_residues is not None and time.time() < deadline:
        residues, uncovered = swap_local_search(best_residues, divisors, deadline)
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_residues = dict(residues)
    
    # Final local search
    if best_residues is not None and time.time() < deadline:
        residues, uncovered = local_search(best_residues, divisors, deadline, max_passes=200)
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_residues = dict(residues)
    
    # If we still have time, do more SA restarts
    if best_residues is not None and time.time() < deadline - 5.0:
        for restart in range(10):
            if time.time() >= deadline - 2.0:
                break
            # Perturb the best solution
            residues = dict(best_residues)
            # Randomly change 3-5 residues
            for _ in range(random.randint(3, 5)):
                m = random.choice(divisors)
                residues[m] = random.randint(0, m - 1)
            
            # Local search
            residues, uncovered = local_search(residues, divisors, deadline - 2.0, max_passes=30)
            if uncovered < best_uncovered:
                best_uncovered = uncovered
                best_residues = dict(residues)
            
            # SA
            if time.time() < deadline - 3.0:
                residues, uncovered = simulated_annealing(best_residues, divisors, deadline - 2.0, T_start=3.0, T_end=0.05, cooling=0.999)
                if uncovered < best_uncovered:
                    best_uncovered = uncovered
                    best_residues = dict(residues)
    
    if best_residues is None:
        # Fallback: all zeros
        best_residues = {m: 0 for m in divisors}
    
    return [[m, best_residues[m]] for m in divisors]
