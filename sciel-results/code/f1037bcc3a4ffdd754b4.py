"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 145.0
    
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    # Max evaluation range
    max_val = 10000
    
    # Compute coverage array efficiently
    def compute_coverage(residues):
        cov = np.zeros(max_val + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start = p  # first positive multiple
            else:
                start = r
            if start <= max_val:
                cov[start::p] += 1
        return cov
    
    def get_L(cov):
        """Find largest L such that cov[1..L] >= 2"""
        if cov[0] < 2 and max_val > 0:
            # Check from 1
            pass
        # Find first index where cov < 2, starting from 1
        below = np.where(cov[1:] < 2)[0]
        if len(below) == 0:
            return max_val
        return below[0]
    
    # Initialize: try r_p = 1 for all
    residues = [1] * n_primes
    cov = compute_coverage(residues)
    L = get_L(cov)
    best_L = L
    best_residues = residues[:]
    
    # Local search with simulated annealing
    current_temp = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        x_target = L + 1
        
        if x_target > max_val:
            break
        
        # Find best single-move improvement
        best_gain = 0
        best_i = -1
        best_new_r = -1
        best_old_r = -1
        
        # Try each prime
        for i in range(n_primes):
            p = primes[i]
            old_r = residues[i]
            new_r = x_target % p
            
            if new_r == old_r:
                continue
            
            # Compute delta in coverage
            # Remove old pattern, add new pattern
            if old_r == 0:
                old_start = p
            else:
                old_start = old_r
            if new_r == 0:
                new_start = p
            else:
                new_start = new_r
            
            # Old coverage points
            old_points = np.arange(old_start, max_val + 1, p) if old_start <= max_val else np.array([], dtype=np.int64)
            new_points = np.arange(new_start, max_val + 1, p) if new_start <= max_val else np.array([], dtype=np.int64)
            
            # Apply change
            cov[old_points] -= 1
            cov[new_points] += 1
            
            # Check if L improved
            new_L = L
            if cov[x_target] >= 2:
                # Scan forward
                x_check = x_target + 1
                while x_check <= max_val and cov[x_check] >= 2:
                    new_L = x_check
                    x_check += 1
            
            # Also check if we broke earlier coverage (L could decrease)
            # For now, only accept if new_L > L
            if new_L > L:
                gain = new_L - L
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_new_r = new_r
                    best_old_r = old_r
            
            # Revert
            cov[old_points] += 1
            cov[new_points] -= 1
        
        if best_i != -1 and best_gain > 0:
            # Apply best move
            i = best_i
            p = primes[i]
            old_r = best_old_r
            new_r = best_new_r
            
            if old_r == 0:
                old_start = p
            else:
                old_start = old_r
            if new_r == 0:
                new_start = p
            else:
                new_start = new_r
            
            old_points = np.arange(old_start, max_val + 1, p) if old_start <= max_val else np.array([], dtype=np.int64)
            new_points = np.arange(new_start, max_val + 1, p) if new_start <= max_val else np.array([], dtype=np.int64)
            
            cov[old_points] -= 1
            cov[new_points] += 1
            residues[i] = new_r
            L += best_gain
            
            if L > best_L:
                best_L = L
                best_residues = residues[:]
        else:
            # No improving move found. Try random perturbations (simulated annealing)
            improved = False
            for _ in range(20):
                if time.time() > deadline:
                    break
                
                # Pick 1-3 random primes to change
                n_changes = random.randint(1, 3)
                indices = random.sample(range(n_primes), n_changes)
                old_values = [residues[i] for i in indices]
                
                for i in indices:
                    p = primes[i]
                    old_r = residues[i]
                    new_r = random.randint(0, p - 1)
                    if new_r == old_r:
                        continue
                    
                    if old_r == 0:
                        old_start = p
                    else:
                        old_start = old_r
                    if new_r == 0:
                        new_start = p
                    else:
                        new_start = new_r
                    
                    old_points = np.arange(old_start, max_val + 1, p) if old_start <= max_val else np.array([], dtype=np.int64)
                    new_points = np.arange(new_start, max_val + 1, p) if new_start <= max_val else np.array([], dtype=np.int64)
                    
                    cov[old_points] -= 1
                    cov[new_points] += 1
                    residues[i] = new_r
                
                new_L = get_L(cov)
                
                # Accept if better, or with probability based on temperature
                if new_L >= L:
                    L = new_L
                    if L > best_L:
                        best_L = L
                        best_residues = residues[:]
                    improved = True
                    break
                else:
                    # Revert
                    for i, old_r in zip(indices, old_values):
                        p = primes[i]
                        new_r = residues[i]
                        
                        if old_r == 0:
                            old_start = p
                        else:
                            old_start = old_r
                        if new_r == 0:
                            new_start = p
                        else:
                            new_start = new_r
                        
                        old_points = np.arange(old_start, max_val + 1, p) if old_start <= max_val else np.array([], dtype=np.int64)
                        new_points = np.arange(new_start, max_val + 1, p) if new_start <= max_val else np.array([], dtype=np.int64)
                        
                        cov[old_points] += 1
                        cov[new_points] -= 1
                        residues[i] = old_r
                    
                    # Simulated annealing acceptance
                    delta = L - new_L
                    if random.random() < np.exp(-delta / max(current_temp, 0.01)):
                        # Accept worse move
                        for i, old_r in zip(indices, old_values):
                            p = primes[i]
                            new_r = random.randint(0, p - 1)
                            if new_r == old_r:
                                continue
                            
                            if old_r == 0:
                                old_start = p
                            else:
                                old_start = old_r
                            if new_r == 0:
                                new_start = p
                            else:
                                new_start = new_r
                            
                            old_points = np.arange(old_start, max_val + 1, p) if old_start <= max_val else np.array([], dtype=np.int64)
                            new_points = np.arange(new_start, max_val + 1, p) if new_start <= max_val else np.array([], dtype=np.int64)
                            
                            cov[old_points] -= 1
                            cov[new_points] += 1
                            residues[i] = new_r
                        
                        L = new_L
                        improved = True
                        break
                    # Otherwise stay with old state (already reverted)
            
            if not improved:
                # Cool down
                current_temp *= cooling
                if current_temp < 0.1:
                    # Try a full random restart
                    residues = [random.randint(0, p - 1) for p in primes]
                    cov = compute_coverage(residues)
                    L = get_L(cov)
                    current_temp = 5.0
    
    return best_residues
# EVOLVE-BLOCK-END
