"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _max_abs_ap_sum(signs):
    """
    Compute the maximum absolute signed sum over any arithmetic progression
    contained in [1, n] for the given signs (list of length n, 0-indexed).
    signs[i] corresponds to position i+1.
    """
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    max_disc = 1  # Singletons always have abs sum 1
    
    # Iterate over all possible steps d (1 to n)
    for d in range(1, n + 1):
        # For a fixed step d, the APs are determined by their start index s.
        # In 0-indexed terms, start s ranges from 0 to d-1 (mod d).
        # Actually, for any start position i in [0, n-1], the AP is i, i+d, i+2d, ...
        # However, APs with the same residue modulo d are disjoint.
        # We can iterate over residues r in [0, d-1].
        # The sequence for residue r is arr[r], arr[r+d], arr[r+2d], ...
        
        for r in range(d):
            if r >= n:
                break
            # Extract the sequence for this residue
            subseq = arr[r::d]
            if len(subseq) == 0:
                continue
            
            # Compute prefix sums
            prefix_sums = np.cumsum(subseq)
            
            # Get max absolute value
            current_max = np.max(np.abs(prefix_sums))
            if current_max > max_disc:
                max_disc = current_max
                # Early termination if we hit the theoretical maximum for this n?
                # The max possible sum is n (if all are +1).
                # But we are minimizing, so we just need the true max.
                # If max_disc == n, we can't do better than n, but that's unlikely.
                # We can't early exit the whole function unless we're sure no other AP is larger,
                # but n is the absolute max. So if max_disc == n, we can return n immediately.
                if max_disc == n:
                    return n
                    
    return int(max_disc)

def construct():
    start_time = time.time()
    # Hard limit is 90s. Use 85s to be safe.
    deadline = start_time + 85.0
    
    n = 256
    rng = random.Random(42)
    
    # Initialize randomly
    signs = [rng.choice([-1, 1]) for _ in range(n)]
    
    current_score = _max_abs_ap_sum(signs)
    best_score = current_score
    best_signs = signs[:]
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.9995
    
    while time.time() < deadline:
        # Pick a random position to flip
        i = rng.randint(0, n - 1)
        signs[i] *= -1
        
        new_score = _max_abs_ap_sum(signs)
        delta = new_score - current_score
        
        if delta <= 0:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_signs = signs[:]
        else:
            # Accept with probability exp(-delta / T)
            if T > T_min:
                # Use random random() for speed
                if rng.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    # Reject: revert
                    signs[i] *= -1
            else:
                # Reject: revert
                signs[i] *= -1
        
        T *= cooling
        
        if T < T_min:
            # Reheat
            T = 10.0
            # Optionally restart from best to escape local minima
            if rng.random() < 0.1:
                signs = best_signs[:]
                current_score = best_score
                
    return best_signs
# EVOLVE-BLOCK-END
