# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n = 256
    
    def calc_disc(a):
        # a is a numpy array of shape (n,) with values -1, 1, dtype int32
        max_abs = 0
        for d in range(1, n + 1):
            for r in range(d):
                p = a[r::d]
                if len(p) == 0:
                    continue
                c = np.concatenate(([0], np.cumsum(p)))
                disc = np.max(c) - np.min(c)
                if disc > max_abs:
                    max_abs = disc
        return max_abs

    # Initial Thue-Morse
    signs_list = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    signs = np.array(signs_list, dtype=np.int32)
    
    best_signs = signs.copy()
    best_disc = calc_disc(best_signs)
    
    start_time = time.time()
    time_limit = 55.0
    rng = random.Random(12345)
    
    # Local Search with Random Restarts
    current_signs = signs.copy()
    current_disc = best_disc
    
    # We will perform a series of hill climbing steps
    # If stuck, perturb
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try flipping each bit in a random order to break symmetry
        indices = list(range(n))
        rng.shuffle(indices)
        
        for i in indices:
            old_val = current_signs[i]
            current_signs[i] = -old_val
            new_disc = calc_disc(current_signs)
            if new_disc < current_disc:
                current_disc = new_disc
                improved = True
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = current_signs.copy()
            else:
                # Revert
                current_signs[i] = old_val
        
        # If no improvement in a full pass, do a random restart or perturbation
        if not improved:
            # Perturb: flip a few random bits
            num_flips = rng.randint(1, 5)
            indices_to_flip = rng.sample(range(n), num_flips)
            for idx in indices_to_flip:
                current_signs[idx] *= -1
            current_disc = calc_disc(current_signs)
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = current_signs.copy()
            # If perturbation didn't improve best, we might want to reset current to best?
            # Or just continue from the perturbed state?
            # Usually, resetting to best is safer to avoid getting stuck in a bad basin.
            # But sometimes a bad basin leads to a better one.
            # Let's reset to best if the perturbation didn't immediately improve best.
            if current_disc >= best_disc:
                current_signs = best_signs.copy()
                current_disc = best_disc

    return best_signs.tolist()
# EVOLVE-BLOCK-END
