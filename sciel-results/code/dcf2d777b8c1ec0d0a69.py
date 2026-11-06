"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n = 256
    
    def calc_disc_np(a_np):
        """
        Calculate maximum absolute signed sum on ANY arithmetic progression.
        Input: a_np is a numpy array of int8.
        """
        max_abs = 0
        # d is the step size
        for d in range(1, n + 1):
            for r in range(d):
                seq = a_np[r::d]
                if len(seq) == 0:
                    continue
                psums = np.cumsum(seq)
                m = np.max(np.abs(psums))
                if m > max_abs:
                    max_abs = m
        return max_abs

    # Initialize with Thue-Morse
    signs = []
    for i in range(n):
        c = bin(i).count('1')
        signs.append(1 if c % 2 == 0 else -1)
    
    current_signs = np.array(signs, dtype=np.int8)
    current_disc = calc_disc_np(current_signs)
    best_signs = current_signs.copy()
    best_disc = current_disc
    
    start_time = time.time()
    time_limit = 55.0
    rng = random.Random(42)
    
    # Local Search: Greedy Hill Climbing
    def local_search(a_np, current_d):
        improved = True
        while improved:
            improved = False
            indices = list(range(n))
            rng.shuffle(indices)
            for i in indices:
                old_val = a_np[i]
                a_np[i] = -old_val
                new_d = calc_disc_np(a_np)
                if new_d < current_d:
                    current_d = new_d
                    improved = True
                else:
                    a_np[i] = old_val
        return current_d

    # First, run local search on initial state
    current_disc = local_search(current_signs, current_disc)
    if current_disc < best_disc:
        best_disc = current_disc
        best_signs = current_signs.copy()

    # Iterated Local Search
    while time.time() - start_time < time_limit:
        # Perturb: Flip k random bits
        k = rng.randint(2, 6)
        flip_indices = rng.sample(range(n), k)
        for i in flip_indices:
            current_signs[i] = -current_signs[i]
        
        # Run local search from this perturbed state
        # We need to calculate current disc before local search to pass to it?
        # local_search takes current_d. 
        # We can just calculate it once here.
        cur_d = calc_disc_np(current_signs)
        new_disc = local_search(current_signs, cur_d)
        
        if new_disc < best_disc:
            best_disc = new_disc
            best_signs = current_signs.copy()
            
    return best_signs.tolist()

# EVOLVE-BLOCK-END
