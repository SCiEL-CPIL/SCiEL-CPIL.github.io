import numpy as np
import time
import math

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # Precompute all arithmetic progressions as index lists (0-based)
    progs = []
    for a in range(n):
        for s in range(1, n):
            idx = list(range(a, n, s))
            if len(idx) >= 1:
                progs.append(idx)
    
    Np = len(progs)
    
    # Precompute for each position i, which APs contain it
    pos_to_progs = [[] for _ in range(n)]
    for pid, idx in enumerate(progs):
        for i in idx:
            pos_to_progs[i].append(pid)
    
    # Precompute the number of APs per position for potential weighting
    # But for standard SA, uniform selection is fine.
    
    def compute_initial_sums(signs):
        sums = np.zeros(Np, dtype=np.int32)
        for pid, idx in enumerate(progs):
            s = 0
            for i in idx:
                s += signs[i]
            sums[pid] = s
        return sums

    def get_max_abs(sums):
        return int(np.max(np.abs(sums)))

    # Initialize
    rng = np.random.RandomState(42)
    signs = rng.choice([-1, 1], size=n)
    sums = compute_initial_sums(signs)
    
    current_score = get_max_abs(sums)
    best_signs = signs.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    # We need to decide on a cooling schedule.
    # Let's use a time-based approach or step-based.
    # Given 85s budget, we can do many steps.
    # Let's estimate steps per second.
    
    # To maximize steps, we want the inner loop to be fast.
    # Flipping one sign updates ~n/s APs on average? 
    # Actually, position i is in many APs.
    # Let's just run SA until time runs out.
    
    T_start = 50.0
    T_end = 0.1
    alpha = 0.99999  # Cooling rate per step
    
    # We need to estimate how many steps we can do.
    # Let's do a quick benchmark or just run in a while loop.
    
    current_T = T_start
    step_count = 0
    
    while time.time() < deadline - 2:
        # Pick a random position to flip
        i = rng.randint(n)
        
        old_sign = signs[i]
        new_sign = -old_sign
        
        # Update sums
        for pid in pos_to_progs[i]:
            sums[pid] += 2 * new_sign
            
        new_score = get_max_abs(sums)
        
        # Acceptance criterion
        if new_score <= current_score:
            # Accept if better or equal
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_signs = signs.copy()
        else:
            # Accept with probability exp(-delta/T)
            delta = new_score - current_score
            if delta == 0:
                prob = 1.0
            else:
                # Avoid overflow/underflow issues, though delta is small integer
                prob = math.exp(-delta / current_T)
            
            if rng.random() < prob:
                current_score = new_score
            else:
                # Revert
                signs[i] = old_sign
                for pid in pos_to_progs[i]:
                    sums[pid] -= 2 * new_sign
        
        # Cool down
        current_T *= alpha
        
        # If temperature drops too low, reheat slightly or just let it die?
        # Standard SA usually doesn't reheat, but if we get stuck, maybe.
        # Let's just let it cool. If T < T_end, we can either stop or restart.
        # Restarting with current best might be good.
        if current_T < T_end:
            # Restart from best solution found so far
            signs = best_signs.copy()
            sums = compute_initial_sums(signs)
            current_score = best_score
            current_T = T_start * 0.5  # Restart with slightly lower start T
            # Or just T_start? Let's use T_start to explore again.
            current_T = T_start
        
        step_count += 1
        
        # Periodic check to ensure we don't overshoot deadline in the middle of a heavy step
        # The inner loop is small, so this is safe.
        
    return [int(x) for x in best_signs]
