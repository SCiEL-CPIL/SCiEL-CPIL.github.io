import time
import math
import random
import numpy as np

def compute_score(U):
    """
    Computes the lower bound exponent: 1 + log(|U-U|/|U+U|)/log(2 max U + 1)
    """
    if len(U) < 2:
        return 0.0
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + U
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    n = len(U)
    # Use sets for sum and difference
    sum_set = set()
    diff_set = set()
    
    # For efficiency with moderate n, nested loops
    for i in range(n):
        a = U[i]
        for j in range(n):
            b = U[j]
            sum_set.add(a + b)
            diff_set.add(a - b)
    
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0.0
    return 1.0 + log_ratio / log_denom

def mian_chowla(n_terms):
    """
    Generate the Mian-Chowla sequence (Sidon sequence) with n_terms elements.
    Starts with 0, 1, and greedily adds the smallest integer that maintains
    the Sidon property (all pairwise sums distinct).
    """
    if n_terms <= 0:
        return []
    seq = [0]
    if n_terms == 1:
        return seq
    seq.append(1)
    if n_terms == 2:
        return seq
    
    # Maintain set of all pairwise sums
    sums = set()
    for i in range(len(seq)):
        for j in range(len(seq)):
            sums.add(seq[i] + seq[j])
    
    while len(seq) < n_terms:
        candidate = seq[-1] + 1
        while True:
            # Check if adding candidate preserves Sidon property
            valid = True
            for i in range(len(seq)):
                new_sum = seq[i] + candidate
                if new_sum in sums:
                    valid = False
                    break
            if valid:
                # Add candidate
                for i in range(len(seq)):
                    sums.add(seq[i] + candidate)
                sums.add(candidate + candidate)
                seq.append(candidate)
                break
            candidate += 1
    return seq

def simulated_annealing(U_init, max_iter=5000, T0=10.0, cooling=0.9995, T_min=0.01):
    """
    Simulated annealing to optimize the set U.
    Moves: add element, remove element, swap element, scale last element.
    """
    U = sorted(set(U_init))
    if 0 not in U:
        U = [0] + U
    
    current_score = compute_score(U)
    best_U = U[:]
    best_score = current_score
    
    T = T0
    rng = random.Random(42)
    
    for it in range(max_iter):
        if T < T_min:
            break
        
        U_curr = U[:]
        max_val = max(U_curr) if U_curr else 1
        
        # Choose a random move
        move = rng.choice(['add', 'remove', 'swap', 'scale'])
        
        if move == 'add' and len(U_curr) < 50:
            # Add a new element
            # Try values in various ranges
            if rng.random() < 0.3:
                # Small value
                new_val = rng.randint(1, max(2, max_val // 10))
            elif rng.random() < 0.6:
                # Medium value
                new_val = rng.randint(max(1, max_val // 4), max_val)
            else:
                # Large value
                new_val = rng.randint(max_val, max_val * 2 + 10)
            
            if new_val not in U_curr:
                U_curr.append(new_val)
                U_curr = sorted(set(U_curr))
        
        elif move == 'remove' and len(U_curr) > 2:
            # Remove a non-zero element
            removable = [i for i, x in enumerate(U_curr) if x != 0]
            if removable:
                idx = rng.choice(removable)
                U_curr.pop(idx)
        
        elif move == 'swap' and len(U_curr) > 2:
            # Replace a non-zero element with a new value
            removable = [i for i, x in enumerate(U_curr) if x != 0]
            if removable:
                idx = rng.choice(removable)
                old_val = U_curr[idx]
                # New value near old or in different range
                if rng.random() < 0.5:
                    new_val = max(1, old_val + rng.randint(-max_val // 4, max_val // 4))
                else:
                    new_val = rng.randint(1, max_val * 2 + 10)
                if new_val != old_val:
                    U_curr[idx] = new_val
                    U_curr = sorted(set(U_curr))
                    if 0 not in U_curr:
                        U_curr = [0] + U_curr
        
        elif move == 'scale':
            # Scale the largest element
            if U_curr:
                max_idx = len(U_curr) - 1
                if U_curr[max_idx] != 0:
                    scale_factor = rng.uniform(0.5, 2.0)
                    new_max = max(1, int(U_curr[max_idx] * scale_factor))
                    if new_max != U_curr[max_idx]:
                        U_curr[max_idx] = new_max
                        U_curr = sorted(set(U_curr))
        
        # Ensure 0 is in the set
        if 0 not in U_curr:
            U_curr = [0] + U_curr
        U_curr = sorted(set(U_curr))
        
        if len(U_curr) < 2:
            continue
        
        new_score = compute_score(U_curr)
        
        # Accept or reject
        delta = new_score - current_score
        if delta > 0 or rng.random() < math.exp(delta / T):
            U = U_curr
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_U = U[:]
        
        T *= cooling
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    best_U = [0, 1]
    best_score = compute_score([0, 1])
    
    # Generate multiple Sidon sequence seeds of different sizes
    seeds = []
    for n in [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
        if time.time() > deadline - 10:
            break
        try:
            seq = mian_chowla(n)
            if len(seq) >= 2:
                seeds.append(seq[:])
        except Exception:
            pass
    
    # Also add some geometric sequence seeds
    for base in [2, 3]:
        for k in range(3, 12):
            if time.time() > deadline - 10:
                break
            U = [base**i for i in range(k)]
            if 0 not in U:
                U = [0] + U
            seeds.append(U[:])
    
    # Run simulated annealing from each seed
    rng = random.Random(123)
    for seed_idx, seed in enumerate(seeds):
        if time.time() > deadline - 5:
            break
        
        # Vary the annealing parameters slightly for diversity
        T0 = rng.uniform(5.0, 20.0)
        cooling = rng.uniform(0.999, 0.9998)
        max_iter = rng.randint(3000, 8000)
        
        try:
            U_opt, score_opt = simulated_annealing(
                seed, max_iter=max_iter, T0=T0, cooling=cooling, T_min=0.01
            )
            if score_opt > best_score:
                best_score = score_opt
                best_U = U_opt[:]
        except Exception:
            continue
    
    # Final local refinement: try small perturbations on the best solution
    if time.time() < deadline - 2 and len(best_U) >= 2:
        current = best_U[:]
        current_score = best_score
        max_val = max(current)
        
        for _ in range(200):
            if time.time() > deadline - 1:
                break
            
            U_try = current[:]
            move = rng.choice(['add', 'remove', 'swap'])
            
            if move == 'add' and len(U_try) < 40:
                new_val = rng.randint(1, max_val * 2 + 10)
                if new_val not in U_try:
                    U_try.append(new_val)
                    U_try = sorted(set(U_try))
            elif move == 'remove' and len(U_try) > 2:
                removable = [i for i, x in enumerate(U_try) if x != 0]
                if removable:
                    idx = rng.choice(removable)
                    U_try.pop(idx)
            elif move == 'swap' and len(U_try) > 2:
                removable = [i for i, x in enumerate(U_try) if x != 0]
                if removable:
                    idx = rng.choice(removable)
                    old_val = U_try[idx]
                    new_val = max(1, old_val + rng.randint(-max_val // 5, max_val // 5))
                    if new_val != old_val:
                        U_try[idx] = new_val
                        U_try = sorted(set(U_try))
            
            if 0 not in U_try:
                U_try = [0] + U_try
            U_try = sorted(set(U_try))
            
            if len(U_try) < 2:
                continue
            
            s = compute_score(U_try)
            if s > current_score:
                current_score = s
                current = U_try[:]
                best_score = s
                best_U = U_try[:]
    
    return best_U

if __name__ == "__main__":
    result = construct()
    print(f"Best set: {result}")
    print(f"Score: {compute_score(result)}")
