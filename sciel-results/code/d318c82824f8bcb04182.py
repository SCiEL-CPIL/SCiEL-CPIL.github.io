import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit

    def is_valid(seq):
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    def count_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    # Pre-compute prime factor counts for all numbers in [2, 128]
    pf_counts = {x: count_prime_factors(x) for x in range(2, 129)}

    # 1. Greedy Initialization with heuristic: prefer numbers with fewer prime factors
    seq = []
    # Sort candidates by (prime_factor_count, value) to prefer "simpler" numbers
    candidates = sorted(range(2, 129), key=lambda x: (pf_counts[x], x))
    for x in candidates:
        candidate = seq + [x]
        # Ensure strictly increasing
        if seq and candidate[-1] <= candidate[-2]:
            continue
        if is_valid(candidate):
            seq = candidate
    
    # Re-sort seq to ensure strictly increasing order for valid sequence
    seq.sort()
    
    best_seq = list(seq)
    best_len = len(seq)

    # 2. Fixed-Point Insertion
    # Try to insert any excluded number at any position
    changed = True
    while changed and time.time() < DEADLINE:
        changed = False
        excluded = [x for x in range(2, 129) if x not in seq]
        # Sort excluded by heuristic (fewer prime factors first) to find valid insertions faster
        excluded.sort(key=lambda x: (pf_counts[x], x))
        for x in excluded:
            # Try all possible positions
            for pos in range(len(seq) + 1):
                # Check if position is valid (strictly increasing)
                if pos > 0 and seq[pos-1] >= x:
                    continue
                if pos < len(seq) and seq[pos] <= x:
                    continue
                
                candidate = seq[:pos] + [x] + seq[pos:]
                if is_valid(candidate):
                    seq = candidate
                    changed = True
                    if len(seq) > best_len:
                        best_len = len(seq)
                        best_seq = list(seq)
                    break # Restart with new excluded set

    # 3. Local Search: 1-out/1-in swaps
    # Try to remove one element and insert another to potentially unlock longer sequences
    if time.time() < DEADLINE - 10.0:
        seq_set = set(seq)
        excluded = [x for x in range(2, 129) if x not in seq_set]
        excluded.sort(key=lambda x: (pf_counts[x], x))
        
        # Limit number of swap attempts to avoid timeout
        max_swap_attempts = 500
        attempts = 0
        
        for remove_idx in range(len(seq)):
            if attempts >= max_swap_attempts or time.time() > DEADLINE - 5.0:
                break
            for add_x in excluded:
                if attempts >= max_swap_attempts or time.time() > DEADLINE - 5.0:
                    break
                attempts += 1
                
                # Create trial sequence: remove seq[remove_idx], add add_x
                # We need to find a valid position for add_x in the remaining sequence
                trial_base = seq[:remove_idx] + seq[remove_idx+1:]
                
                # Find valid positions for add_x in trial_base
                valid_positions = []
                for pos in range(len(trial_base) + 1):
                    if pos > 0 and trial_base[pos-1] >= add_x:
                        continue
                    if pos < len(trial_base) and trial_base[pos] <= add_x:
                        continue
                    valid_positions.append(pos)
                
                # Try each valid position
                for pos in valid_positions:
                    trial = trial_base[:pos] + [add_x] + trial_base[pos:]
                    if is_valid(trial):
                        # Check if this is an improvement (should be same length, but maybe better structure)
                        # Actually, 1-out/1-in keeps length same. 
                        # The benefit is if it allows future insertions.
                        # So we accept it if it's valid, and then re-run insertion.
                        if len(trial) == len(seq):
                            # Only accept if it might lead to better insertion opportunities
                            # Heuristic: accept if the new set of products is "different" or just accept randomly
                            # To avoid cycles, we can check if this specific (remove_idx, add_x, pos) was already tried
                            # For now, just accept and restart insertion
                            seq = trial
                            seq_set = set(seq)
                            excluded = [x for x in range(2, 129) if x not in seq_set]
                            excluded.sort(key=lambda x: (pf_counts[x], x))
                            
                            # Re-run fixed-point insertion
                            changed = True
                            while changed and time.time() < DEADLINE - 5.0:
                                changed = False
                                for x in excluded:
                                    for pos in range(len(seq) + 1):
                                        if pos > 0 and seq[pos-1] >= x:
                                            continue
                                        if pos < len(seq) and seq[pos] <= x:
                                            continue
                                        candidate = seq[:pos] + [x] + seq[pos:]
                                        if is_valid(candidate):
                                            seq = candidate
                                            changed = True
                                            if len(seq) > best_len:
                                                best_len = len(seq)
                                                best_seq = list(seq)
                                            break
                                    if changed:
                                        break
                            break # Break out of add_x loop after one successful swap+insertion
                if changed:
                    break # Break out of remove_idx loop
        # Note: The logic above is a bit complex. Let's simplify by just trying swaps and then doing one more insertion pass.
    
    return best_seq
