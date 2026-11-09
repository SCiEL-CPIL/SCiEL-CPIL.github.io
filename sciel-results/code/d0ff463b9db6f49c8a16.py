import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 128
    target_min = 128
    target_max = 256
    targets = np.arange(target_min, target_max + 1)
    
    # Use numpy for fast sumset calculation
    def get_counts(A):
        # A is a set or list
        if not A:
            return np.zeros(2 * n + 1, dtype=int)
        # Create indicator vector
        vec = np.zeros(n + 1, dtype=int)
        for a in A:
            vec[a] = 1
        # Convolution
        conv = np.convolve(vec, vec, mode='full')
        return conv
        
    def get_score(A):
        counts = get_counts(A)
        # counts[s] is number of pairs (a,b) with a+b=s.
        # Note: np.convolve counts ordered pairs (a,b) and (b,a) separately if a!=b.
        # The task says "unordered pairs ... permit equal summands".
        # So we need to adjust the counts.
        # If a != b, count in conv is 2 * (unordered count).
        # If a == b, count in conv is 1 * (unordered count).
        # So unordered_count[s] = (conv[s] + (1 if s is even and s/2 in A else 0)) / 2 ?
        # Actually, simpler:
        # Let C_ordered[s] be the number of ordered pairs.
        # C_unordered[s] = (C_ordered[s] + I(s/2 in A and s even)) / 2.
        
        c_unordered = np.zeros_like(counts, dtype=float)
        for s in range(target_min, target_max + 1):
            ordered = counts[s]
            # Check if s/2 is in A
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= n and (half in A):
                    # The pair (half, half) is counted once in ordered.
                    # In unordered, it's 1.
                    # In ordered, it's 1.
                    # Other pairs (a,b) with a!=b are counted 2 times in ordered, 1 in unordered.
                    # So: 2 * C_unordered[s] - I(half in A) = C_ordered[s]
                    # C_unordered[s] = (C_ordered[s] + I(half in A)) / 2
                    c_unordered[s] = (ordered + 1) / 2.0
                else:
                    c_unordered[s] = ordered / 2.0
            else:
                c_unordered[s] = ordered / 2.0
                
        # Extract counts for targets
        target_counts = c_unordered[target_min:target_max + 1]
        
        missing = np.sum(target_counts == 0)
        max_mult = np.max(target_counts) if len(target_counts) > 0 else 0
        
        # Score
        return (missing * 10000 + max_mult, missing, max_mult)

    # Initialize
    best_A = set()
    best_score = float('inf')
    
    # Try seed
    seed = set(range(64, 129))
    seed_score, seed_missing, seed_max = get_score(seed)
    best_score = seed_score
    best_A = seed
    
    # Local Search
    current_A = set(seed)
    current_score, current_missing, current_max = get_score(current_A)
    
    T = 5.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < deadline:
        # Random move
        r = random.random()
        if r < 0.3:
            # Add
            if len(current_A) < n + 1:
                candidates = [x for x in range(n + 1) if x not in current_A]
                if candidates:
                    new_A = current_A | {random.choice(candidates)}
                    new_score, new_missing, new_max = get_score(new_A)
                    delta = new_score - current_score
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        current_A = new_A
                        current_score = new_score
                        current_missing = new_missing
                        current_max = new_max
                        if current_score < best_score:
                            best_score = current_score
                            best_A = set(current_A)
        elif r < 0.6:
            # Remove
            if len(current_A) > 1:
                new_A = set(current_A)
                new_A.remove(random.choice(list(current_A)))
                new_score, new_missing, new_max = get_score(new_A)
                delta = new_score - current_score
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_A = new_A
                    current_score = new_score
                    current_missing = new_missing
                    current_max = new_max
                    if current_score < best_score:
                        best_score = current_score
                        best_A = set(current_A)
        else:
            # Swap
            if len(current_A) > 1:
                in_el = random.choice(list(current_A))
                candidates = [x for x in range(n + 1) if x not in current_A]
                if candidates:
                    out_el = random.choice(candidates)
                    new_A = (current_A - {in_el}) | {out_el}
                    new_score, new_missing, new_max = get_score(new_A)
                    delta = new_score - current_score
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        current_A = new_A
                        current_score = new_score
                        current_missing = new_missing
                        current_max = new_max
                        if current_score < best_score:
                            best_score = current_score
                            best_A = set(current_A)
                            
        T *= cooling
        if T < T_min:
            T = T_min

    # Final check
    if not best_A:
        return list(range(64, 129))
        
    bs, bm, bmm = get_score(best_A)
    if bm > 0:
        return list(range(64, 129))
        
    return list(best_A)
