import time
import random

def _score_fast(S):
    """
    Calculate the number of distinct d > 0 such that a, a+d, a+2d are in S.
    Uses bitset operations for speed.
    """
    # Map values 1..512 to bits 0..511
    # Bit i corresponds to value i+1
    mask = 0
    for v in S:
        mask |= (1 << (v - 1))
    
    valid_d_count = 0
    # Max d: if a=1, a+2d <= 512 => 2d <= 511 => d <= 255
    for d in range(1, 256):
        # Check if there exists a such that a, a+d, a+2d are in S
        # This corresponds to bits i, i+d, i+2d being set in mask
        # (mask & (mask >> d) & (mask >> 2*d)) != 0
        if (mask & (mask >> d) & (mask >> 2 * d)) != 0:
            valid_d_count += 1
            
    return valid_d_count

def score_set(S):
    # Kept for compatibility if called elsewhere, but we will use _score_fast in construct
    return _score_fast(S)

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s buffer
    
    # Initial set: Random or Heuristic
    # A random start is often fine with SA
    best_S = [random.randint(1, 512) for _ in range(32)]
    # Ensure distinct
    while len(set(best_S)) < 32:
        best_S = [random.randint(1, 512) for _ in range(32)]
    
    best_score = _score_fast(best_S)
    
    # Simulated Annealing
    # Parameters tuned for the faster scorer
    T = 50.0
    T_min = 0.01
    cooling = 0.9995
    S = best_S[:]
    current_score = best_score
    
    # Track time per iteration to adjust if needed, but simple loop is fine
    iter_count = 0
    while time.time() < deadline and T > T_min:
        # Generate neighbor
        # 70% single replace, 30% double swap (replace 2 elements)
        if random.random() < 0.7:
            i = random.randint(0, 31)
            old_val = S[i]
            # Find a new value
            while True:
                new_val = random.randint(1, 512)
                if new_val not in S:
                    break
            S[i] = new_val
            new_score = _score_fast(S)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_S = S[:]
            else:
                S[i] = old_val
        else:
            i, j = random.sample(range(32), 2)
            old_i, old_j = S[i], S[j]
            while True:
                new_i = random.randint(1, 512)
                if new_i not in S:
                    break
            S[i] = new_i
            while True:
                new_j = random.randint(1, 512)
                if new_j not in S and new_j != new_i:
                    break
            S[j] = new_j
            new_score = _score_fast(S)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_S = S[:]
            else:
                S[i] = old_i
                S[j] = old_j
                
        T *= cooling
        iter_count += 1
        
        # Periodic restarts if stuck? 
        # Or just let it cool.
        
    # Polish phase
    # Use the fast scorer here too
    improved = True
    polish_pass = 0
    while improved and time.time() < deadline:
        improved = False
        polish_pass += 1
        for i in range(32):
            old_val = best_S[i]
            best_local_val = old_val
            best_local_score = best_score
            # Try all possible replacements
            for new_val in range(1, 513):
                if new_val in best_S:
                    continue
                best_S[i] = new_val
                s = _score_fast(best_S)
                if s > best_local_score:
                    best_local_score = s
                    best_local_val = new_val
            best_S[i] = best_local_val
            if best_local_val != old_val:
                improved = True
                best_score = best_local_score
                
    return best_S
