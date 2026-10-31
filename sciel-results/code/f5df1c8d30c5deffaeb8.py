import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 110.0  # Safety margin vs 150s kill

def _score_fft(u):
    """
    Compute score using FFT. 
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
    """
    if not u:
        return 0.0
    if len(u) == 1:
        return 1.0
        
    m = max(u)
    if m < 1:
        return 1.0
        
    N = 1
    # We need convolution size up to 2*M
    while N < 2 * m + 1:
        N *= 2
        
    f = np.zeros(N, dtype=np.float64)
    for x in u:
        f[x] = 1.0
        
    F = np.fft.fft(f)
    
    # Sumset: Convolution of f with f
    conv_sum = np.fft.ifft(F * F).real
    # We only care about indices 0 to 2*m
    size_sum = np.sum(conv_sum[:2 * m + 1] > 0.5)
    
    # Difference set: Convolution of f with conjugate(f)
    # This gives counts of differences. 
    # auto_diff[k] corresponds to difference k (if k < N/2) or N-k (if k > N/2)
    # Actually, ifft(F * conj(F)) gives the autocorrelation.
    # C[k] = sum_x f[x] f[x+k]. 
    # For difference set, we want distinct values of a-b.
    # The autocorrelation at lag k is the number of pairs (a,b) such that a-b=k.
    # So we count non-zero entries in the autocorrelation.
    auto_corr = np.fft.ifft(F * np.conj(F)).real
    
    # The autocorrelation is symmetric. 
    # C[0] = |U|. 
    # C[k] for k>0 counts pairs with difference k.
    # C[N-k] counts pairs with difference -k.
    # We want the number of distinct differences.
    # This is the number of k in [1, m] such that C[k] > 0 OR C[N-k] > 0.
    # Since C is real and symmetric in the circular sense, and we are within [0, 2m],
    # we can just check indices 1 to m.
    
    # Note: If 2*m >= N/2, there might be wrap-around issues if m is large relative to N.
    # But we chose N >= 2m+1, so N/2 >= m + 0.5. 
    # So indices 1..m are safe and do not overlap with the negative side wrap-around 
    # which would appear at N-1, N-2, etc.
    
    pos_diffs = np.sum(auto_corr[1:m + 1] > 0.5)
    size_diff = 1 + 2 * pos_diffs  # +1 for 0, +2 for +/- differences
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / float(size_sum)
    denom = math.log(2 * m + 1)
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def _construct_golomb(n):
    """
    Construct a good initial set using a greedy Golomb ruler approach.
    This ensures a good starting point for |U-U|/span.
    """
    if n <= 1:
        return [0]
    
    # Simple greedy: start with 0, 1, then pick next largest possible
    # that doesn't create duplicate differences.
    # For small n, this is fast.
    # However, optimal Golomb rulers are known for small n.
    # Let's use a precomputed list for n up to 20 if available, or greedy.
    
    # Greedy Golomb Ruler
    u = [0, 1]
    diffs = {1}
    
    for _ in range(n - 2):
        best_candidate = None
        best_span = 0
        
        # Try adding a new element
        # Heuristic: try larger jumps first to maximize span efficiency? 
        # No, we want to maximize |U-U| for a given span.
        # Actually, for the score, we want small span (denominator) and large ratio.
        # Golomb rulers maximize |U-U| = n(n-1)+1 for a given n, but span grows as ~n^2.
        # We want to minimize span for a given n to minimize log(2M+1).
        
        # Let's just use the known optimal small rulers if n is small.
        pass
        
    # Fallback: Randomized greedy with repair
    # Since we have time, let's just use a simple randomized construction
    # that ensures distinct differences.
    
    u = [0]
    diffs = set()
    
    # Add 1
    u.append(1)
    diffs.add(1)
    
    for _ in range(n - 2):
        found = False
        # Try candidates in random order, biased towards smaller values to keep span small?
        # Or larger values to increase differences?
        # Small span is good for denominator.
        # Let's try to fit as many as possible in a small range.
        
        max_current = u[-1]
        # Try values from max_current+1 up to some bound
        # Or try to insert in gaps?
        
        # Simple approach: pick a random new max
        # But we want to minimize final max.
        
        # Let's use a simple heuristic: 
        # Start with 0. 
        # Next is 1.
        # Next is the smallest x > 1 such that all diffs x-u[i] are new.
        
        # This tends to produce small spans.
        
        x = max_current + 1
        while True:
            valid = True
            for y in u:
                d = abs(x - y)
                if d in diffs:
                    valid = False
                    break
            if valid:
                # Check all diffs
                new_diffs = [abs(x - y) for y in u]
                if len(set(new_diffs).union(diffs)) == len(diffs) + len(new_diffs):
                    u.append(x)
                    diffs.update(new_diffs)
                    found = True
                    break
            x += 1
            
            # Safety break
            if x > 10000:
                break
        
        if not found:
            # Fallback: just append a large number
            u.append(x)
            diffs.update([abs(x-y) for y in u])
            
    return sorted(u)

def construct():
    global _DEADLINE
    _DEADLINE = max(time.time(), _START + 110.0)
    
    random.seed(42)
    
    # Phase 1: Generate initial candidates
    # We want to explore different sizes n.
    # The score depends on n and M.
    # For small n, M is small.
    # Let's try n from 5 to 30.
    
    best_u = None
    best_s = -1.0
    
    # Try a few sizes
    for n in [5, 8, 10, 15, 20]:
        u = _construct_golomb(n)
        s = _score_fft(u)
        if s > best_s:
            best_s = s
            best_u = list(u)
            
    # Phase 2: Simulated Annealing
    # We will fix the size n for each run to avoid the complexity of adding/removing
    # which changes the structure significantly.
    # We will run SA for a few different n values.
    
    # Determine best n from Phase 1
    # Actually, let's just run SA on the best one found, and maybe one other.
    
    # Let's pick the top 2 distinct sizes from Phase 1
    # Re-evaluate to be sure
    candidates = []
    for n in [5, 8, 10, 12, 15, 20]:
        u = _construct_golomb(n)
        s = _score_fft(u)
        candidates.append((s, n, u))
        
    candidates.sort(key=lambda x: -x[0])
    
    # Run SA on the best candidate
    # And potentially a second one if time permits
    
    for s_init, n_init, u_init in candidates[:2]:
        if time.time() > _DEADLINE - 5.0:
            break
            
        current_u = list(u_init)
        current_s = s_init
        best_u_run = list(current_u)
        best_s_run = current_s
        
        # SA Parameters
        temp = 0.1
        cooling = 0.999
        
        # We want to keep the size fixed? 
        # If we allow size change, it might jump to a better local optimum.
        # But swapping is safer.
        # Let's allow:
        # 1. Swap two elements (keep size, keep 0)
        # 2. Move one element (keep size)
        # 3. Add/Remove (change size) - risky but might find better n
        
        # Let's stick to Swap and Move to maintain n, as changing n requires re-tuning.
        # Actually, the problem doesn't fix n.
        # But SA with fixed n is more stable.
        
        # Let's try to optimize the current n.
        
        while time.time() < _DEADLINE - 2.0:
            # Generate neighbor
            move_type = random.random()
            
            if move_type < 0.5:
                # Swap two non-zero elements
                if len(current_u) > 2:
                    non_zero_idx = [i for i in range(1, len(current_u))]
                    i1, i2 = random.sample(non_zero_idx, 2)
                    new_u = current_u.copy()
                    new_u[i1], new_u[i2] = new_u[i2], new_u[i1]
                    # Ensure sorted and unique
                    new_u = sorted(set(new_u))
                    if 0 not in new_u:
                        new_u.append(0)
                    if len(new_u) != len(current_u):
                        # This shouldn't happen with swap of distinct values
                        continue
                else:
                    continue
            elif move_type < 0.9:
                # Move one non-zero element
                if len(current_u) > 1:
                    non_zero_idx = random.randint(1, len(current_u) - 1)
                    old_val = current_u[non_zero_idx]
                    # New value: random in range [1, 2*max]
                    max_val = max(current_u)
                    new_val = random.randint(1, int(max_val * 1.5) + 10)
                    
                    new_u = current_u.copy()
                    new_u[non_zero_idx] = new_val
                    new_u = sorted(set(new_u))
                    if 0 not in new_u:
                        new_u.append(0)
                        # If we lost an element due to collision, we need to add one back?
                        # If new_val == 0 or new_val == existing, set size drops.
                        # We want to keep size constant for stability.
                        if len(new_u) < len(current_u):
                            # Revert or pick another value
                            continue
                else:
                    continue
            else:
                # Add or Remove (Change size)
                # 50% chance add, 50% remove
                if random.random() < 0.5:
                    # Add
                    max_val = max(current_u)
                    new_val = random.randint(1, int(max_val * 1.2) + 5)
                    new_u = current_u.copy()
                    new_u.append(new_val)
                    new_u = sorted(set(new_u))
                else:
                    # Remove
                    if len(current_u) > 2:
                        idx = random.randint(1, len(current_u) - 1)
                        new_u = current_u.copy()
                        new_u.pop(idx)
                    else:
                        continue
                        
            # Ensure validity
            if not new_u or 0 not in new_u:
                continue
                
            new_s = _score_fft(new_u)
            
            if new_s > current_s:
                current_u = new_u
                current_s = new_s
                if current_s > best_s_run:
                    best_s_run = current_s
                    best_u_run = list(current_u)
            else:
                # Accept with probability
                delta = current_s - new_s
                if delta < 0:
                    accept = True
                else:
                    accept = random.random() < math.exp(-delta / temp)
                    
                if accept:
                    current_u = new_u
                    current_s = new_s
            
            temp *= cooling
            if temp < 0.001:
                break
        
        if best_s_run > best_s:
            best_s = best_s_run
            best_u = best_u_run
            
    # Final check
    if best_u is None:
        return [0, 1, 4, 9]
        
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
        
    return best_u
# EVOLVE-BLOCK-END
