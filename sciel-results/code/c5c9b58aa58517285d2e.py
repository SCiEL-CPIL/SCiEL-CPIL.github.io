import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    # Target: every t in [128, 256] must be expressible as a+b with a<=b, a,b in A
    # Minimize max multiplicity over t in [128, 256]
    
    # Precompute FFT size for fast convolution
    fft_size = 512  # power of 2 >= 257
    
    def fast_convolve(v):
        """Compute convolution v*v using FFT."""
        V = np.fft.rfft(v, n=fft_size)
        conv = np.fft.irfft(V * V, n=fft_size)
        return np.round(conv).astype(np.int32)
    
    def compute_unordered(conv, A_set):
        """Compute unordered representation counts from convolution and set."""
        target = np.zeros(129, dtype=np.int32)
        for t in range(128, 257):
            ordered = int(conv[t])
            diag = 1 if (t % 2 == 0 and (t // 2) in A_set) else 0
            unordered = (ordered + diag) // 2
            target[t - 128] = unordered
        return target
    
    def evaluate(A_set):
        """Return (max_mult, covered, target_counts) for the given set A."""
        v = np.zeros(129, dtype=np.float64)
        for x in A_set:
            v[x] = 1.0
        conv = fast_convolve(v)
        target = compute_unordered(conv, A_set)
        max_mult = int(target.max()) if len(target) > 0 else 0
        covered = bool(target.min() >= 1) if len(target) > 0 else False
        return max_mult, covered, target
    
    def build_initial_set():
        """Build a valid covering set using greedy approach."""
        A = set()
        A.add(0)
        A.add(128)
        
        # Greedy: for uncovered t, add most useful element
        for t in range(128, 257):
            covered_t = False
            for a in A:
                b = t - a
                if b in A and b >= a:
                    covered_t = True
                    break
            if not covered_t:
                # Try all possible a values
                candidates = []
                for a in range(max(0, t - 128), min(128, t // 2) + 1):
                    if a not in A:
                        candidates.append(a)
                if candidates:
                    best_a = None
                    best_count = -1
                    for a in candidates:
                        count = 0
                        for t2 in range(128, 257):
                            b2 = t2 - a
                            if b2 in A and b2 >= a:
                                count += 1
                        if count > best_count:
                            best_count = count
                            best_a = a
                    if best_a is not None:
                        A.add(best_a)
        
        return A
    
    A0 = build_initial_set()
    score0, covered0, target0 = evaluate(A0)
    
    if not covered0:
        A0 = set(range(0, 129, 2))
        score0, covered0, target0 = evaluate(A0)
        if not covered0:
            A0 = set(range(0, 129))
            score0, covered0, target0 = evaluate(A0)
    
    best_A = set(A0)
    best_score = score0
    best_covered = covered0
    best_target = target0.copy()
    
    # Simulated annealing with incremental updates
    current_A = set(A0)
    current_score, current_covered, current_target = evaluate(current_A)
    
    T = 20.0
    T_min = 0.01
    cooling = 0.9998
    
    # Track which targets have high multiplicity
    def get_hot_targets(target, threshold=None):
        """Get targets with multiplicity >= threshold or above median."""
        if threshold is None:
            threshold = int(np.median(target)) + 1
        return [i + 128 for i in range(len(target)) if target[i] >= threshold]
    
    while time.time() < deadline - 3 and T > T_min:
        # Generate a move
        r = random.random()
        
        if r < 0.25 and len(current_A) > 1:
            # Remove: prefer elements that contribute to hot targets
            hot = get_hot_targets(current_target)
            if hot:
                # Find elements that contribute most to hot targets
                contrib = {}
                for x in current_A:
                    count = 0
                    for t in hot:
                        y = t - x
                        if y in current_A and y >= x:
                            count += 1
                    contrib[x] = count
                # Remove element with highest contribution to hot targets
                max_contrib = max(contrib.values()) if contrib else 0
                if max_contrib > 0:
                    candidates = [x for x, c in contrib.items() if c == max_contrib]
                    x = random.choice(candidates)
                else:
                    x = random.choice(sorted(current_A))
            else:
                x = random.choice(sorted(current_A))
            new_A = current_A - {x}
        elif r < 0.50:
            # Add: prefer elements that help cover low-multiplicity targets
            low = [i + 128 for i in range(len(current_target)) if current_target[i] <= 1]
            if low:
                # Find element that covers most low targets
                contrib = {}
                for x in range(0, 129):
                    if x in current_A:
                        continue
                    count = 0
                    for t in low:
                        y = t - x
                        if y in current_A and y >= x:
                            count += 1
                    if count > 0:
                        contrib[x] = count
                if contrib:
                    max_contrib = max(contrib.values())
                    candidates = [x for x, c in contrib.items() if c == max_contrib]
                    x = random.choice(candidates)
                else:
                    x = random.randint(0, 128)
            else:
                x = random.randint(0, 128)
            if x in current_A:
                continue
            new_A = current_A | {x}
        elif r < 0.75 and len(current_A) > 1:
            # Swap
            hot = get_hot_targets(current_target)
            if hot:
                contrib = {}
                for x in current_A:
                    count = 0
                    for t in hot:
                        y = t - x
                        if y in current_A and y >= x:
                            count += 1
                    contrib[x] = count
                max_contrib = max(contrib.values()) if contrib else 0
                if max_contrib > 0:
                    candidates = [x for x, c in contrib.items() if c == max_contrib]
                    x_remove = random.choice(candidates)
                else:
                    x_remove = random.choice(sorted(current_A))
            else:
                x_remove = random.choice(sorted(current_A))
            
            low = [i + 128 for i in range(len(current_target)) if current_target[i] <= 1]
            if low:
                contrib_add = {}
                for x in range(0, 129):
                    if x in current_A:
                        continue
                    count = 0
                    for t in low:
                        y = t - x
                        if y in current_A and y >= x:
                            count += 1
                    if count > 0:
                        contrib_add[x] = count
                if contrib_add:
                    max_contrib_add = max(contrib_add.values())
                    candidates_add = [x for x, c in contrib_add.items() if c == max_contrib_add]
                    x_add = random.choice(candidates_add)
                else:
                    x_add = random.randint(0, 128)
                    if x_add in current_A:
                        continue
            else:
                x_add = random.randint(0, 128)
                if x_add in current_A:
                    continue
            
            new_A = current_A - {x_remove} | {x_add}
        else:
            # Double swap
            if len(current_A) < 2:
                continue
            x1, x2 = random.sample(sorted(current_A), 2)
            x3 = random.randint(0, 128)
            x4 = random.randint(0, 128)
            if x3 in current_A or x4 in current_A or x3 == x4:
                continue
            new_A = current_A - {x1, x2} | {x3, x4}
        
        new_score, new_covered, new_target = evaluate(new_A)
        
        # Cost function with heavy penalty for uncovered
        if new_covered:
            new_cost = new_score
        else:
            new_cost = new_score + 10000
        
        if current_covered:
            current_cost = current_score
        else:
            current_cost = current_score + 10000
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_A = new_A
            current_score = new_score
            current_covered = new_covered
            current_target = new_target
            
            if current_covered and (not best_covered or current_score < best_score):
                best_A = set(current_A)
                best_score = current_score
                best_covered = True
                best_target = current_target.copy()
        
        T *= cooling
        
        # Periodic reheating
        if int((1.0 - T / 20.0) * 100) % 20 == 0 and T > T_min * 10:
            T = max(T, 5.0)
    
    # Final polish: try removing elements one by one
    if best_covered and time.time() < deadline - 1:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for x in sorted(best_A):
                if time.time() > deadline - 1:
                    break
                new_A = best_A - {x}
                s, c, t = evaluate(new_A)
                if c and s <= best_score:
                    best_A = new_A
                    best_score = s
                    best_target = t
                    improved = True
    
    # Ensure validity
    if not best_covered:
        best_A = set(range(0, 129))
    
    return sorted(best_A)
