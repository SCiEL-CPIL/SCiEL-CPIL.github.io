"""Rewritten program: deterministic multi-start local search with vectorized numpy pair-sum updates."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)
    
    # Precompute pair sums for all unordered pairs (i, j) where i <= j, i,j in [0, 128]
    # We'll use a different representation: select a subset A of [0, 128]
    # The pair sum matrix for elements a, b in A gives counts.
    
    # Key insight: represent A as a binary vector of length 129.
    # The multiplicity vector for target sums can be computed via convolution-like operation.
    # For unordered pairs with equal summands allowed:
    # count[s] = sum over a in A, b in A, a <= b, a+b=s of 1
    # This equals: for a < b: 1 if both in A and a+b=s
    #              for a = b: 1 if a in A and 2a=s
    # We can compute this as:
    # Let v be the indicator vector of A (length 129, indices 0..128)
    # The ordered pair sum convolution: c[s] = sum_{a=0}^{128} v[a] * v[s-a] for s-a in [0,128]
    # For unordered: count[s] = (c[s] + v[s//2] * (1 if s%2==0 and s//2 in [0,128] else 0)) / 2
    # Wait, let me think more carefully.
    # Ordered pairs: (a,b) and (b,a) are distinct if a != b.
    # c[s] = sum_{a} v[a] * v[s-a] counts ordered pairs.
    # If a != b and a+b=s, both (a,b) and (b,a) contribute, so 2 per unordered pair.
    # If a = b = s/2, only (a,a) contributes, so 1.
    # So unordered count[s] = (c[s] + v[s//2] * (1 if s even and 0 <= s//2 <= 128 else 0)) / 2
    
    def compute_multiplicities(v):
        """v is a boolean array of length 129. Returns multiplicity array for sums 0..256."""
        # Use numpy convolution
        # c[s] = sum_{a} v[a] * v[s-a]
        # This is the convolution of v with v, but we need to be careful about bounds.
        # v has length 129 (indices 0..128)
        # The convolution will have length 257 (indices 0..256)
        c = np.convolve(v.astype(float), v.astype(float), mode='full')
        # c[s] for s in 0..256
        # Now adjust for unordered pairs
        counts = c.copy()
        # For even s, the diagonal term v[s//2]^2 = v[s//2] (since v is 0/1)
        # In the ordered count c[s], the diagonal contributes 1 (v[s//2]*v[s//2])
        # In unordered, it should contribute 1.
        # For off-diagonal, ordered gives 2, unordered gives 1.
        # So: unordered[s] = (c[s] + diag[s]) / 2 where diag[s] = v[s//2] if s even else 0
        # Actually: c[s] = 2 * (off-diag count) + diag_count
        # unordered[s] = off-diag count + diag_count = (c[s] - diag_count + 2*diag_count) / 2 = (c[s] + diag_count) / 2
        diag = np.zeros(257)
        for s in range(0, 257, 2):
            a = s // 2
            if 0 <= a <= 128:
                diag[s] = v[a]
        counts = (c + diag) / 2.0
        return counts
    
    def peak_and_coverage(v):
        """Return (peak multiplicity in target, coverage boolean)."""
        counts = compute_multiplicities(v)
        target_counts = counts[min_sum:max_sum+1]
        peak = int(np.max(target_counts))
        covered = np.all(target_counts >= 1)
        return peak, bool(covered), target_counts
    
    def local_search(v, max_iters=2000, time_limit=5.0):
        """Perform local search starting from binary vector v. Returns best (v, peak)."""
        t0 = time.time()
        current_v = v.copy()
        current_peak, current_cov, current_tc = peak_and_coverage(current_v)
        best_v = current_v.copy()
        best_peak = current_peak if current_cov else float('inf')
        best_cov = current_cov
        
        # If not covered, we need to add elements first
        if not current_cov:
            # Greedily add elements to cover
            while not current_cov:
                if time.time() > time_limit + t0:
                    break
                # Find element that covers most uncovered sums
                uncovered_sums = np.where(current_tc == 0)[0] + min_sum
                best_elem = -1
                best_gain = -1
                for x in range(129):
                    if current_v[x]:
                        continue
                    new_v = current_v.copy()
                    new_v[x] = True
                    _, cov, tc = peak_and_coverage(new_v)
                    gain = np.sum(tc >= 1) - np.sum(current_tc >= 1)
                    if cov and gain > best_gain:
                        best_gain = gain
                        best_elem = x
                    elif gain > best_gain:
                        best_gain = gain
                        best_elem = x
                if best_elem == -1:
                    break
                current_v[best_elem] = True
                current_peak, current_cov, current_tc = peak_and_coverage(current_v)
                if current_cov and current_peak < best_peak:
                    best_peak = current_peak
                    best_v = current_v.copy()
                    best_cov = True
        
        # Now do local search with swaps and additions/removals
        iters = 0
        while iters < max_iters:
            if time.time() > time_limit + t0:
                break
            iters += 1
            
            # Try random swaps: remove one element, add another
            # Or try single additions/removals
            move_type = random.choice(['swap', 'add', 'remove'])
            
            if move_type == 'swap':
                if np.sum(current_v) < 2:
                    continue
                # Pick random element to remove
                elems_in = np.where(current_v)[0]
                if len(elems_in) < 2:
                    continue
                rem_elem = random.choice(elems_in)
                # Pick random element to add
                elems_out = np.where(~current_v)[0]
                if len(elems_out) == 0:
                    continue
                add_elem = random.choice(elems_out)
                
                new_v = current_v.copy()
                new_v[rem_elem] = False
                new_v[add_elem] = True
                new_peak, new_cov, new_tc = peak_and_coverage(new_v)
                
                if new_cov and new_peak < best_peak:
                    best_peak = new_peak
                    best_v = new_v.copy()
                    best_cov = True
                if new_cov and new_peak <= current_peak:
                    current_v = new_v
                    current_peak = new_peak
                    current_cov = new_cov
                    current_tc = new_tc
                elif new_cov and random.random() < 0.1:
                    # Accept worse moves with some probability
                    current_v = new_v
                    current_peak = new_peak
                    current_cov = new_cov
                    current_tc = new_tc
                    
            elif move_type == 'add':
                elems_out = np.where(~current_v)[0]
                if len(elems_out) == 0:
                    continue
                add_elem = random.choice(elems_out)
                new_v = current_v.copy()
                new_v[add_elem] = True
                new_peak, new_cov, new_tc = peak_and_coverage(new_v)
                
                if new_cov and new_peak < best_peak:
                    best_peak = new_peak
                    best_v = new_v.copy()
                    best_cov = True
                if new_cov and new_peak <= current_peak:
                    current_v = new_v
                    current_peak = new_peak
                    current_cov = new_cov
                    current_tc = new_tc
                    
            elif move_type == 'remove':
                elems_in = np.where(current_v)[0]
                if len(elems_in) < 2:
                    continue
                rem_elem = random.choice(elems_in)
                new_v = current_v.copy()
                new_v[rem_elem] = False
                new_peak, new_cov, new_tc = peak_and_coverage(new_v)
                
                if new_cov and new_peak < best_peak:
                    best_peak = new_peak
                    best_v = new_v.copy()
                    best_cov = True
                if new_cov and new_peak <= current_peak:
                    current_v = new_v
                    current_peak = new_peak
                    current_cov = new_cov
                    current_tc = new_tc
        
        return best_v, best_peak if best_cov else float('inf')
    
    best_v = None
    best_peak = float('inf')
    
    # Multi-start local search with different initial configurations
    random.seed(42)
    
    # Start 1: Random dense set
    for trial in range(15):
        if time.time() > deadline - 2.0:
            break
        
        # Different initialization strategies
        if trial == 0:
            # Contiguous block in the middle
            v = np.zeros(129, dtype=bool)
            v[64:97] = True  # 33 elements
        elif trial == 1:
            # Random sparse
            v = np.zeros(129, dtype=bool)
            v[random.sample(range(129), 20)] = True
        elif trial == 2:
            # Arithmetic progression
            v = np.zeros(129, dtype=bool)
            for x in range(0, 129, 4):
                v[x] = True
        elif trial == 3:
            # Two intervals
            v = np.zeros(129, dtype=bool)
            v[32:64] = True
            v[96:129] = True
        elif trial == 4:
            # Random with higher density
            v = np.zeros(129, dtype=bool)
            v[random.sample(range(129), 35)] = True
        elif trial == 5:
            # Geometric-like
            v = np.zeros(129, dtype=bool)
            for x in [1, 2, 4, 8, 16, 32, 64, 96, 112, 120, 124, 126, 127, 128]:
                if x <= 128:
                    v[x] = True
            v[random.sample(range(129), 15)] = True
        elif trial == 6:
            # Symmetric around 64
            v = np.zeros(129, dtype=bool)
            for offset in range(0, 30, 2):
                if 64 - offset >= 0:
                    v[64 - offset] = True
                if 64 + offset <= 128:
                    v[64 + offset] = True
        elif trial == 7:
            # Random
            v = np.zeros(129, dtype=bool)
            v[random.sample(range(129), 25)] = True
        elif trial == 8:
            # Dense random
            v = np.zeros(129, dtype=bool)
            v[random.sample(range(129), 40)] = True
        elif trial == 9:
            # Fibonacci-like
            v = np.zeros(129, dtype=bool)
            fibs = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
            for f in fibs:
                if f <= 128:
                    v[f] = True
            v[random.sample(range(129), 20)] = True
        elif trial == 10:
            # Powers of 2 plus random
            v = np.zeros(129, dtype=bool)
            for p in range(0, 8):
                v[2**p] = True
            v[random.sample(range(129), 25)] = True
        elif trial == 11:
            # Random with bias toward middle
            v = np.zeros(129, dtype=bool)
            weights = [1 + abs(64 - x) / 64.0 for x in range(129)]
            weights = [w ** 2 for w in weights]
            total = sum(weights)
            weights = [w / total for w in weights]
            selected = random.choices(range(129), weights=weights, k=30)
            for x in set(selected):
                v[x] = True
        elif trial == 12:
            # Random with bias toward edges
            v = np.zeros(129, dtype=bool)
            weights = [1 + (abs(x - 64) / 64.0) ** 2 for x in range(129)]
            total = sum(weights)
            weights = [w / total for w in weights]
            selected = random.choices(range(129), weights=weights, k=30)
            for x in set(selected):
                v[x] = True
        elif trial == 13:
            # Alternating
            v = np.zeros(129, dtype=bool)
            for x in range(0, 129, 3):
                v[x] = True
        elif trial == 14:
            # Random
            v = np.zeros(129, dtype=bool)
            v[random.sample(range(129), 30)] = True
        
        v, peak = local_search(v, max_iters=1500, time_limit=5.0)
        
        if peak < best_peak:
            best_peak = peak
            best_v = v.copy()
    
    if best_v is None:
        # Fallback
        best_v = np.zeros(129, dtype=bool)
        best_v[64:97] = True
    
    # Convert to list
    A = [i for i in range(129) if best_v[i]]
    
    # Final verification
    counts = compute_multiplicities(best_v)
    target_counts = counts[min_sum:max_sum+1]
    if not np.all(target_counts >= 1):
        # Fallback to a known working set
        A = list(range(64, 129))
    
    return A
# EVOLVE-BLOCK-END
