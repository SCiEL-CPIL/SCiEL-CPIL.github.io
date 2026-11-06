"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    N = 1000
    max_sum = 2 * N  # 2000
    
    def greedy_forward(counts, chosen, start_val=1):
        """Continue greedy from current state. Returns (chosen, counts)."""
        for x in range(start_val, N + 1):
            # Check if adding x keeps all sums <= 2
            ok = True
            # Check 2x
            if counts.get(2 * x, 0) >= 2:
                continue
            # Check x + y for all y in chosen
            for y in chosen:
                s = x + y
                if counts.get(s, 0) >= 2:
                    ok = False
                    break
            if ok:
                # Update counts
                for y in chosen:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                s = 2 * x
                counts[s] = counts.get(s, 0) + 1
                chosen.append(x)
        return chosen, counts

    def greedy_with_skip(counts, chosen, skip_prob=0.05, start_val=1):
        """Greedy with occasional skips to explore different paths."""
        for x in range(start_val, N + 1):
            # Check if adding x keeps all sums <= 2
            ok = True
            if counts.get(2 * x, 0) >= 2:
                continue
            for y in chosen:
                s = x + y
                if counts.get(s, 0) >= 2:
                    ok = False
                    break
            if ok and random.random() > skip_prob:
                for y in chosen:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                s = 2 * x
                counts[s] = counts.get(s, 0) + 1
                chosen.append(x)
        return chosen, counts

    def greedy_min_tight(counts, chosen, start_val=1):
        """Greedy that picks the candidate with minimum tight count."""
        candidates = list(range(start_val, N + 1))
        while candidates:
            if time.time() > deadline:
                break
            best_x = None
            best_tight = float('inf')
            for x in candidates:
                # Check validity
                valid = True
                tight = 0
                c2x = counts.get(2 * x, 0)
                if c2x >= 2:
                    continue
                if c2x == 1:
                    tight += 1
                for y in chosen:
                    s = x + y
                    c = counts.get(s, 0)
                    if c >= 2:
                        valid = False
                        break
                    if c == 1:
                        tight += 1
                if not valid:
                    continue
                if tight < best_tight:
                    best_tight = tight
                    best_x = x
            if best_x is None:
                break
            # Add best_x
            counts[2 * best_x] = counts.get(2 * best_x, 0) + 1
            for y in chosen:
                s = best_x + y
                counts[s] = counts.get(s, 0) + 1
            chosen.append(best_x)
            candidates.remove(best_x)
        return chosen, counts

    def local_search_add(chosen, counts, max_attempts=200):
        """Try to add elements that were skipped."""
        chosen_set = set(chosen)
        for _ in range(max_attempts):
            if time.time() > deadline:
                break
            # Find all elements not in chosen
            not_in = [x for x in range(1, N + 1) if x not in chosen_set]
            if not not_in:
                break
            # Try to add a random element
            random.shuffle(not_in)
            added = False
            for x in not_in:
                # Check if adding x is valid
                ok = True
                if counts.get(2 * x, 0) >= 2:
                    continue
                for y in chosen:
                    s = x + y
                    if counts.get(s, 0) >= 2:
                        ok = False
                        break
                if ok:
                    # Add it
                    counts[2 * x] = counts.get(2 * x, 0) + 1
                    for y in chosen:
                        s = x + y
                        counts[s] = counts.get(s, 0) + 1
                    chosen.append(x)
                    chosen_set.add(x)
                    added = True
                    break
            if not added:
                break
        return chosen, counts

    def local_search_swap(chosen, counts, max_steps=100):
        """Try to improve by swapping elements."""
        chosen_set = set(chosen)
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            # Pick a random element to remove
            idx = random.randint(0, len(chosen) - 1)
            x = chosen[idx]
            # Remove x
            for y in chosen:
                if y == x:
                    counts[2 * x] -= 1
                    if counts[2 * x] == 0:
                        del counts[2 * x]
                else:
                    s = x + y
                    counts[s] -= 1
                    if counts[s] == 0:
                        del counts[s]
            chosen_without_x = [z for z in chosen if z != x]
            chosen_set.remove(x)
            
            # Try to find a better replacement
            best_y = None
            best_tight = float('inf')
            candidates = [y for y in range(1, N + 1) if y not in chosen_set]
            random.shuffle(candidates)
            for y in candidates:
                # Check if y is valid
                valid = True
                tight = 0
                c2y = counts.get(2 * y, 0)
                if c2y >= 2:
                    continue
                if c2y == 1:
                    tight += 1
                for z in chosen_without_x:
                    s = y + z
                    c = counts.get(s, 0)
                    if c >= 2:
                        valid = False
                        break
                    if c == 1:
                        tight += 1
                if not valid:
                    continue
                if tight < best_tight:
                    best_tight = tight
                    best_y = y
            
            if best_y is not None:
                # Add best_y
                counts[2 * best_y] = counts.get(2 * best_y, 0) + 1
                for z in chosen_without_x:
                    s = best_y + z
                    counts[s] = counts.get(s, 0) + 1
                chosen[idx] = best_y
                chosen_set.add(best_y)
            else:
                # Put x back
                for y in chosen_without_x:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
                chosen[idx] = x
                chosen_set.add(x)
        return chosen, counts

    def fft_spectral_score(chosen):
        """Compute a spectral entropy-like score using FFT."""
        # Create boolean vector
        v = np.zeros(max_sum + 1, dtype=np.float64)
        for x in chosen:
            v[x] = 1.0
        
        # Compute autocorrelation via FFT
        fft_v = np.fft.rfft(v, n=2 * (max_sum + 1))
        power = np.abs(fft_v) ** 2
        # Normalize
        total_power = np.sum(power)
        if total_power == 0:
            return 0.0
        # Spectral entropy
        p = power / total_power
        p = p[p > 1e-12]
        entropy = -np.sum(p * np.log(p))
        return entropy

    # Phase 1: Run multiple greedy strategies
    best_set = []
    best_size = 0
    
    # Strategy 1: Simple forward greedy
    chosen, counts = [], defaultdict(int)
    chosen, counts = greedy_forward(counts, chosen)
    if len(chosen) > best_size:
        best_size = len(chosen)
        best_set = chosen.copy()
    
    # Strategy 2: Greedy with skip
    for _ in range(5):
        if time.time() > deadline:
            break
        chosen, counts = [], defaultdict(int)
        chosen, counts = greedy_with_skip(counts, chosen, skip_prob=0.05)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen.copy()
    
    # Strategy 3: Min-tight greedy
    if time.time() < deadline - 30:
        chosen, counts = [], defaultdict(int)
        chosen, counts = greedy_min_tight(counts, chosen)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen.copy()
    
    # Phase 2: Local search improvements on the best set
    if time.time() < deadline - 10:
        # Rebuild counts for best_set
        counts = defaultdict(int)
        for x in best_set:
            counts[2 * x] += 1
            for y in best_set:
                if y > x:
                    counts[x + y] += 1
        
        # Try adding elements
        best_set, counts = local_search_add(best_set, counts, max_attempts=100)
        
        # Try swaps
        if time.time() < deadline - 5:
            best_set, counts = local_search_swap(best_set, counts, max_steps=50)
    
    return best_set
# EVOLVE-BLOCK-END
