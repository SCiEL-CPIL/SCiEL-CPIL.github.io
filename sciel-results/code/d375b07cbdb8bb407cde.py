"""Solve additive base problem: find A ⊆ [0,128] covering [128,256] by pair sums, minimizing peak multiplicity."""
import time
import random
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    n = 128
    target_lo, target_hi = 128, 256  # inclusive
    target_range = target_hi - target_lo + 1  # 129 values
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    def compute_multiplicities(A_set):
        """Compute multiplicity for each target using FFT convolution."""
        arr = np.zeros(2 * n + 1)
        for a in A_set:
            arr[a] = 1.0
        # Convolve with itself
        conv = np.fft.fft(arr, n=4 * n + 1)
        conv = conv * conv
        conv = np.fft.ifft(conv).real
        # Extract target range
        muls = conv[target_lo:target_hi + 1].round().astype(int)
        return muls
    
    def peak_mult(A_set):
        muls = compute_multiplicities(A_set)
        return int(muls.max()) if len(muls) > 0 else 0
    
    def is_valid(A_set):
        muls = compute_multiplicities(A_set)
        return np.all(muls >= 1)
    
    def variance_of_muls(A_set):
        muls = compute_multiplicities(A_set).astype(float)
        return float(np.var(muls))
    
    # --- Structured initialization ---
    def make_initial_set():
        """Create a valid initial set using symmetric structure."""
        # Strategy: use a set that's roughly symmetric around n/2
        # Start with a grid-like structure and fill gaps
        A = set()
        # Add elements in a pattern that ensures coverage
        # Use a "thickened" arithmetic progression
        step = 2
        for i in range(0, n + 1, step):
            A.add(i)
        # Ensure coverage by adding missing elements
        for _ in range(50):
            if is_valid(A):
                break
            # Find uncovered targets
            muls = compute_multiplicities(A)
            uncovered = [target_lo + i for i in range(len(muls)) if muls[i] == 0]
            if not uncovered:
                break
            # Try to add elements that help cover uncovered targets
            added = False
            for t in uncovered:
                # Try to find a, b not in A such that a + b = t
                for a in range(0, n + 1):
                    b = t - a
                    if 0 <= b <= n and a not in A and b not in A:
                        A.add(a)
                        A.add(b)
                        added = True
                        break
                if added:
                    break
            if not added:
                # Add a random element
                A.add(random.randint(0, n))
        return A
    
    # Alternative: use a known good structure
    def make_grid_set():
        """Create a set based on a coarse grid with refinements."""
        A = set()
        # Base grid
        for i in range(0, n + 1, 3):
            A.add(i)
        # Add some midpoints
        for i in range(1, n + 1, 3):
            if random.random() < 0.5:
                A.add(i)
        # Ensure validity
        for _ in range(100):
            if is_valid(A):
                break
            muls = compute_multiplicities(A)
            uncovered = [target_lo + i for i in range(len(muls)) if muls[i] == 0]
            if not uncovered:
                break
            for t in uncovered:
                for a in range(0, n + 1):
                    b = t - a
                    if 0 <= b <= n:
                        A.add(a)
                        A.add(b)
                        break
                break
        return A
    
    # Generate initial set
    A = make_grid_set()
    if not is_valid(A):
        A = make_initial_set()
    
    # Further refinement: try to reduce size while maintaining validity
    # Remove redundant elements
    for _ in range(200):
        if time_left() < 10:
            break
        removed = False
        for a in list(A):
            A.remove(a)
            if is_valid(A):
                removed = True
            else:
                A.add(a)
            if removed:
                break
        if not removed:
            break
    
    best_peak = peak_mult(A)
    best_A = set(A)
    
    # --- Local search ---
    def local_search(A, max_iters=5000):
        """Local search to minimize peak multiplicity."""
        A = set(A)
        current_peak = peak_mult(A)
        best_peak = current_peak
        best_A = set(A)
        
        for it in range(max_iters):
            if time_left() < 5:
                break
            
            # Generate random moves
            move_type = random.random()
            
            if move_type < 0.3:
                # Try removing an element
                if len(A) <= 10:
                    continue
                a = random.choice(list(A))
                A.remove(a)
                if is_valid(A):
                    new_peak = peak_mult(A)
                    if new_peak <= current_peak:
                        current_peak = new_peak
                        if new_peak < best_peak:
                            best_peak = new_peak
                            best_A = set(A)
                    else:
                        A.add(a)
                else:
                    A.add(a)
            
            elif move_type < 0.6:
                # Try swapping: remove one, add another
                a = random.choice(list(A))
                b = random.randint(0, n)
                if b == a or b in A:
                    continue
                A.remove(a)
                A.add(b)
                if is_valid(A):
                    new_peak = peak_mult(A)
                    if new_peak <= current_peak:
                        current_peak = new_peak
                        if new_peak < best_peak:
                            best_peak = new_peak
                            best_A = set(A)
                    else:
                        A.add(a)
                        A.remove(b)
                else:
                    A.add(a)
                    A.remove(b)
            
            else:
                # Try adding an element (to help balance)
                b = random.randint(0, n)
                if b in A:
                    continue
                A.add(b)
                new_peak = peak_mult(A)
                # Only keep if it reduces variance significantly
                muls = compute_multiplicities(A).astype(float)
                old_muls = compute_multiplicities(set(best_A)).astype(float)
                if np.var(muls) < np.var(old_muls) - 0.5:
                    current_peak = new_peak
                    if new_peak < best_peak:
                        best_peak = new_peak
                        best_A = set(A)
                else:
                    A.remove(b)
        
        return best_A, best_peak
    
    # Run local search
    A, best_peak = local_search(A, max_iters=3000)
    
    # Second pass with different initialization
    if time_left() > 15:
        A2 = make_grid_set()
        if is_valid(A2):
            # Reduce size
            for _ in range(100):
                if time_left() < 10:
                    break
                removed = False
                for a in list(A2):
                    A2.remove(a)
                    if is_valid(A2):
                        removed = True
                    else:
                        A2.add(a)
                    if removed:
                        break
                if not removed:
                    break
            A2, peak2 = local_search(A2, max_iters=2000)
            if peak2 < best_peak:
                best_peak = peak2
                A = A2
    
    # Final polish: try targeted improvements on high-multiplicity targets
    if time_left() > 10:
        for _ in range(500):
            if time_left() < 5:
                break
            muls = compute_multiplicities(A)
            if muls.max() <= 4:
                break
            # Find target with highest multiplicity
            max_idx = int(np.argmax(muls))
            max_target = target_lo + max_idx
            max_count = muls[max_idx]
            
            # Try to reduce multiplicity at this target
            # Find pairs that sum to max_target
            pairs = []
            A_list = list(A)
            for i, a in enumerate(A_list):
                b = max_target - a
                if b in A and (a <= b or b not in A_list[i+1:]):
                    pairs.append((a, b))
            
            # Try removing elements that participate in many pairs
            pair_count = Counter()
            for a, b in pairs:
                pair_count[a] += 1
                if a != b:
                    pair_count[b] += 1
            
            for elem, cnt in pair_count.most_common():
                if cnt >= 2:
                    A.remove(elem)
                    if is_valid(A):
                        new_peak = peak_mult(A)
                        if new_peak < best_peak:
                            best_peak = new_peak
                            A = set(A)
                    else:
                        A.add(elem)
                    break
    
    # Ensure final validity
    if not is_valid(A):
        # Emergency fix
        muls = compute_multiplicities(A)
        for i in range(len(muls)):
            if muls[i] == 0:
                t = target_lo + i
                for a in range(0, n + 1):
                    b = t - a
                    if 0 <= b <= n:
                        A.add(a)
                        A.add(b)
                        break
    
    return sorted(A)
# EVOLVE-BLOCK-END
