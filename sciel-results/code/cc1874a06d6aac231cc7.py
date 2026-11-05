import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82.0

    N = 128
    min_sum = 128
    max_sum = 256

    def compute_counts(A):
        """Compute pair-sum counts for all sums 0..256."""
        A_sorted = sorted(A)
        k = len(A_sorted)
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        for i in range(k):
            ai = A_sorted[i]
            s = ai + ai
            if s <= max_sum:
                counts[s] += 1
            for j in range(i + 1, k):
                aj = A_sorted[j]
                s = ai + aj
                if s <= max_sum:
                    counts[s] += 1
                else:
                    break
        return counts

    def peak_and_uncovered(counts):
        """Return (peak, uncovered_count) for target sums [128,256]."""
        target = counts[min_sum:max_sum + 1]
        uncovered = int(np.sum(target == 0))
        if uncovered > 0:
            return (1000 + uncovered, uncovered)
        peak = int(np.max(target))
        return (peak, 0)

    def try_greedy_construct(k, seed):
        """Try to build a set of size k that covers all targets, minimizing peak."""
        rng = random.Random(seed)
        # Start with evenly spaced points
        A = set()
        for i in range(k):
            x = round(i * 128.0 / (k - 1)) if k > 1 else 0
            A.add(x)
        
        counts = compute_counts(A)
        peak, uncov = peak_and_uncovered(counts)
        
        # Greedy: add elements to cover uncovered targets
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            target = counts[min_sum:max_sum + 1]
            uncovered_sums = [s for s in range(min_sum, max_sum + 1) if counts[s] == 0]
            if not uncovered_sums:
                break
            
            # Find best element to add that covers most uncovered sums
            best_x = None
            best_gain = 0
            for x in range(129):
                if x in A:
                    continue
                # Count how many uncovered sums x would help
                gain = 0
                for s in uncovered_sums:
                    # x + y = s for some y in A, or 2x = s
                    if s - x in A and s - x >= 0:
                        gain += 1
                    elif 2 * x == s:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None or best_gain == 0:
                # Try random
                candidates = [x for x in range(129) if x not in A]
                if not candidates:
                    break
                best_x = rng.choice(candidates)
            
            A.add(best_x)
            counts = compute_counts(A)
            peak, uncov = peak_and_uncovered(counts)
        
        return A, peak, uncov

    def local_search(A, max_iters=30000):
        """Local search to minimize peak multiplicity."""
        A = set(A)
        counts = compute_counts(A)
        peak, uncov = peak_and_uncovered(counts)
        best_peak = peak
        best_A = set(A)
        
        outside = [x for x in range(129) if x not in A]
        rng = random.Random(42)
        
        # Targeted local search: focus on reducing the peak
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            target = counts[min_sum:max_sum + 1]
            current_peak = int(np.max(target)) if uncov == 0 else 1000
            
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = set(A)
            
            if best_peak <= 3:
                break
            
            # Find the sums with highest multiplicity
            peak_sums = [s for s in range(min_sum, max_sum + 1) if counts[s] == current_peak]
            focus_sum = rng.choice(peak_sums) if peak_sums else rng.randint(min_sum, max_sum)
            
            # Try swaps: remove x from A, add y, see if peak decreases
            improved = False
            A_list = list(A)
            rng.shuffle(A_list)
            for x in A_list[:20]:  # Try first 20
                if time.time() > deadline - 5:
                    break
                # Temporarily remove x
                A.remove(x)
                # Subtract x's contributions
                for y in A:
                    s = x + y
                    if s <= max_sum:
                        counts[s] -= 1
                s2 = 2 * x
                if s2 <= max_sum:
                    counts[s2] -= 1
                
                # Try adding candidates
                for y in outside[:30]:
                    if time.time() > deadline - 5:
                        break
                    # Add y's contributions
                    for z in A:
                        s = y + z
                        if s <= max_sum:
                            counts[s] += 1
                    s2 = 2 * y
                    if s2 <= max_sum:
                        counts[s2] += 1
                    
                    new_target = counts[min_sum:max_sum + 1]
                    if np.min(new_target) == 0:
                        # Coverage broken
                        # Undo
                        for z in A:
                            s = y + z
                            if s <= max_sum:
                                counts[s] -= 1
                        if s2 <= max_sum:
                            counts[s2] -= 1
                        continue
                    
                    new_peak = int(np.max(new_target))
                    if new_peak < current_peak:
                        # Accept
                        A.add(y)
                        outside.remove(y)
                        outside.append(x)
                        improved = True
                        break
                    else:
                        # Undo
                        for z in A:
                            s = y + z
                            if s <= max_sum:
                                counts[s] -= 1
                        if s2 <= max_sum:
                            counts[s2] -= 1
                
                if improved:
                    break
                
                # Restore x
                A.add(x)
                for y in A:
                    if y == x:
                        continue
                    s = x + y
                    if s <= max_sum:
                        counts[s] += 1
                s2 = 2 * x
                if s2 <= max_sum:
                    counts[s2] += 1
            
            if not improved and it % 1000 == 0:
                # Occasional random restart of local search
                pass
        
        return best_A, best_peak

    # Phase 1: Try multiple greedy constructions with different sizes
    best_A = None
    best_peak = 10000
    
    for k in [18, 20, 22, 24, 26]:
        for seed in [0, 1, 2, 3, 4]:
            if time.time() > deadline - 30:
                break
            A, peak, uncov = try_greedy_construct(k, seed)
            if uncov == 0 and peak < best_peak:
                best_peak = peak
                best_A = set(A)
    
    if best_A is None:
        # Fallback: dense set
        best_A = set(range(0, 129, 4))
        counts = compute_counts(best_A)
        best_peak, _ = peak_and_uncovered(counts)
    
    # Phase 2: Local search on the best construction
    if time.time() < deadline - 10:
        best_A, best_peak = local_search(best_A)
    
    # Phase 3: Try a few more local searches from different starting points
    if time.time() < deadline - 10:
        for k in [20, 22, 24]:
            if time.time() > deadline - 10:
                break
            A, peak, uncov = try_greedy_construct(k, 100 + k)
            if uncov == 0:
                A2, peak2 = local_search(A, max_iters=10000)
                if peak2 < best_peak:
                    best_peak = peak2
                    best_A = A2
    
    # Final verification
    counts = compute_counts(best_A)
    target = counts[min_sum:max_sum + 1]
    assert np.min(target) > 0, "Coverage failure!"
    
    return list(best_A)
