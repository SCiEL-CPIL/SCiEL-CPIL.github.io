import time
import random
import numpy as np

def _compute_peak(A_set):
    """
    Computes the peak representation multiplicity for sums in [128, 256].
    Returns (peak, coverage_ok).
    """
    A = sorted(A_set)
    n = len(A)
    # counts for sums 128..256
    counts = np.zeros(129, dtype=np.int32)
    for i in range(n):
        a = A[i]
        for j in range(i, n):
            s = a + A[j]
            if 128 <= s <= 256:
                counts[s - 128] += 1
    peak = int(counts.max())
    coverage_ok = bool((counts > 0).all())
    return peak, coverage_ok

def _local_search(A_set, max_iters=5000, deadline=None):
    """
    Local search to minimize peak multiplicity while maintaining coverage.
    """
    A_set = set(A_set)
    peak, cov = _compute_peak(A_set)
    if not cov:
        return A_set, peak, False
    
    best_peak = peak
    best_set = set(A_set)
    rng = random.Random(int(time.time() * 1000) % 10000)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Random swap move
        if len(A_set) < 2:
            continue
            
        a = rng.choice(list(A_set))
        # Try a few random candidates for b
        improved = False
        for _ in range(5):
            b = rng.randint(0, 128)
            if b == a:
                continue
            if b in A_set:
                continue
                
            new_set = A_set - {a} | {b}
            new_peak, new_cov = _compute_peak(new_set)
            
            if new_cov and new_peak < best_peak:
                best_peak = new_peak
                best_set = new_set
                A_set = new_set
                peak = new_peak
                improved = True
                break
            elif new_cov and new_peak == peak and len(new_set) < len(A_set):
                # Prefer smaller sets if peak is same? No, size doesn't matter for score, only peak.
                # But smaller sets might be easier to optimize?
                # Actually, we want to minimize peak. Size is irrelevant to the score.
                # However, removing elements might break coverage.
                # We only accept if coverage is maintained.
                pass
            elif new_cov and new_peak == peak:
                # Accept with low probability to escape local minima?
                # Since peak is the same, it's neutral.
                if rng.random() < 0.1:
                    A_set = new_set
                    best_set = new_set
                    improved = True
                    break

        if not improved:
            # Try adding an element to escape?
            # If peak is stuck, maybe we need more elements to distribute sums?
            # Or fewer elements?
            # Usually, more elements -> higher peak.
            # So we want to keep size small.
            pass

    return best_set, best_peak, True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best_set = None
    best_peak = float('inf')
    
    # Multiple restarts
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize with a structured set
        # Try different sizes and structures
        # Structure 1: Two blocks
        k1 = random.randint(10, 20)
        k2 = random.randint(10, 20)
        
        # Block 1: [0, k1-1]
        # Block 2: [128-k2+1, 128]
        A_init = set(range(0, k1))
        A_init.update(range(128 - k2 + 1, 129))
        
        # Add some random middle elements to help cover gaps
        for _ in range(random.randint(0, 5)):
            A_init.add(random.randint(30, 100))
            
        # Ensure coverage
        # Simple greedy addition
        A_curr = set(A_init)
        for _ in range(50): # Max 50 additions
            peak, cov = _compute_peak(A_curr)
            if cov:
                break
            if time.time() > deadline - 5:
                break
                
            # Find uncovered
            A_sorted = sorted(A_curr)
            counts = np.zeros(129, dtype=np.int32)
            for i in range(len(A_sorted)):
                a = A_sorted[i]
                for j in range(i, len(A_sorted)):
                    s = a + A_sorted[j]
                    if 128 <= s <= 256:
                        counts[s - 128] += 1
                        
            uncovered = [s for s in range(129) if counts[s] == 0]
            if not uncovered:
                break
                
            # Find best element to add
            best_elem = None
            best_cover = 0
            for cand in range(129):
                if cand in A_curr:
                    continue
                cover = 0
                for t in uncovered:
                    target = t + 128
                    needed = target - cand
                    if 0 <= needed <= 128 and needed in A_curr:
                        cover += 1
                if cover > best_cover:
                    best_cover = cover
                    best_elem = cand
                    
            if best_elem is not None and best_cover > 0:
                A_curr.add(best_elem)
            else:
                A_curr.add(random.randint(0, 128))
                
        if time.time() > deadline - 5:
            continue
            
        peak, cov = _compute_peak(A_curr)
        if not cov:
            continue
            
        # Local search
        A_opt, peak_opt, cov_opt = _local_search(A_curr, max_iters=2000, deadline=deadline - 2)
        
        if cov_opt and peak_opt < best_peak:
            best_peak = peak_opt
            best_set = A_opt
            
        if best_peak <= 2:
            break
            
    if best_set is None:
        # Fallback
        return list(range(0, 129))
        
    return sorted(best_set)
