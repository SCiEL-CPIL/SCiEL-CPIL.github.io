"""Erdos minimum overlap - find step heights minimizing max overlap with shifted complement."""
import numpy as np
import time

N = 100

def compute_max_overlap(h):
    """Compute the maximum overlap of h with shifted (1-h) over all shifts."""
    N = len(h)
    width = 2.0 / N
    max_overlap = 0.0
    for s in range(N):
        # Shift by s steps
        # h_shifted[i] = h[(i + s) % N]
        # overlap[i] = h[i] * (1 - h_shifted[i])
        # But we need to handle the wrap-around carefully
        # Actually for shifts, we consider x in [0,2] and x+s in [0,2]
        # So for shift s, x ranges from 0 to 2-s (in step units, 0 to N-s-1)
        # And we integrate h(x) * (1-h(x+s)) over x in [0, 2-s]
        # But the problem might consider circular shifts or linear shifts
        # Let's assume linear: for shift s steps, x from step 0 to step N-s-1
        if s == 0:
            continue  # shift 0: overlap = integral of h*(1-h) = 0 for binary
        # For shift s, valid x indices: 0 to N-s-1
        # h[x] * (1 - h[x+s])
        overlap = np.sum(h[:N-s] * (1 - h[s:])) * width
        max_overlap = max(max_overlap, overlap)
    return max_overlap

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline with margin
    
    best_h = np.full(N, 0.5)
    best_score = compute_max_overlap(best_h)
    
    # Try periodic binary patterns
    # For period P (even), pattern of P steps with P/2 ones
    for P in range(2, N+1, 2):
        if time.time() > deadline - 10:
            break
        if N % P != 0:
            continue
        
        # Generate all patterns with P/2 ones in P positions
        # This is combinatorial, so for small P try all, for large P use heuristics
        if P <= 20:
            from itertools import combinations
            ones_positions = list(range(P))
            for combo in combinations(ones_positions, P // 2):
                pattern = np.zeros(P)
                for pos in combo:
                    pattern[pos] = 1.0
                # Check if pattern is aperiodic (not a repeat of a smaller period)
                # Build full array
                h = np.tile(pattern, N // P)
                # Ensure sum is N/2
                if np.sum(h) != N / 2:
                    continue
                score = compute_max_overlap(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
        else:
            # For larger periods, try block patterns: k consecutive ones, k consecutive zeros
            # This is the 111...1000...0 pattern
            k = P // 2
            pattern = np.zeros(P)
            pattern[:k] = 1.0
            h = np.tile(pattern, N // P)
            if np.sum(h) == N / 2:
                score = compute_max_overlap(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
    
    # Also try the 1100 pattern specifically (period 4)
    pattern_1100 = np.array([1.0, 1.0, 0.0, 0.0])
    if N % 4 == 0:
        h_1100 = np.tile(pattern_1100, N // 4)
        score_1100 = compute_max_overlap(h_1100)
        if score_1100 < best_score:
            best_score = score_1100
            best_h = h_1100.copy()
    
    # Ensure sum is exactly N/2
    if np.sum(best_h) != N / 2:
        # Adjust
        diff = N / 2 - np.sum(best_h)
        if abs(diff) < 1e-10:
            best_h = np.round(best_h, 10)
        else:
            # Fallback to 0.5
            best_h = np.full(N, 0.5)
    
    return best_h
