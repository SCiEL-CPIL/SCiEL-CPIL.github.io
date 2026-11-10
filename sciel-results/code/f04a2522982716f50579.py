import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    n = 128
    targets = list(range(128, 257))  # 128 to 256 inclusive
    
    def get_representations(A_set):
        """For each target, count number of unordered pairs (a,b) in A with a<=b and a+b=target"""
        reps = {t: 0 for t in targets}
        A_list = sorted(A_set)
        for i in range(len(A_list)):
            for j in range(i, len(A_list)):
                s = A_list[i] + A_list[j]
                if s in reps:
                    reps[s] += 1
        return reps
    
    def peak_and_uncovered(A_set):
        reps = get_representations(A_set)
        peak = max(reps.values()) if reps else 0
        uncovered = [t for t in targets if reps[t] == 0]
        return peak, uncovered, reps
    
    # Initial set: try a structured approach
    # Use two clusters to create balanced coverage
    # Start with A = {0, 2, 4, ..., 126} (evens) - 64 elements
    # This gives sums that are even. We need odd sums too.
    # Better: A = {0, 1, 2, ..., 63} gives sums 0..126, not enough.
    
    # Let's try: A = {0, 1, ..., 31} ∪ {96, 97, ..., 128}
    # Sums range from 0 to 256
    # But coverage of [128,256] needs checking
    
    # Actually, let me start with the baseline and improve
    A = set(range(64, 129))  # baseline
    peak, uncovered, reps = peak_and_uncovered(A)
    best_peak = peak
    best_A = set(A)
    
    # Try a different initial structure: symmetric around 64
    # A = {0, 1, ..., 64} ∪ {64, 65, ..., 128} = {0..128} too big
    # A = {0, 2, 4, ..., 128} ∪ {1, 3, 5, ..., 127}? That's all of [0,128]
    
    # Let me try a smaller set with good coverage
    # If A = {0, 1, ..., k-1}, max sum is 2k-2. Need 2k-2 >= 256, so k >= 129. Too big.
    # So we need elements spread out.
    
    # Strategy: use elements in [0,128] such that pair sums cover [128,256]
    # Minimum size: if we have k elements, max sum is at most 256 (128+128)
    # To cover sum 128, need a+b=128 with a,b in A
    # To cover sum 256, need 128+128, so 128 must be in A
    # To cover sum 128, need pairs summing to 128
    
    # Let me try a greedy construction
    # Start with 128 (needed for sum 256)
    # Then add elements to cover remaining sums
    
    # Alternative: use local search from multiple initializations
    initializations = [
        set(range(64, 129)),  # baseline
        set(range(0, 64)),    # low end
        set(range(0, 65)) | set(range(64, 129)),  # all (too big but valid)
        {i for i in range(0, 129) if i % 2 == 0} | {128},  # evens + 128
        {i for i in range(0, 129) if i % 2 == 1} | {128},  # odds + 128
        set(range(32, 97)),   # middle block
        set(range(0, 32)) | set(range(96, 129)),  # two blocks
        set(range(0, 16)) | set(range(112, 129)),  # extreme blocks
    ]
    
    best_overall_peak = float('inf')
    best_overall_A = None
    
    for init_A in initializations:
        if time.time() - start_time > deadline - 5:
            break
        # Ensure 128 is in A (needed for sum 256)
        init_A = set(init_A)
        if 128 not in init_A:
            init_A.add(128)
        
        peak, uncovered, reps = peak_and_uncovered(init_A)
        
        # If there are uncovered targets, try to fix by adding elements
        if uncovered:
            # Greedily add elements to cover uncovered sums
            changed = True
            while uncovered and changed and time.time() - start_time < deadline - 5:
                changed = False
                # For each uncovered sum, find which element to add would help most
                best_gain = 0
                best_elem = None
                for elem in range(0, 129):
                    if elem in init_A:
                        continue
                    # Check how many uncovered sums this would help cover
                    gain = 0
                    for t in uncovered:
                        other = t - elem
                        if other in init_A or other == elem:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_elem = elem
                if best_elem is not None and best_gain > 0:
                    init_A.add(best_elem)
                    changed = True
                    peak, uncovered, reps = peak_and_uncovered(init_A)
        
        if not uncovered:
            if peak < best_overall_peak:
                best_overall_peak = peak
                best_overall_A = set(init_A)
    
    # Now do local search on the best found
    if best_overall_A is not None:
        A = set(best_overall_A)
    else:
        A = set(range(64, 129))  # fallback to baseline
    
    peak, uncovered, reps = peak_and_uncovered(A)
    if uncovered:
        # Fallback: use baseline which is known to work
        A = set(range(64, 129))
        peak, uncovered, reps = peak_and_uncovered(A)
    
    best_peak = peak
    best_A = set(A)
    
    # Local search: try swapping elements
    random.seed(42)
    iterations = 0
    while time.time() - start_time < deadline - 2:
        iterations += 1
        # Pick a random element to remove and a random element to add
        if len(A) <= 1:
            break
        
        elem_remove = random.choice(list(A))
        elem_add = random.randint(0, 128)
        
        if elem_add == elem_remove:
            continue
        if elem_add in A:
            continue
        
        new_A = set(A)
        new_A.discard(elem_remove)
        new_A.add(elem_add)
        
        new_peak, new_uncovered, new_reps = peak_and_uncovered(new_A)
        
        if not new_uncovered:
            if new_peak < best_peak:
                best_peak = new_peak
                best_A = new_A
                A = new_A
        # Also accept if same peak but smaller set (for potential future improvement)
        elif not new_uncovered and new_peak == best_peak and len(new_A) < len(A):
            A = new_A
            best_A = new_A
    
    # Also try pure addition/removal to reduce size while maintaining peak
    while time.time() - start_time < deadline - 1:
        improved = False
        for elem in list(A):
            test_A = set(A)
            test_A.discard(elem)
            if len(test_A) == 0:
                continue
            tp, tu, tr = peak_and_uncovered(test_A)
            if not tu and tp <= best_peak:
                A = test_A
                improved = True
                break
        if not improved:
            break
    
    # Final check
    final_peak, final_uncovered, _ = peak_and_uncovered(A)
    if final_uncovered:
        # Emergency fallback
        A = set(range(64, 129))
    
    return list(A)
