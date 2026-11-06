import time
import numpy as np
import random

def construct():
    deadline = time.time() + 115.0
    base = 51975
    
    # Generate divisors > 1
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
            
    # Precompute masks for all possible classes
    # class_index -> (m, r, mask)
    # We will store masks as numpy arrays of uint8 for fast bitwise ops
    classes = []
    for m in divisors:
        for r in range(m):
            # Create mask for this class
            # np.arange(r, base, m) gives indices
            # We can use a boolean array or integer array
            # Using uint8 array of size base
            mask = np.zeros(base, dtype=np.uint8)
            mask[r::m] = 1
            classes.append((m, r, mask))
            
    # Sort classes by size descending (heuristic for greedy)
    # Size is base // m (approx)
    classes.sort(key=lambda x: base // x[0], reverse=True)
    
    best_uncovered = base
    best_classes = []
    
    # Run multiple greedy passes with different tie-breaking/randomness
    # Since the number of classes is ~12k, and we pick ~20-40 of them,
    # the loop runs ~40 times. Each step scans 12k items. 40 * 12k = 480k ops.
    # Each op is a bitwise AND and count on 51975 elements.
    # 51975 * 480k is too slow if done naively in Python loops.
    # We need to be careful.
    
    # Alternative: 
    # 1. Group classes by modulus.
    # 2. For each modulus, precompute the best residue for the current covered mask?
    #    No, the best residue changes as covered mask changes.
    
    # Let's optimize the gain calculation.
    # gain = np.sum(mask & (1 - covered_mask))
    # This is O(base) per candidate.
    # Total candidates ~12k.
    # Steps ~ number of selected moduli.
    # Worst case we select all 47 moduli.
    # 47 * 12k * 52k operations = ~30 billion operations. Too slow for Python/NumPy in 120s?
    # NumPy is C-backed. 52k elements is small.
    # 12k candidates * 52k elements = 600M operations per step? No.
    # For each step, we iterate through available classes.
    # Available classes decreases as we pick moduli.
    # Initially 12k classes.
    # Step 1: Evaluate 12k classes. Each eval: 52k bitwise ops. 12k * 52k = 624M ops.
    # This might take ~0.5 - 1 second in NumPy.
    # Step 2: Remove all classes with chosen modulus.
    # If we pick small moduli first, we remove many classes?
    # No, we pick ONE class per modulus.
    # So if we pick modulus m, we remove ALL classes with modulus m.
    # Number of classes with modulus m is m.
    # So the number of available classes drops significantly if we pick large moduli?
    # No, the number of *moduli* is small (47).
    # The number of *classes* is sum(m).
    # If we pick modulus m, we remove m classes from the pool.
    
    # Strategy:
    # Maintain a list of available classes.
    # In each step, find the class with max gain.
    # To speed up, we can sample or use a heuristic.
    # But 12k * 52k * 47 steps might be tight.
    
    # Let's try to optimize by grouping by modulus.
    # For each modulus m, we only need to consider the best residue r for that m given the current covered mask.
    # So for each modulus m, we calculate the gain for all r in [0, m) and pick the best r_m.
    # Then we compare the best r_m across all available moduli m.
    # This reduces the number of gain calculations.
    # Total gain calcs per step = sum(m for m in available_moduli).
    # This is the same as number of available classes.
    # But we can parallelize or use vectorization better?
    
    # Let's just implement the straightforward NumPy version and see if it fits in time.
    # If not, we can reduce the number of restarts or steps.
    
    def greedy_pass(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        covered = np.zeros(base, dtype=np.uint8)
        selected = []
        used_mods = set()
        
        # List of (m, r, mask)
        available = classes[:]
        
        # To allow randomness in tie-breaking or initial order
        # We can shuffle the available list slightly or pick randomly among top gains
        
        while True:
            if time.time() > deadline - 5.0:
                break
            
            best_gain = -1
            best_class = None
            best_m = None
            best_r = None
            
            # Iterate through available classes
            # Optimization: We can skip classes where m is already used?
            # No, we remove them when we pick a modulus.
            
            for m, r, mask in available:
                if m in used_mods:
                    continue
                    
                # Calculate gain
                # gain = number of 1s in mask where covered is 0
                # gain = np.sum(mask & (1 - covered))
                # Using integer arithmetic might be faster?
                # mask and covered are uint8.
                # 1 - covered is uint8.
                # mask & (1 - covered) is uint8.
                # sum is int.
                
                # Vectorized gain calculation
                # We can precompute 1-covered? No, covered changes.
                
                # Let's try:
                # uncovered_mask = 1 - covered
                # gain = np.dot(mask, uncovered_mask) # Since mask is 0/1
                # np.dot is fast.
                
                # But we need to do this for each candidate.
                # This is the bottleneck.
                
                # Alternative:
                # Precompute list of indices for each mask? No, memory.
                
                # Let's just do it.
                gain = int(np.dot(mask, 1 - covered))
                
                if gain > best_gain:
                    best_gain = gain
                    best_class = (m, r, mask)
                    best_m = m
                    best_r = r
            
            if best_gain <= 0:
                break
            
            selected.append([best_m, best_r])
            used_mods.add(best_m)
            
            # Update covered
            covered |= best_class[2]
            
            # Remove all classes with modulus best_m from available
            # This is O(N) but N is small (12k) and we do it 47 times max.
            available = [c for c in available if c[0] != best_m]
            
            if not available:
                break
            
        uncovered = int(np.sum(1 - covered))
        return uncovered, selected

    # Run multiple passes
    # Pass 1: Standard
    u, s = greedy_pass()
    if u < best_uncovered:
        best_uncovered = u
        best_classes = s
        
    # Pass 2-10: Randomized order or tie-breaking
    # To introduce randomness, we can add a small random noise to the gain?
    # Or shuffle the 'available' list at each step?
    # Shuffling 12k items 47 times is cheap.
    
    for i in range(10):
        if time.time() > deadline - 10.0:
            break
        # Modify greedy to accept a random seed for tie-breaking
        # We can just call greedy_pass with a seed that shuffles the initial list
        # But the greedy logic picks max gain. Ties are broken by iteration order.
        # So shuffling the initial list changes tie-breaking.
        
        # Create a shuffled copy of classes for this pass
        shuffled_classes = classes[:]
        random.seed(i)
        random.shuffle(shuffled_classes)
        
        # We need to modify greedy_pass to take a list of classes
        # Let's refactor greedy_pass to accept the class list
        
        # Actually, let's just write a separate function or inline it for speed/clarity
        covered = np.zeros(base, dtype=np.uint8)
        selected = []
        used_mods = set()
        available = shuffled_classes[:]
        
        while True:
            if time.time() > deadline - 5.0:
                break
            
            best_gain = -1
            best_m = None
            best_r = None
            best_mask = None
            
            for m, r, mask in available:
                if m in used_mods:
                    continue
                gain = int(np.dot(mask, 1 - covered))
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
                    best_mask = mask
            
            if best_gain <= 0:
                break
            
            selected.append([best_m, best_r])
            used_mods.add(best_m)
            covered |= best_mask
            available = [c for c in available if c[0] != best_m]
            if not available:
                break
        
        uncovered = int(np.sum(1 - covered))
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_classes = selected

    return best_classes
