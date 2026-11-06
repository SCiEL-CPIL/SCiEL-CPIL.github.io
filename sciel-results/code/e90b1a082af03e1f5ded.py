import math
import numpy as np
import time
import random
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 30
    
    # Calculate LCM of 1..30
    H = reduce(math.lcm, range(1, n + 1))
    
    # Weights W_i = H / i
    W = [H // i for i in range(1, n + 1)]
    
    # Suffix sums for pruning
    R = [0] * (n + 2)
    for i in range(n, 0, -1):
        R[i] = R[i + 1] + W[i - 1]
    
    # Phase 1: Beam search to find good initial candidates
    K = 20000  # Larger beam width for better coverage
    
    current_states = [(0, [0] * n)]
    
    for i in range(n):
        next_states = []
        w = W[i]
        
        for s, coeffs in current_states:
            for c in [-1, 0, 1]:
                new_s = s + c * w
                new_coeffs = coeffs.copy()
                new_coeffs[i] = c
                next_states.append((new_s, new_coeffs))
        
        # Sort by absolute value of sum
        next_states.sort(key=lambda x: abs(x[0]))
        
        # Keep top K
        current_states = next_states[:K]
        
        # Time check
        if time.time() > deadline - 30:
            break
    
    # Find best non-zero candidates from beam search
    candidates = []
    for s, coeffs in current_states:
        if s != 0:
            candidates.append((abs(s), coeffs))
    
    candidates.sort(key=lambda x: x[0])
    
    # Take top candidates for local search
    num_candidates = min(100, len(candidates))
    top_candidates = [coeffs for _, coeffs in candidates[:num_candidates]]
    
    # Phase 2: Local search on top candidates
    best_val = float('inf')
    best_coeffs = None
    
    def update(coeffs):
        nonlocal best_val, best_coeffs
        total = sum(c * W[i] for i, c in enumerate(coeffs))
        if total == 0:
            return
        val = abs(total)
        if val < best_val:
            best_val = val
            best_coeffs = list(coeffs)
    
    # Initialize with trivial pair: 1/29 - 1/30
    trivial = [0] * 28 + [1, -1]
    update(trivial)
    
    # Local search on each candidate
    random.seed(42)
    
    for start_idx, init_coeffs in enumerate(top_candidates):
        if time.time() > deadline - 5:
            break
        
        current = list(init_coeffs)
        update(current)
        
        # Local search: try single flips and small mutations
        for step in range(200):
            if time.time() > deadline - 5:
                break
            
            # Try flipping a random coefficient
            idx = random.randint(0, 29)
            old_val = current[idx]
            
            # Choose new value different from old
            options = [v for v in [-1, 0, 1] if v != old_val]
            current[idx] = random.choice(options)
            
            total = sum(c * W[i] for i, c in enumerate(current))
            if total != 0:
                val = abs(total)
                if val < best_val:
                    best_val = val
                    best_coeffs = list(current)
                else:
                    # Revert with high probability
                    if random.random() > 0.1:
                        current[idx] = old_val
            else:
                current[idx] = old_val
        
        # Try targeted improvement: focus on coefficients that contribute most to cancellation
        if time.time() > deadline - 5:
            break
        
        # Find the index where flipping would most reduce |sum|
        current_sum = sum(c * W[i] for i, c in enumerate(current))
        if current_sum != 0:
            sign = 1 if current_sum > 0 else -1
            # Find coefficient that, when flipped, reduces |sum| the most
            best_idx = -1
            best_improvement = 0
            for i in range(n):
                old_c = current[i]
                for new_c in [-1, 0, 1]:
                    if new_c == old_c:
                        continue
                    delta = (new_c - old_c) * W[i]
                    new_sum = current_sum + delta
                    if new_sum != 0:
                        improvement = abs(current_sum) - abs(new_sum)
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_idx = i
                            best_new_c = new_c
            if best_idx >= 0 and best_improvement > 0:
                current[best_idx] = best_new_c
                update(current)
    
    # Phase 3: Try specific constructions that are known to work well
    # Alternating patterns with consecutive indices
    for start in range(1, 25):
        for length in range(3, 10):
            if start + length > 31:
                break
            coeffs = [0] * 30
            for k in range(length):
                coeffs[start - 1 + k] = 1 if k % 2 == 0 else -1
            update(coeffs)
    
    # Phase 4: Random restarts with local search
    for restart in range(50):
        if time.time() > deadline - 3:
            break
        
        # Random sparse starting point
        current = [0] * 30
        num_ones = random.randint(2, 6)
        indices = random.sample(range(30), num_ones)
        for idx in indices:
            current[idx] = random.choice([-1, 1])
        update(current)
        
        # Local search
        for step in range(100):
            if time.time() > deadline - 3:
                break
            
            idx = random.randint(0, 29)
            old_val = current[idx]
            options = [v for v in [-1, 0, 1] if v != old_val]
            current[idx] = random.choice(options)
            
            total = sum(c * W[i] for i, c in enumerate(current))
            if total != 0:
                val = abs(total)
                if val < best_val:
                    best_val = val
                    best_coeffs = list(current)
                else:
                    if random.random() > 0.3:
                        current[idx] = old_val
            else:
                current[idx] = old_val
    
    # Ensure we return a valid non-zero solution
    if best_coeffs is None:
        # Fallback to trivial solution
        best_coeffs = [0] * 28 + [1, -1]
    
    # Final verification
    total = sum(c * W[i] for i, c in enumerate(best_coeffs))
    if total == 0:
        # Force non-zero
        min_idx = min(range(n), key=lambda i: W[i])
        if best_coeffs[min_idx] == 1:
            best_coeffs[min_idx] = -1
        else:
            best_coeffs[min_idx] = 1
    
    return best_coeffs
