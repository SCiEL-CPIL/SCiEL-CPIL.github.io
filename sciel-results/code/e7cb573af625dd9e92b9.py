"""
Strategy:
1. Generate Batcher's Odd-Even Merge Sort network (63 comparators) as the starting point.
2. Perform an initial greedy deletion pass to quickly remove obviously redundant comparators.
3. Use Simulated Annealing (SA) to search the space of comparator sets.
   - State: A set of comparators (list of pairs).
   - Moves: 
     a) Remove a random comparator (decrease size).
     b) Insert a random comparator (increase size, helps escape local minima).
     c) Replace a random comparator with another (same size, structural change).
   - Fitness: Number of unsorted boolean inputs (violations). Lower is better.
     - If violations == 0, it is a valid sorting network.
     - We prefer smaller networks among valid ones.
   - Temperature Schedule: Exponential cooling.
   - Acceptance: 
     - If new state is valid and smaller, always accept.
     - If new state is valid and same size, always accept.
     - If new state is invalid, accept with probability exp(-delta_violations / T) 
       (where delta_violations is the increase in violations).
       However, since we care primarily about validity first, then size, 
       we define a cost function: Cost = (violations * 10000) + size.
       This prioritizes finding ANY valid network over a small invalid one, 
       but once valid, minimizes size.
       Actually, a better cost for SA in this context:
       If current is valid and new is valid: accept if new_size <= current_size (or with prob if larger).
       If current is valid and new is invalid: accept with prob exp(-T_inv) where T_inv is high penalty.
       If current is invalid and new is valid: always accept.
       If both invalid: accept if new has fewer violations (or with prob).
       
       Let's use a unified cost:
       cost = violations * 10000 + size
       delta_cost = new_cost - old_cost
       accept if delta_cost <= 0 or rand() < exp(-delta_cost / T)
       
       This effectively searches for the minimum cost state.
"""

import time
import numpy as np
import math

N = 16
MAX_TIME = 58.0 # Safety margin for 60s limit

def _generate_batcher_network():
    """
    Generates Batcher's Odd-Even Merge Sort network for N=16.
    """
    def merge(n, offset):
        if n == 1:
            return [(offset, offset + 1)]
        
        h = n // 2
        # Step 1: Compare (offset+i) with (offset+i+n)
        step1 = [(offset + i, offset + i + n) for i in range(n)]
        
        def strided_merge(n, offset, stride):
            if n == 1:
                return [(offset, offset + stride)]
            
            h = n // 2
            # Step 1: Compare (offset + i*stride) with (offset + (n+i)*stride)
            step1 = [(offset + i * stride, offset + (n + i) * stride) for i in range(n)]
            
            # Recursive merges for even and odd indices
            even = strided_merge(h, offset, 2 * stride)
            odd = strided_merge(h, offset + stride, 2 * stride)
            
            return step1 + even + odd
        
        even_merge = strided_merge(h, offset, 2)
        odd_merge = strided_merge(h, offset + 1, 2)
        
        return step1 + even_merge + odd_merge

    def sort(n, offset):
        if n <= 1:
            return []
        h = n // 2
        left = sort(h, offset)
        right = sort(h, offset + h)
        m = merge(h, offset)
        return left + right + m

    return sort(N, 0)

def _apply_network_vectorized(comps, inputs):
    """
    Apply the network to a batch of inputs.
    """
    vec = inputs.copy()
    for i, j in comps:
        min_val = np.minimum(vec[:, i], vec[:, j])
        max_val = np.maximum(vec[:, i], vec[:, j])
        vec[:, i] = min_val
        vec[:, j] = max_val
    return vec

def _get_violations(comps, bits):
    """
    Returns the number of boolean inputs that are NOT sorted by the network.
    """
    out = _apply_network_vectorized(comps, bits)
    diffs = np.diff(out, axis=1)
    # A row is sorted if all diffs are >= 0. 
    # Unsorted if any diff < 0.
    is_sorted = np.all(diffs >= 0, axis=1)
    violations = np.sum(~is_sorted)
    return violations

def _cost(comps, violations):
    """
    Cost function for SA.
    Prioritizes validity (0 violations) heavily, then minimizes size.
    """
    return violations * 10000 + len(comps)

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Pre-compute all boolean inputs
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    # Bit i corresponds to wire i. 
    # Input is an integer 0..2^16-1. 
    # We want bits such that bit k of the integer corresponds to wire k.
    # np.arange(N) is [0, 1, ..., 15].
    # (indices[:, None] >> np.arange(N)) & 1 gives column j as bit j of the index.
    # This maps index bit j to wire j.
    bits = (indices[:, None] >> np.arange(N)) & 1
    
    # 1. Generate initial network (Batcher's, 63 comparators)
    current_comps = _generate_batcher_network()
    
    # Verify initial network
    init_violations = _get_violations(current_comps, bits)
    if init_violations != 0:
        # Fallback: Odd-Even Transposition (should be valid)
        current_comps = []
        for i in range(N):
            if i % 2 == 0:
                for j in range(0, N - 1, 2):
                    current_comps.append((j, j + 1))
            else:
                for j in range(1, N - 1, 2):
                    current_comps.append((j, j + 1))
    
    best_comps = current_comps[:]
    best_violations = _get_violations(best_comps, bits)
    best_cost = _cost(best_comps, best_violations)
    
    # 2. Initial Greedy Deletion Pass (fast)
    # Try removing comparators from the end
    i = len(current_comps) - 1
    while i >= 0 and time.time() < deadline - 10.0:
        comp = current_comps[i]
        current_comps.pop(i)
        v = _get_violations(current_comps, bits)
        c = _cost(current_comps, v)
        if c < best_cost:
            best_comps = current_comps[:]
            best_violations = v
            best_cost = c
        else:
            current_comps.insert(i, comp)
            i -= 1
            
    # Try removing from the beginning
    i = 0
    while i < len(current_comps) and time.time() < deadline - 10.0:
        comp = current_comps[i]
        current_comps.pop(i)
        v = _get_violations(current_comps, bits)
        c = _cost(current_comps, v)
        if c < best_cost:
            best_comps = current_comps[:]
            best_violations = v
            best_cost = c
        else:
            current_comps.insert(i, comp)
            i += 1
            
    # 3. Simulated Annealing
    # Start from best found so far
    current_comps = best_comps[:]
    current_violations = best_violations
    current_cost = best_cost
    
    # Pre-compute all possible comparators for random insertion/replacement
    all_comps = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng = np.random.default_rng(42)
    
    # SA Parameters
    T_start = 50.0
    T_end = 0.1
    cooling_rate = 0.995
    
    T = T_start
    
    # Track steps to adjust cooling if necessary
    # Estimate steps per second roughly: 
    # _get_violations takes ~1-2ms for 65536 inputs. 
    # So ~500-1000 steps/sec. 
    # Total time ~40s. Total steps ~20000-40000.
    # We will run until time runs out.
    
    while time.time() < deadline - 1.0:
        if T < T_end:
            T = T_end
            
        # Generate a neighbor
        move_type = rng.random()
        
        # 70% Remove, 15% Insert, 15% Replace
        if move_type < 0.70:
            # Remove
            if len(current_comps) == 0:
                continue
            idx = rng.integers(0, len(current_comps))
            removed_comp = current_comps.pop(idx)
            new_comps = current_comps
            # Note: we mutated current_comps. We need to restore if rejected.
            # For efficiency, we calculate cost on the mutated list.
            
        elif move_type < 0.85:
            # Insert
            # Choose a random comparator not in the current set (or even if it is, adding duplicate is harmless but useless, 
            # but let's try to pick one not present to increase diversity)
            # For simplicity, pick from all_comps.
            new_comp = all_comps[rng.integers(0, len(all_comps))]
            current_comps.append(new_comp)
            new_comps = current_comps
            
        else:
            # Replace
            if len(current_comps) == 0:
                continue
            idx = rng.integers(0, len(current_comps))
            old_comp = current_comps[idx]
            new_comp = all_comps[rng.integers(0, len(all_comps))]
            if new_comp == old_comp:
                continue
            current_comps[idx] = new_comp
            new_comps = current_comps
            removed_comp = old_comp # Not used for restore in replace, we just revert index
            
        # Evaluate
        new_violations = _get_violations(new_comps, bits)
        new_cost = _cost(new_comps, new_violations)
        
        delta_cost = new_cost - current_cost
        
        # Acceptance
        accept = False
        if delta_cost <= 0:
            accept = True
        else:
            # Probability of accepting worse state
            # Avoid division by zero or overflow
            prob = math.exp(-delta_cost / T)
            if rng.random() < prob:
                accept = True
        
        if accept:
            current_cost = new_cost
            current_violations = new_violations
            # Update best if improved
            if new_cost < best_cost:
                best_cost = new_cost
                best_violations = new_violations
                best_comps = new_comps[:]
                
        else:
            # Reject: Restore state
            if move_type < 0.70:
                # Was Remove
                current_comps.insert(idx, removed_comp)
            elif move_type < 0.85:
                # Was Insert
                current_comps.pop()
            else:
                # Was Replace
                current_comps[idx] = old_comp
                
        # Cool down
        T *= cooling_rate
        
    return best_comps

# EVOLVE-BLOCK-END
