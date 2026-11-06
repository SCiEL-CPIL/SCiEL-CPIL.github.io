"""
Strategy:
1. Constructive Greedy Search:
   - Instead of starting with a known large network (like Batcher's 63-comparator network) and pruning it,
     we build the network from scratch (or from a minimal seed).
   - We maintain a set of "unsorted" zero-one patterns.
   - At each step, we select the comparator (i, j) that maximizes the reduction in the number of unsorted patterns.
   - This is a "generative" approach. It avoids getting stuck in local minima of the pruning space because
     it explores the space of valid networks by building them up.
   - To handle the combinatorial explosion, we use a fast vectorized check to evaluate the impact of each potential comparator.
   - We stop when all patterns are sorted or we hit the time limit.
   - We then try to prune the resulting network using the same fast checks.

2. Hybrid Initialization:
   - Start with an empty network.
   - Use a heuristic to prioritize comparators that are likely to be useful (e.g., those in Batcher's network).
   - Actually, a pure random constructive search might be too slow. 
   - Better: Start with a small valid sub-network or use a "repair" strategy.
   - Let's try a different angle: The "unsorted patterns" set is dynamic.
   - However, evaluating 120 comparators * 65536 inputs for every step is expensive if done naively.
   - We can pre-compute the effect of each comparator on the current state of all inputs? No, the state changes.
   - We need a fast way to estimate the benefit.
   - Alternatively, we can use the "Pruning" approach but start from a DIFFERENT baseline.
   - The prompt asks for a MATERIAL DIFFERENCE. Pruning is the current strategy.
   - Constructive is the proposed alternative.
   
   Let's refine the Constructive approach for speed:
   - We cannot afford to check all 120 comparators against 65536 inputs for every single step if we do 60 steps.
   - 60 steps * 120 comps * (65536 * 16 ops) is too much.
   - We need a faster surrogate.
   - Idea: Sample a subset of the 65536 inputs (e.g., 1024 random inputs) to guide the construction.
   - Once a candidate network is built using the surrogate, verify it against ALL 65536 inputs.
   - If it fails, try to repair it by adding more comparators or switching to pruning from a known good network.
   
   Actually, the most robust "different" strategy that is fast is:
   - Start with the Optimal 60-comparator network (if we had it hardcoded, but we don't in the seed).
   - Wait, the seed program generates Batcher's (63) and prunes.
   - Inspiration 3 mentions a "known optimal 16-wire topology".
   - If I can hardcode a 60-comparator network, I can just return it. But I don't have it memorized perfectly.
   - Let's stick to the "Constructive with Surrogate" idea.
   
   Revised Strategy:
   1. Generate a pool of candidate comparators (all 120 pairs).
   2. Initialize network = [].
   3. Maintain a set of "hard" inputs (a random sample of 2^16, e.g., 4096 inputs) that are currently unsorted.
   4. Loop until all sampled inputs are sorted or time runs out:
      a. For each candidate comparator (i,j), simulate its effect on the current network state for the hard inputs.
      b. Count how many hard inputs become sorted.
      c. Pick the comparator that maximizes this count (tie-break by random or structural heuristic).
      d. Add it to the network.
      e. Update the network state for the hard inputs.
   5. Verify the resulting network against ALL 65536 inputs.
   6. If valid, attempt to prune it (greedy removal) to reduce size further.
   7. If invalid, fallback to Batcher + Pruning (the current strategy) but with a tighter time budget for the constructive part.
   
   This is materially different: it builds UP rather than prunes DOWN, and uses a surrogate (sampled inputs) for speed during construction.
"""

import time
import numpy as np

N = 16
MAX_TIME = 58.0
SAMPLE_SIZE = 4096  # Number of inputs to use for the constructive surrogate

def _generate_batcher_network():
    """
    Generates Batcher's Odd-Even Merge Sort network for N=16.
    Returns a flat list of comparators.
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

def _is_sorted_network_vectorized(comps, bits):
    """
    Check if the network sorts all 2^16 boolean inputs.
    """
    out = _apply_network_vectorized(comps, bits)
    diffs = np.diff(out, axis=1)
    return np.all(diffs >= 0)

def _constructive_search(bits_sample, deadline, rng):
    """
    Attempts to build a sorting network from scratch using a greedy surrogate approach.
    Returns the list of comparators if successful, else None.
    """
    n_samples = bits_sample.shape[0]
    current_state = bits_sample.copy()
    network = []
    all_comps = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Track which samples are currently unsorted
    unsorted_mask = ~np.all(np.diff(current_state, axis=1) >= 0, axis=1)
    num_unsorted = np.sum(unsorted_mask)
    
    if num_unsorted == 0:
        return network
    
    max_comparators = 80  # Don't build too large
    start_time = time.time()
    
    while num_unsorted > 0 and len(network) < max_comparators and time.time() < deadline - 5.0:
        best_comp = None
        best_gain = -1
        best_new_state = None
        
        # To speed this up, we can't just loop through 120 comps and apply the whole network state change naively for each.
        # We need to simulate the effect of adding (i,j) to the CURRENT state.
        # current_state is the result of applying `network` to bits_sample.
        # If we add (i,j), the new state is just one step further.
        
        for comp in all_comps:
            i, j = comp
            # Simulate the compare-exchange on the current state
            new_state = current_state.copy()
            min_val = np.minimum(new_state[:, i], new_state[:, j])
            max_val = np.maximum(new_state[:, i], new_state[:, j])
            new_state[:, i] = min_val
            new_state[:, j] = max_val
            
            # Check which previously unsorted samples are now sorted
            # Note: A sample might be sorted before but become unsorted? No, compare-exchange preserves the property that if a sequence is sorted, it remains sorted? 
            # Actually, compare-exchange on a sorted sequence leaves it unchanged. So sorted samples remain sorted.
            # We only care about unsorted samples becoming sorted.
            
            new_diffs = np.diff(new_state, axis=1)
            new_sorted_mask = np.all(new_diffs >= 0, axis=1)
            
            # Gain is the number of samples that were unsorted and are now sorted
            # Since sorted samples stay sorted, gain = sum(new_sorted_mask & unsorted_mask)
            gain = np.sum(new_sorted_mask & unsorted_mask)
            
            if gain > best_gain:
                best_gain = gain
                best_comp = comp
                best_new_state = new_state
        
        if best_gain <= 0:
            # No single comparator can fix any more unsorted samples?
            # This might happen if the network is "stuck" or if the sample is too hard.
            # In a valid sorting network, there is always a comparator that helps if the input is unsorted?
            # Not necessarily for a partial network.
            # If we get stuck, we break and let the fallback handle it.
            break
        
        # Add the best comparator
        network.append(best_comp)
        current_state = best_new_state
        unsorted_mask = ~np.all(np.diff(current_state, axis=1) >= 0, axis=1)
        num_unsorted = np.sum(unsorted_mask)
        
        # Optional: Periodically verify against full set if we are close to done?
        # For now, trust the surrogate.
        
    # Final check against full set
    if time.time() < deadline - 1.0:
        if _is_sorted_network_vectorized(network, bits_all):
            return network
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Pre-compute all boolean inputs
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    bits_all = (indices[:, None] >> np.arange(N)[::-1]) & 1
    
    # Prepare sample for constructive search
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(n_inputs, size=SAMPLE_SIZE, replace=False)
    bits_sample = bits_all[sample_indices]
    
    best_comps = None
    best_len = 1000
    
    # Strategy 1: Constructive Search
    # We allocate 20 seconds to the constructive phase
    constructive_deadline = start_time + 20.0
    construct_result = _constructive_search(bits_sample, constructive_deadline, rng)
    
    if construct_result is not None:
        best_comps = construct_result
        best_len = len(best_comps)
        
        # Try to prune the constructed network
        i = len(best_comps) - 1
        while i >= 0 and time.time() < deadline - 5.0:
            if time.time() > deadline - 1.0:
                break
            comp = best_comps[i]
            best_comps.pop(i)
            if _is_sorted_network_vectorized(best_comps, bits_all):
                best_len = len(best_comps)
            else:
                best_comps.insert(i, comp)
                i -= 1
    else:
        # Strategy 2: Fallback to Batcher + Pruning (Current Strategy)
        # Allocate remaining time to this
        current_comps = _generate_batcher_network()
        best_comps = current_comps[:]
        best_len = len(best_comps)
        
        # Pruning Pass 1: Reverse
        i = len(best_comps) - 1
        while i >= 0 and time.time() < deadline - 5.0:
            if time.time() > deadline - 1.0:
                break
            comp = best_comps[i]
            best_comps.pop(i)
            if _is_sorted_network_vectorized(best_comps, bits_all):
                best_len = len(best_comps)
            else:
                best_comps.insert(i, comp)
                i -= 1
        
        # Pruning Pass 2: Forward
        i = 0
        while i < len(best_comps) and time.time() < deadline - 5.0:
            if time.time() > deadline - 1.0:
                break
            comp = best_comps[i]
            best_comps.pop(i)
            if _is_sorted_network_vectorized(best_comps, bits_all):
                best_len = len(best_comps)
            else:
                best_comps.insert(i, comp)
                i += 1
        
        # Local Search
        all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
        while time.time() < deadline - 1.0:
            if time.time() > deadline - 1.0:
                break
            current_comps = best_comps[:]
            improved = False
            if len(current_comps) > 1:
                idx = rng.integers(0, len(current_comps))
                removed = current_comps.pop(idx)
                if _is_sorted_network_vectorized(current_comps, bits_all):
                    best_comps = current_comps
                    best_len = len(best_comps)
                    improved = True
                else:
                    current_comps.insert(idx, removed)
                    
            if not improved and len(current_comps) > 1:
                idx = rng.integers(0, len(current_comps))
                old_comp = current_comps[idx]
                new_comp = all_comps[rng.integers(0, len(all_comps))]
                if new_comp != old_comp:
                    current_comps[idx] = new_comp
                    if _is_sorted_network_vectorized(current_comps, bits_all):
                        best_comps = current_comps
                        # best_len unchanged
                        improved = True
                    else:
                        current_comps[idx] = old_comp

    return best_comps

# EVOLVE-BLOCK-END
