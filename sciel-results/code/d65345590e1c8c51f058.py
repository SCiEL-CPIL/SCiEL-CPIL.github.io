"""
Strategy:
1. Generate a valid Batcher Odd-Even Merge Sort network (63 comparators).
2. Use a highly optimized vectorized validity check (Zero-One Principle).
   - Instead of checking all 2^16 inputs (expensive), we can use a smaller set of "witness" inputs 
     if we are certain, but for safety and correctness certification, we must check all.
   - However, checking all 65536 inputs for ~60 comparators is ~3.9M operations per check.
   - With NumPy, this is fast (~1-2ms per check).
   - 58 seconds / 0.002s = ~29,000 checks possible.
3. Systematic Reduction Phase:
   - Iterate through the list of comparators from index 0 to N-1.
   - For each comparator, try to remove it.
   - If the network remains valid, keep it removed and continue.
   - If invalid, restore it and move to the next.
   - This guarantees we find the minimal network reachable by a sequence of single removals 
     in a specific order. It is much more effective than random removal because it doesn't 
     miss "easy" removals that might be hidden by the order of random selection.
4. If a full pass yields no removals (stuck at a local minimum for single removals), 
   we can attempt a limited number of random swaps to escape, but the primary power 
   comes from the systematic pass.
   - Actually, a better escape mechanism is to try removing a comparator and *simultaneously* 
     adding a different one (a 1-swap), but that's expensive.
   - Given the time budget, the systematic removal is the key. Batcher's 63 is very close 
     to the 60 reference. It is highly likely that 3 comparators are redundant in a way 
     that allows sequential removal.

Implementation Details:
- _generate_batcher_network(): Standard implementation.
- _is_valid(comps, bits): Vectorized check.
- construct():
  - Precompute bits.
  - Generate Batcher.
  - Loop:
    - For each index i in current network:
      - Remove comp[i].
      - Check validity.
      - If valid: accept removal (list shrinks).
      - Else: restore.
    - If no removals in a full pass:
      - Try a few random swaps (remove i, add j) to break ties.
      - If time is low, stop.
"""

import time
import numpy as np

N = 16
MAX_TIME = 58.0

def _generate_batcher_network():
    """
    Generates Batcher's Odd-Even Merge Sort network for N=16.
    """
    def merge(n, offset):
        if n == 1:
            return [(offset, offset + 1)]
        
        h = n // 2
        step1 = [(offset + i, offset + i + n) for i in range(n)]
        
        def strided_merge(n, offset, stride):
            if n == 1:
                return [(offset, offset + stride)]
            
            h = n // 2
            step1 = [(offset + i * stride, offset + (n + i) * stride) for i in range(n)]
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

def _is_valid_network(comps, bits):
    """
    Check if the network sorts all 2^N boolean inputs.
    """
    if not comps:
        return True
        
    vec = bits.copy()
    for i, j in comps:
        min_val = np.minimum(vec[:, i], vec[:, j])
        max_val = np.maximum(vec[:, i], vec[:, j])
        vec[:, i] = min_val
        vec[:, j] = max_val
        
    # Check if sorted: diffs should be >= 0
    diffs = np.diff(vec, axis=1)
    return np.all(diffs >= 0)

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Pre-compute all boolean inputs
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    bits = (indices[:, None] >> np.arange(N)[::-1]) & 1
    
    # 1. Generate initial network (Batcher's, 63 comparators)
    best_comps = _generate_batcher_network()
    
    # Verify initial network (should be valid)
    if not _is_valid_network(best_comps, bits):
        # Fallback to Odd-Even Transposition (120 comps) if Batcher generation is buggy
        best_comps = []
        for i in range(N):
            if i % 2 == 0:
                for j in range(0, N - 1, 2):
                    best_comps.append((j, j + 1))
            else:
                for j in range(1, N - 1, 2):
                    best_comps.append((j, j + 1))
    
    rng = np.random.default_rng(42)
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Main Search Loop
    # We use a "Systematic Removal" strategy primarily.
    # This is deterministic and ensures we don't miss removals due to random ordering.
    
    while time.time() < deadline - 2.0:  # Leave 2s buffer
        improved = False
        current_len = len(best_comps)
        
        # 1. Systematic Removal Pass
        # We iterate through the list. If we remove an item, the indices shift, 
        # so we restart the pass or handle carefully. 
        # Restarting is safer and simpler. 
        # Since we expect few removals (63->60), restarting is cheap.
        
        for _ in range(current_len):
            if time.time() > deadline - 2.0:
                break
                
            # Try to remove each comparator in the current list
            # We iterate from 0 to len-1
            found_removal = False
            for i in range(len(best_comps)):
                if time.time() > deadline - 2.0:
                    break
                    
                # Temporarily remove
                removed = best_comps.pop(i)
                
                # Check validity
                if _is_valid_network(best_comps, bits):
                    # Valid! Keep it removed.
                    improved = True
                    found_removal = True
                    # Restart the outer loop logic immediately by breaking inner loop 
                    # and letting the outer 'for _ in range' restart with new length
                    break
                else:
                    # Invalid, restore
                    best_comps.insert(i, removed)
            
            if found_removal:
                # If we found a removal, restart the systematic pass with the new shorter list
                # This ensures we check the new configuration thoroughly
                break
            
            if not found_removal:
                # No removals found in a full pass. 
                # We are stuck in a local minimum for single-removal moves.
                # Try a few random swaps to escape.
                # A swap is: remove one, add another.
                # We limit the number of swaps to avoid wasting time.
                for _ in range(10):
                    if time.time() > deadline - 2.0:
                        break
                    
                    if len(best_comps) < 2:
                        break
                    
                    idx = rng.integers(0, len(best_comps))
                    old_comp = best_comps[idx]
                    new_comp = all_comps[rng.integers(0, len(all_comps))]
                    
                    if new_comp != old_comp:
                        best_comps[idx] = new_comp
                        if _is_valid_network(best_comps, bits):
                            # Valid swap found. This might open up new removals.
                            improved = True
                            # Break out of swap loop to restart systematic removal
                            break
                        else:
                            best_comps[idx] = old_comp
                
                if improved:
                    # Restart systematic removal
                    continue
                else:
                    # No swaps worked either. 
                    # If we are at the known optimum (e.g., 60 or 55), we might be stuck.
                    # For now, if we can't improve, we stop to save time, 
                    # unless we have a lot of time left and want to try more random restarts.
                    # Given the time limit, stopping is safer.
                    # However, we can try a "perturbation": add a random comparator, 
                    # then try to remove a different one.
                    
                    # Try 1-Add-1-Remove (2-opt style)
                    if len(best_comps) < 100: # Don't bloat too much
                        new_comp = all_comps[rng.integers(0, len(all_comps))]
                        if new_comp not in best_comps:
                            best_comps.append(new_comp)
                            # Now try to remove a random comparator (not the one we just added)
                            for _ in range(5):
                                if time.time() > deadline - 2.0:
                                    break
                                idx = rng.integers(0, len(best_comps)-1) # Exclude the last one we added? 
                                                                        # Actually, we can remove any.
                                old_comp = best_comps.pop(idx)
                                if _is_valid_network(best_comps, bits):
                                    # Valid! We added one, removed one. Length is same.
                                    # But the structure changed. Restart systematic removal.
                                    improved = True
                                    break
                                else:
                                    best_comps.insert(idx, old_comp)
                            
                            if improved:
                                continue
                            else:
                                # Restore the network to the state before the perturbation
                                # by removing the added comparator if we didn't find a valid state
                                # Actually, if the loop finished without 'improved', 
                                # best_comps might have the added comp if we didn't pop it correctly?
                                # In the loop above, if invalid, we insert back.
                                # So best_comps is back to original + added comp.
                                # We need to remove the added comp to return to the previous valid state.
                                if new_comp in best_comps:
                                    best_comps.remove(new_comp)
                                # Now we are back to the stuck state. 
                                # If we can't escape, we stop.
                                break
                    else:
                        break

    return best_comps

# EVOLVE-BLOCK-END
