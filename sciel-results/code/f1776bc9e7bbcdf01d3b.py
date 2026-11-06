import sys
import numpy as np
import time
import random

N = 18

def construct():
    """
    Constructs a sorting network for N wires using a local search heuristic 
    starting from the Odd-Even Transposition network.
    Uses NumPy for fast verification against all 2^N Boolean inputs.
    """
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer

    # 1. Generate Seed: Odd-Even Transposition Network
    # Phases: 0..N-1
    # Phase p: comparators (i, i+1) for i in range(p%2, N-1, 2)
    def generate_odd_even_transposition(n):
        comps = []
        for phase in range(n):
            for i in range(phase % 2, n - 1, 2):
                comps.append((i, i + 1))
        return comps

    current_net = generate_odd_even_transposition(N)
    
    # 2. Fast Verification using NumPy
    # Precompute the initial state matrix for all 2^N inputs
    # Shape: (2^N, N)
    # Input k, wire j = (k >> j) & 1
    # We can generate this efficiently.
    
    def verify_network(comps):
        """
        Returns True if the network sorts all 2^N Boolean inputs.
        """
        # Generate all 2^N inputs
        # np.arange(2**N) is 262144 elements.
        # Reshape to bits.
        # Using np.unpackbits is efficient if we handle endianness correctly, 
        # but simple bit shifting is also fast enough for 262k * 18.
        
        # Let's use a vectorized approach for the simulation.
        # State matrix: (2**N, N)
        # Initial state:
        #   inputs = np.arange(2**N)
        #   bits = np.zeros((2**N, N), dtype=np.uint8)
        #   for j in range(N):
        #       bits[:, j] = (inputs >> j) & 1
        
        # This loop is N=18 iterations, very fast.
        
        inputs = np.arange(1 << N, dtype=np.uint64)
        bits = np.zeros((1 << N, N), dtype=np.uint8)
        for j in range(N):
            bits[:, j] = (inputs >> j) & 1
        
        # Apply comparators
        for i, j in comps:
            # min goes to i, max goes to j
            # min_val = np.minimum(bits[:, i], bits[:, j])
            # max_val = np.maximum(bits[:, i], bits[:, j])
            # bits[:, i] = min_val
            # bits[:, j] = max_val
            
            # Vectorized swap if needed
            # Since it's boolean, min is AND, max is OR?
            # No, min(0,1)=0, max(0,1)=1.
            # min(0,0)=0, max(0,0)=0.
            # min(1,1)=1, max(1,1)=1.
            # So min is bitwise AND, max is bitwise OR.
            
            # bits[:, i] &= bits[:, j]  # This is wrong if we do it in place sequentially without temp
            # We need temp vars.
            
            tmp_i = bits[:, i].copy()
            tmp_j = bits[:, j].copy()
            
            # min
            bits[:, i] = np.minimum(tmp_i, tmp_j)
            # max
            bits[:, j] = np.maximum(tmp_i, tmp_j)
            
        # Check if sorted: bits[:, 0] <= bits[:, 1] <= ... <= bits[:, N-1]
        # For boolean, this means no 1 followed by 0.
        # i.e., bits[:, :-1] <= bits[:, 1:]
        
        # Check if all rows are sorted
        # sorted_rows = np.all(bits[:, :-1] <= bits[:, 1:], axis=1)
        # return np.all(sorted_rows)
        
        # Optimization: Check for any inversion
        # inversions = (bits[:, :-1] > bits[:, 1:])
        # if np.any(inversions): return False
        # return True
        
        return not np.any(bits[:, :-1] > bits[:, 1:])

    # 3. Local Search: Greedy Removal
    # Start with the seed network.
    # Iteratively try to remove comparators.
    # To avoid getting stuck, we can shuffle the order of attempts.
    # We also track the best network found.
    
    best_net = current_net[:]
    best_count = len(current_net)
    
    # Verify seed (should be true)
    if not verify_network(current_net):
        # Fallback to a simpler known good network if seed fails (unlikely)
        # But OET is always valid.
        pass

    # Pre-compute verification costs? No, just run it.
    # 262k * 150 comps * 2 ops = ~78M ops. NumPy handles this in ~10-50ms.
    # So we can do ~1000-5000 verifications in 55s.
    # With 153 comparators, we can try removing each one multiple times.
    
    # Strategy:
    # 1. Create a list of indices for current_net.
    # 2. Shuffle indices.
    # 3. Iterate through indices. For each index, try to remove the comparator.
    #    If valid, remove it from the list.
    # 4. Repeat until time runs out.
    
    # To improve efficiency, we can keep a "dirty" flag or just re-verify.
    # Re-verification is safest.
    
    # We will use a "pool" of comparators and try to shrink it.
    # Actually, simple in-place removal is fine.
    
    # Let's add some randomness to escape local minima if we get stuck?
    # If no removal is successful in a full pass, we might try swapping?
    # Swapping is harder to verify.
    # Let's stick to removal first. If stuck, try random restarts?
    # Restarting from scratch is expensive.
    # Instead, we can try "mutating" the network:
    # - Swap two comparators.
    # - Insert a random comparator.
    # - Remove a random comparator.
    
    # But removal is the only way to reduce count.
    # So we focus on removal.
    
    # To make it more robust, we can run multiple passes.
    # In each pass, we shuffle the order of comparators and try to remove each.
    
    while time.time() - start_time < time_limit:
        # Shuffle the current network indices
        indices = list(range(len(current_net)))
        random.shuffle(indices)
        
        removed_any = False
        
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
            
            # Try to remove current_net[idx]
            # Create a copy without this comparator
            # To avoid copying the whole list every time (expensive),
            # we can just simulate the removal?
            # No, we need to verify the specific sequence.
            # Copying a list of ~100 ints is fast.
            
            test_net = current_net[:idx] + current_net[idx+1:]
            
            if verify_network(test_net):
                # Success! Remove it permanently.
                current_net = test_net
                removed_any = True
                # Update best if this is better
                if len(current_net) < best_count:
                    best_count = len(current_net)
                    best_net = current_net[:]
                    
                # Since we removed an item, indices shift.
                # We are iterating over a static list of indices from the start of the loop.
                # If we remove index `idx`, the items after it shift left.
                # The `indices` list is stale.
                # However, we are just trying to remove *some* comparator.
                # If we remove one, the next index in `indices` might point to a different comparator now.
                # This is okay for a heuristic, but might skip some.
                # To be more precise, we should restart the loop or adjust indices.
                # Given the low cost of verification, restarting the inner loop is safer and simpler.
                # But restarting means we might try the same "easy" removals first again.
                # Let's just break and restart the outer while loop to re-shuffle.
                # This ensures we always try random orders.
                
                # Optimization: Instead of breaking, we can just continue, but we need to be careful with indices.
                # If we break, we re-shuffle and try again. This is a standard "hill climbing" step.
                break
        
        if not removed_any:
            # We couldn't remove any more comparators in this pass.
            # We are stuck in a local minimum.
            # Strategy: Try a different seed or a different mutation?
            # For now, we can try to "perturb" the network.
            # But perturbation usually increases size or breaks validity.
            # 
            # Alternative: Try a different initial seed?
            # We can generate a Bitonic Sort network and try to reduce that.
            # Bitonic sort for N=16 is 48 comparators? 
            # Bitonic sort for N=18 is not standard.
            # 
            # Let's try to generate a Shellsort-like network?
            # Or just accept the local minimum if time is running out.
            # 
            # To explore more, we can try to **swap** two comparators?
            # Swapping doesn't change the count, but might allow further removals?
            # Unlikely to help with count directly.
            
            # Let's try a "random restart" with a different seed.
            # Generate a random valid network? Hard.
            
            # Instead, let's try to **insert** a comparator and then remove others?
            # No, we want to minimize.
            
            # Let's try a different seed: Batcher's Odd-Merge?
            # I will implement a simple Batcher's Odd-Merge generator.
            
            if best_count > 80: # If we are still far from the best known (77), try another seed
                # Generate Batcher's Odd-Merge for N=18
                # This is complex to implement correctly from scratch in a short time.
                # Let's stick to OET and hope the local search is good enough.
                # Or use a hardcoded known good network?
                # The prompt says "do not reuse current designs" but I can use standard algorithms.
                # 
                # Let's try to generate a network using a simple heuristic:
                # "Shellsort" with specific gaps.
                # Gaps: 9, 4, 2, 1.
                # Gap 9: (0,9), (1,10), ..., (8,17)
                # Gap 4: (0,4), (1,5), ..., (13,17)
                # Gap 2: (0,2), (1,3), ..., (15,17)
                # Gap 1: (0,1), (1,2), ..., (16,17)
                # Total: 9 + 14 + 9 + 17 = 49.
                # Is this a sorting network?
                # Shellsort with these gaps is NOT guaranteed to sort all inputs as a fixed network unless it's a specific "Shellsort network" which is rare.
                # Standard Shellsort is an adaptive algorithm. A fixed network must handle all inputs.
                # 
                # Let's try **Bitonic Sort** for N=16 and handle 2 extra wires?
                # Bitonic Sort for N=16:
                # It is a valid sorting network.
                # We can take the N=16 network, and add 2 wires that are sorted against the 16?
                # This is complex.
                
                # Let's just continue with OET local search. 
                # If we get stuck, we can try to **randomly swap** two comparators 
                # to see if it unlocks new removals? 
                # This is a "sideways" move.
                
                # Let's try one random swap per pass if stuck.
                if len(current_net) > 1:
                    i, j = random.sample(range(len(current_net)), 2)
                    current_net[i], current_net[j] = current_net[j], current_net[i]
                    # Don't verify swap, just hope it helps next removal pass?
                    # No, swap must maintain validity.
                    # So we must verify the swap.
                    if verify_network(current_net):
                        # Swap was valid. Continue to next pass.
                        pass
                    else:
                        # Swap was invalid, revert.
                        current_net[i], current_net[j] = current_net[j], current_net[i]

    return best_net

# EVOLVE-BLOCK-END
