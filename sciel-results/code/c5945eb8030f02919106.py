"""
Size-efficient sorting networks on 17 wires.
Strategy:
1. Start with a known valid network (Odd-Even Transposition, 136 comparators).
2. Use vectorized NumPy verification for speed.
3. Perform aggressive local search:
   a. Removal: Try removing each comparator.
   b. Relocation: Try moving a comparator to a different position.
   c. Replacement: Try replacing a comparator with a different pair (i, j).
4. Iterate until no improvement or time limit reached.
"""

import numpy as np
import time
import random

N = 17
TIME_LIMIT = 55.0  # Seconds

def construct():
    # Precompute all 2^N input masks
    all_masks = np.arange(1 << N, dtype=np.uint32)
    # Precompute the initial bit matrix for all masks
    # bits[k, i] is the i-th bit of mask k
    bits_template = (all_masks[:, None] >> np.arange(N, dtype=np.uint8)) & 1
    
    # Sorted masks check:
    # A mask is sorted if it is of the form 0...01...1
    # We can precompute the set of sorted masks, or just check the property.
    # Property: for all i < j, bit_i <= bit_j.
    # Equivalent: no '10' pattern.
    # In the bit matrix, if we look at columns i and i+1, we shouldn't have (1, 0).
    # Actually, simpler: count the number of 1s. If the mask is sorted,
    # the 1s must be in the highest positions.
    # Let's just use the diff method on the final bits matrix.
    
    def verify_network(network):
        """
        Returns True if the network sorts all inputs.
        """
        if len(network) == 0:
            return N == 1 # Trivial case, but N=17 so False
        
        # Copy the template
        bits = bits_template.copy()
        
        for (i, j) in network:
            bi = bits[:, i]
            bj = bits[:, j]
            # Swap if bi=1 and bj=0
            swap_mask = (bi == 1) & (bj == 0)
            if np.any(swap_mask):
                bits[swap_mask, i] = 0
                bits[swap_mask, j] = 1
        
        # Check if sorted
        # Check for any '10' pattern in adjacent columns
        # If bits[:, k] == 1 and bits[:, k+1] == 0, it's not sorted.
        # We can check this for all k from 0 to N-2.
        # Vectorized: 
        # left = bits[:, :-1]
        # right = bits[:, 1:]
        # invalid = (left == 1) & (right == 0)
        # return not np.any(invalid)
        
        # This is fast.
        left = bits[:, :-1]
        right = bits[:, 1:]
        invalid = (left == 1) & (right == 0)
        return not np.any(invalid)

    # Initialize with Odd-Even Transposition Network
    # Phase p: compare (p%2, p%2+2, ...), (p%2+1, p%2+3, ...)
    # For N=17, we need 17 phases?
    # Standard OET:
    # for phase in range(N):
    #   for i in range(phase % 2, N - 1, 2):
    #       network.append((i, i+1))
    # Let's verify this logic.
    # Phase 0: (0,1), (2,3), ..., (14,15) -> 8 comparators
    # Phase 1: (1,2), (3,4), ..., (15,16) -> 8 comparators
    # ...
    # Total: 8 * 17 = 136. Correct.
    
    def generate_seed_network():
        net = []
        for phase in range(N):
            start = phase % 2
            for i in range(start, N - 1, 2):
                net.append((i, i + 1))
        return net

    network = generate_seed_network()
    
    # Verify seed
    if not verify_network(network):
        # Should not happen, but fallback
        return network

    start_time = time.time()
    improved = True
    best_size = len(network)
    
    # Main search loop
    while improved and (time.time() - start_time) < TIME_LIMIT:
        improved = False
        current_size = len(network)
        
        # 1. Try Removal
        # Iterate in reverse to avoid index shifting issues if we remove
        for idx in range(len(network) - 1, -1, -1):
            if (time.time() - start_time) > TIME_LIMIT:
                break
            
            candidate = network[:idx] + network[idx+1:]
            if verify_network(candidate):
                network = candidate
                improved = True
                best_size = len(network)
                break # Restart loop with new network
        
        if improved:
            continue
            
        # 2. Try Relocation
        # Move comparator at idx to position new_idx
        # We only try a limited number of moves to save time
        moves_tried = 0
        max_moves = 50 # Limit per iteration
        
        for idx in range(len(network)):
            if (time.time() - start_time) > TIME_LIMIT:
                break
            comp = network[idx]
            # Try moving to a few random positions or nearby positions
            # Let's try nearby first: idx-1, idx+1
            for delta in [-1, 1]:
                new_idx = idx + delta
                if 0 <= new_idx < len(network) and new_idx != idx:
                    candidate = network.copy()
                    candidate.pop(idx)
                    candidate.insert(new_idx, comp)
                    if verify_network(candidate):
                        network = candidate
                        improved = True
                        best_size = len(network)
                        break
            if improved:
                break
                
            # If nearby didn't work, try a random position
            if moves_tried < max_moves:
                new_idx = random.randint(0, len(network) - 1)
                if new_idx != idx:
                    candidate = network.copy()
                    candidate.pop(idx)
                    candidate.insert(new_idx, comp)
                    if verify_network(candidate):
                        network = candidate
                        improved = True
                        best_size = len(network)
                        break
                    moves_tried += 1
            if improved:
                break

        if improved:
            continue

        # 3. Try Replacement
        # Replace comparator at idx with a different (i, j)
        # This is more expensive. We'll try a few random replacements.
        replacements_tried = 0
        max_replacements = 20
        
        for idx in range(len(network)):
            if (time.time() - start_time) > TIME_LIMIT:
                break
            
            # Try replacing with a random valid pair
            for _ in range(5): # 5 random attempts per index
                if (time.time() - start_time) > TIME_LIMIT:
                    break
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                new_comp = (i, j)
                if new_comp == network[idx]:
                    continue
                
                candidate = network.copy()
                candidate[idx] = new_comp
                if verify_network(candidate):
                    network = candidate
                    improved = True
                    best_size = len(network)
                    break
                replacements_tried += 1
                if replacements_tried > max_replacements:
                    break
            if improved:
                break
            if replacements_tried > max_replacements:
                break

    return network

# Ensure the function is called correctly if executed directly
if __name__ == "__main__":
    net = construct()
    print(f"Found network with {len(net)} comparators")
