import time
import random
import numpy as np

N = 16
MAX_COMPARATORS = 4096
TIME_LIMIT = 85.0  # Safety margin for 90s limit

# Precompute all binary inputs (0 to 2^16 - 1) as uint32
ALL_INPUTS = np.arange(2**16, dtype=np.uint32)

def _verify_network_full(network):
    """
    Verify if the network sorts all 2^16 inputs.
    Returns True if valid, False otherwise.
    """
    if not network:
        return False
        
    states = ALL_INPUTS.copy()
    for i, j in network:
        bit_i = (states >> i) & 1
        bit_j = (states >> j) & 1
        # If bit_i is 1 and bit_j is 0, we need to swap.
        swap_mask = bit_i & (~bit_j) & 1
        mask = (swap_mask == 1)
        if np.any(mask):
            states[mask] ^= ((1 << i) | (1 << j))
            
    # Check if all states are sorted (0...01...1)
    # Let x = ~state. x must be of the form 2^m - 1.
    # This is true iff x & (x + 1) == 0.
    x = (~states) & 0xFFFF
    sorted_mask = (x & (x + 1)) == 0
    return np.all(sorted_mask)

def _get_unsorted_indices(network):
    """
    Returns a boolean mask of length 2^16 indicating which inputs are NOT sorted
    by the current network.
    """
    states = ALL_INPUTS.copy()
    for i, j in network:
        bit_i = (states >> i) & 1
        bit_j = (states >> j) & 1
        swap_mask = bit_i & (~bit_j) & 1
        mask = (swap_mask == 1)
        if np.any(mask):
            states[mask] ^= ((1 << i) | (1 << j))
            
    x = (~states) & 0xFFFF
    sorted_mask = (x & (x + 1)) == 0
    return ~sorted_mask

def _fallback_construct():
    """
    Returns the standard Odd-Even Transposition network.
    Valid sorting network with 120 comparators.
    """
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def construct():
    start_time = time.time()
    
    # 1. Initialize with a known valid network
    # Odd-Even Transposition is 120 comparators.
    best_network = _fallback_construct()
    best_score = len(best_network)
    
    if not _verify_network_full(best_network):
        return best_network

    # 2. Greedy Construction / Reduction Strategy
    # We will try to reduce the network by finding a set of comparators that are "redundant"
    # in a specific sense, or more effectively, we will attempt a "Greedy Removal" 
    # but guided by the "Unsorted Inputs" of the *remaining* network.
    
    # Actually, the most robust "different strategy" is to start from scratch or a 
    # different baseline and build up, OR to use a global search.
    # However, building from scratch with 16 wires is hard to get below 60 quickly.
    # 
    # Let's try a different angle: **Global Greedy Removal with Lookahead**.
    # The current program does random removal. 
    # This program will do:
    # 1. Calculate unsorted inputs for current net.
    # 2. For each comparator, calculate how many of those unsorted inputs it "fixes" 
    #    if we were to remove it? No, that's complex.
    #
    # Let's stick to the prompt's request for a "materially different" strategy.
    # The current one is Local Search (SA/Random).
    # New Strategy: **Greedy Set Cover Approximation on Unsorted Inputs**.
    # 
    # Wait, the network is already valid. We want to minimize size.
    # A comparator is "essential" if removing it makes the network invalid.
    # The current program tries to remove random ones.
    # 
    # A better heuristic: 
    # 1. Identify all "critical" comparators (removal causes invalidity).
    # 2. If no critical comparators, remove one.
    # 3. If there are critical comparators, we are stuck in a local minimum for single removals.
    #    To escape, we need to change the network structure.
    #
    # Let's try a **Simulated Annealing with a different move set**:
    # Move: "Swap endpoints". Instead of (i, j) -> (i', j'), try (i, j) -> (i, j+1) or (i-1, j).
    # This explores the space of valid networks more densely than random replacement.
    #
    # BUT, the prompt asks for a strategy that is "materially different".
    # Let's implement **Greedy Construction from Scratch** using the Zero-One Principle.
    # 
    # Algorithm:
    # 1. Start with empty network.
    # 2. While not sorted:
    #    a. Find the set of unsorted inputs.
    #    b. For every possible comparator (i, j), calculate how many unsorted inputs it would fix 
    #       (i.e., how many inputs become sorted if this comparator is added to the current net).
    #       Note: Adding a comparator to a non-sorting network might not make it sorting, 
    #       but it reduces the "distance" to sorted.
    #       Actually, we must ensure the network *eventually* sorts.
    #       The Zero-One Principle says if it sorts all binary inputs, it sorts all.
    #       So we just need to reach a state where all binary inputs are sorted.
    #    c. Pick the comparator that maximizes the reduction in unsorted inputs.
    #    d. Add it to the network.
    # 3. Once valid, try to remove comparators (greedy removal).
    
    # This "Build Up" strategy is fundamentally different from "Start with 120 and Remove".
    # It might find a more compact structure because it doesn't inherit the structure of Odd-Even.
    
    current_network = []
    current_unsorted = _get_unsorted_indices(current_network) # All are unsorted initially? 
    # Wait, empty network sorts only 00..0 and 11..1. 
    # Most are unsorted.
    
    # Optimization: We don't need to check all 120 comparators at every step if we are smart.
    # But with 65k inputs, checking 120 comparators is 120 * 65k ops. 
    # 120 * 65536 * 16 bits... 
    # NumPy vectorization makes this fast.
    
    # Precompute masks for all comparators?
    # For a comparator (i, j), the "fix" depends on the current state.
    
    # Let's implement the Greedy Builder.
    
    # To speed up, we can use a heuristic: 
    # Only consider comparators that involve wires that are currently "misordered" in the unsorted inputs.
    
    # However, for N=16, the space of comparators is small (120).
    # We can afford to check all of them every step.
    
    # Step 1: Build
    while time.time() - start_time < TIME_LIMIT * 0.5: # Use half time for building
        if not np.any(current_unsorted):
            break
            
        best_comparator = None
        best_gain = 0
        best_new_unsorted = None
        
        # Get current states to evaluate gain
        states = ALL_INPUTS.copy()
        for i, j in current_network:
            bit_i = (states >> i) & 1
            bit_j = (states >> j) & 1
            swap_mask = bit_i & (~bit_j) & 1
            mask = (swap_mask == 1)
            if np.any(mask):
                states[mask] ^= ((1 << i) | (1 << j))
        
        # For each possible comparator, calculate new unsorted count
        for i in range(N):
            for j in range(i+1, N):
                # Simulate adding (i, j)
                # We can do this vectorized?
                # It's easier to just copy states and apply (i, j)
                # But 120 copies of 65k uint32s is ~3MB * 120 = 360MB per iteration? 
                # Too slow/memory heavy if done naively in Python loop.
                
                # Let's optimize:
                # We only care about the inputs that are currently unsorted.
                # If an input is already sorted, adding a comparator might unsort it?
                # No, if a network is a prefix of a sorting network, adding comparators 
                # doesn't "unsort" sorted inputs? 
                # Actually, it can. Sorting networks are monotonic in a sense, 
                # but adding a comparator can change the order.
                # Example: 01 -> 10 (swap). If I have 01 (unsorted), I swap to 10 (sorted).
                # If I have 10 (sorted), and I swap, it becomes 01 (unsorted).
                # So we MUST check all inputs.
                
                # To make it fast, we can use the fact that we only need to check 
                # the comparators that are "plausible".
                # Or, we can just accept that this step is expensive and limit iterations.
                
                # Let's try a different approach: **Random Greedy**.
                # Instead of checking all 120, check a random subset of 10-20.
                # This reduces overhead.
                
                pass # Placeholder for loop body logic below
        
        # Revised Loop Body:
        # To avoid O(120 * 65k) per step, we sample.
        candidates = []
        # Sample 20 random comparators
        for _ in range(20):
            i = random.randint(0, N-2)
            j = random.randint(i+1, N-1)
            candidates.append((i, j))
            
        best_gain = -1
        best_cand = None
        
        for i, j in candidates:
            # Apply (i, j) to a copy of states
            temp_states = states.copy()
            bit_i = (temp_states >> i) & 1
            bit_j = (temp_states >> j) & 1
            swap_mask = bit_i & (~bit_j) & 1
            mask = (swap_mask == 1)
            if np.any(mask):
                temp_states[mask] ^= ((1 << i) | (1 << j))
            
            # Count unsorted
            x = (~temp_states) & 0xFFFF
            sorted_mask = (x & (x + 1)) == 0
            unsorted_count = np.sum(~sorted_mask)
            
            # Gain is reduction in unsorted count
            # We want to minimize unsorted_count, so maximize -unsorted_count
            gain = -unsorted_count
            if gain > best_gain:
                best_gain = gain
                best_cand = (i, j)
                best_new_unsorted = ~sorted_mask # Store for next iteration if we pick this? 
                # No, we need to re-calculate unsorted for the whole network anyway.
        
        if best_cand:
            current_network.append(best_cand)
            # Recalculate unsorted mask for the new network
            current_unsorted = _get_unsorted_indices(current_network)
            
            # Early exit if valid
            if not np.any(current_unsorted):
                break

    # If we built a network, verify it
    if current_network and _verify_network_full(current_network):
        if len(current_network) < best_score:
            best_network = current_network[:]
            best_score = len(current_network)
    else:
        # If building failed or was too slow, fall back to the 120-comparator net
        current_network = best_network[:]

    # 3. Local Search Refinement (Greedy Removal)
    # Now that we have a valid network (either built or fallback), try to remove.
    # This part is similar to the original but now starts from a potentially better base.
    
    while time.time() - start_time < TIME_LIMIT:
        progress = False
        # Try to remove random comparators
        for _ in range(10):
            if not current_network:
                break
            idx = random.randint(0, len(current_network) - 1)
            cand = current_network[:idx] + current_network[idx+1:]
            if _verify_network_full(cand):
                current_network = cand
                if len(current_network) < best_score:
                    best_network = current_network[:]
                    best_score = len(current_network)
                progress = True
                break
        
        if progress:
            continue
            
        # If stuck, try mutation (endpoint swap)
        if current_network:
            idx = random.randint(0, len(current_network) - 1)
            i, j = current_network[idx]
            
            # Try swapping one endpoint
            # e.g., (i, j) -> (i, j+1) or (i-1, j)
            new_i, new_j = i, j
            if random.random() < 0.5:
                if j + 1 < N:
                    new_j = j + 1
                elif i > 0:
                    new_i = i - 1
            else:
                if i > 0:
                    new_i = i - 1
                elif j + 1 < N:
                    new_j = j + 1
            
            if (new_i, new_j) == (i, j) or new_i >= new_j:
                continue
                
            cand = current_network[:]
            cand[idx] = (new_i, new_j)
            
            if _verify_network_full(cand):
                current_network = cand

    return best_network

# Ensure imports are at the top for the final block
import time
import random
import numpy as np
