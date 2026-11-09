"""Size-efficient sorting network on 18 wires via fast bitset verification and local search."""
import time
import random
import numpy as np

N = 18

# EVOLVE-BLOCK-START

def _precompute_targets():
    """
    Precompute the target bitset for each wire.
    Target_i has bit 'm' set if and only if the sorted output for input 'm' 
    has a 1 on wire i.
    Wire 0 is the minimum, Wire N-1 is the maximum.
    For input m with k ones, sorted output has 0s in positions 0..N-1-k and 1s in N-k..N-1.
    So wire i is 1 iff k >= N - i.
    """
    targets = [0] * N
    # Optimization: Group inputs by popcount?
    # 2^18 = 262144.
    # We can iterate through all masks.
    # To speed up popcount, we can use a lookup table or just bin(m).count('1') which is fast enough for 262k calls.
    # 262k * ~100ns = ~26ms. Acceptable.
    
    # Even faster:
    # targets[N-1] corresponds to k >= 1. All masks except 0.
    # targets[N-2] corresponds to k >= 2.
    # ...
    # targets[0] corresponds to k >= 18. Only mask (1<<18)-1.
    
    # Let's just loop.
    for m in range(1 << N):
        k = m.bit_count() # Python 3.8+ has int.bit_count()
        # If k >= N - i, set bit m in targets[i]
        # This is O(N) per m. 18 * 262144 = 4.7M ops.
        # We can optimize by noting that if k >= N - i, then k >= N - (i+1) is also true? No.
        # If k is large, it satisfies many targets.
        # targets[i] |= (1 << m) if k >= N - i
        
        # Let's update targets in reverse order to be efficient?
        # Actually, just do the check.
        for i in range(N):
            if k >= N - i:
                targets[i] |= (1 << m)
                
    return targets

# Precompute targets at module level to avoid doing it in construct()
_TARGETS = _precompute_targets()

def _verify_network(net):
    """
    Verify if the network sorts all 2^N Boolean inputs using the 0-1 principle.
    Uses Python integers as bitsets for speed.
    """
    if not net:
        # Empty network is only valid if N <= 1
        return N <= 1
        
    # Initial state:
    # Wire i has bit m set iff bit i of m is 1.
    # We can generate these efficiently.
    # states[i] is an integer.
    states = [0] * N
    
    # Optimization for initial state generation:
    # states[i] has bits set at positions m where m has i-th bit set.
    # This is a pattern of 0s and 1s.
    # For i=0: 010101...
    # For i=1: 00110011...
    # We can generate these using masks or loops.
    # Looping 2^N times for each wire is 18 * 262144 = 4.7M ops.
    # Let's do it once.
    
    # Actually, we can just use the bitset logic.
    # For each m in 0..2^N-1, for each i, if m has bit i, set bit m in states[i].
    # This is the same cost as target precomputation.
    # Since we call verify many times, we should cache the initial states?
    # No, the initial states are constant.
    
    # Let's precompute initial states globally too.
    # But wait, if we precompute initial states, we still have to copy them for each verification.
    # Copying 18 integers is fast.
    
    # Let's precompute INITIAL_STATES.
    
    # To avoid global state complexity in the snippet, I'll compute it in _verify_network
    # but cache it in a global variable.
    
    global _INITIAL_STATES
    if '_INITIAL_STATES' not in globals() or _INITIAL_STATES is None:
        _INITIAL_STATES = [0] * N
        for m in range(1 << N):
            for i in range(N):
                if (m >> i) & 1:
                    _INITIAL_STATES[i] |= (1 << m)
    
    states = _INITIAL_STATES[:] # Copy
    
    # Apply network
    for i, j in net:
        # min goes to i, max goes to j
        # For bits: min = AND, max = OR
        si = states[i]
        sj = states[j]
        states[i] = si & sj
        states[j] = si | sj
        
    # Check against targets
    for i in range(N):
        if states[i] != _TARGETS[i]:
            return False
            
    return True


def construct():
    """
    Construct a sorting network on 18 wires using local search with fast verification.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Start with odd-even transposition network
    def odd_even_transposition():
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    best_net = odd_even_transposition()
    
    # Verify seed
    if not _verify_network(best_net):
        # Fallback if something went wrong, though OET is always valid
        return best_net
        
    # Main Search Loop
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Greedy Removal with Randomized Order
        # Try to remove comparators one by one.
        # Randomize order to escape local minima related to removal order.
        indices = list(range(len(best_net)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            if idx >= len(best_net):
                continue
            
            candidate = best_net[:idx] + best_net[idx+1:]
            if _verify_network(candidate):
                best_net = candidate
                improved = True
                break  # Restart the loop with the new shorter network
        
        if improved:
            continue
            
        # Phase 2: Local Search (Swaps)
        # If no removals were possible, try swapping adjacent comparators
        # that share a wire to change the network structure.
        # This might enable removals in the next Phase 1 iteration.
        
        # Find candidate swaps
        valid_swaps = []
        for i in range(len(best_net) - 1):
            w1 = best_net[i]
            w2 = best_net[i+1]
            # Check if they share a wire
            if len({w1[0], w1[1], w2[0], w2[1]}) < 4:
                valid_swaps.append(i)
                
        if not valid_swaps:
            # No overlapping adjacent comparators. 
            # This might mean we are in a "parallel" layer or similar.
            # Try swapping non-adjacent? No, that's complex.
            # If no swaps and no removals, we are stuck.
            # Break to avoid infinite loop.
            break
            
        # Pick a random swap
        swap_idx = random.choice(valid_swaps)
        
        # Perform swap
        new_net = best_net[:]
        new_net[swap_idx], new_net[swap_idx + 1] = new_net[swap_idx + 1], new_net[swap_idx]
        
        # Check if this swap results in a valid network (it should, as swaps of overlapping comparators are just a different valid sequence)
        # Actually, we don't need to verify validity of the swap itself if we are going to try removals from it.
        # But we must ensure the swapped network is valid before trying to remove from it?
        # Yes, we only work with valid networks.
        # Swapping two comparators in a valid network results in another valid network?
        # Not necessarily! 
        # Example: (0,1), (1,2). Swap -> (1,2), (0,1).
        # Both are valid sorting networks? 
        # OET is valid. Swapping adjacent comparators in OET might break it?
        # No, the 0-1 principle says if it sorts all 0-1 inputs, it's valid.
        # Does swapping comparators preserve the property of sorting 0-1 inputs?
        # No! A specific sequence of comparators is a function. Changing the sequence changes the function.
        # So we MUST verify the swapped network.
        
        if _verify_network(new_net):
            # Try to remove from this new structure
            # We do a quick greedy removal pass on new_net
            temp_net = new_net
            temp_improved = False
            temp_indices = list(range(len(temp_net)))
            random.shuffle(temp_indices)
            
            for idx in temp_indices:
                if time.time() > deadline:
                    break
                if idx >= len(temp_net):
                    continue
                candidate = temp_net[:idx] + temp_net[idx+1:]
                if _verify_network(candidate):
                    temp_net = candidate
                    temp_improved = True
                    break
            
            if temp_improved:
                best_net = temp_net
                # Loop back to Phase 1
                continue
            else:
                # Swap didn't lead to immediate removal.
                # We can accept the swap to explore further, or revert.
                # Let's accept it to change the landscape, but we need to be careful not to cycle.
                # Since we only accept if it leads to a removal, we effectively only move if we get smaller.
                # If we don't get smaller, we revert.
                pass # Revert by not updating best_net
        
    return best_net

# EVOLVE-BLOCK-END
