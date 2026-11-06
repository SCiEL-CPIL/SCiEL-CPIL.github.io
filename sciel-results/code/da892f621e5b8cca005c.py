import time
import random

N = 17
TOTAL_INPUTS = 1 << N  # 131072

def _build_initial_masks():
    """
    Builds bitmask representation for the 0-1 principle.
    mask[k] is an integer where bit m is set iff input m has a 1 on wire k.
    Input m is represented by its binary representation of m (0 to 2^N-1).
    """
    masks = [0] * N
    # Optimization: Use bit tricks to build masks faster.
    # mask[k] has bits set at positions m where the k-th bit of m is 1.
    # This is a regular pattern.
    # For k=0: 010101... -> bits 1, 3, 5...
    # For k=1: 00110011... -> bits 2, 3, 6, 7...
    # We can construct these using arithmetic or precomputation.
    # Simple loop is 17 * 131072 = 2.2M iterations. In Python this is ~0.2-0.5s.
    # Let's try to optimize with list comprehension or numpy if available, but stdlib is safer.
    # Actually, we can build mask[k] by iterating m.
    # A faster way in pure Python:
    # mask[k] = sum(1 << m for m in range(TOTAL_INPUTS) if (m >> k) & 1)
    # This is still a loop.
    
    # Let's use a recursive/iterative doubling approach?
    # No, simple loop is fine if we can make the inner loop fast.
    
    # Alternative: 
    # mask[0] = 0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA (for 64 bits, but we need 131072 bits)
    # This is a huge integer.
    
    # Let's stick to the loop but optimize it.
    for k in range(N):
        m = 1 << k
        # Iterate through blocks
        # Bit k is 1 for m in [2^k, 2^(k+1)-1], [2^(k+2), 2^(k+3)-1], ...
        # This is complex to generate directly as a bitmask integer efficiently without loops.
        
    # Let's just use the loop but maybe process in chunks?
    # Actually, 2.2M operations in Python is acceptable (<1s).
    # Let's try to speed it up by using `int.from_bytes` or similar? No, too complex.
    
    # We'll use the loop but ensure it's as tight as possible.
    for m in range(TOTAL_INPUTS):
        for k in range(N):
            if (m >> k) & 1:
                masks[k] |= (1 << m)
    return masks

def _simulate_tail(masks_before, tail_comparators):
    """
    Simulates the effect of 'tail_comparators' on 'masks_before'.
    Returns the final masks.
    """
    masks = masks_before[:]
    for i, j in tail_comparators:
        mi = masks[i]
        mj = masks[j]
        masks[i] = mi & mj
        masks[j] = mi | mj
    return masks

def _is_sorted(masks):
    """
    Checks if the output masks represent a sorted network.
    """
    for k in range(N - 1):
        if (masks[k] & ~masks[k + 1]) != 0:
            return False
    return True

def _build_states(network, initial_masks):
    """
    Builds the list of states (masks) after each comparator in the network.
    states[0] = initial_masks
    states[k] = masks after applying network[0:k]
    """
    states = [initial_masks]
    current_masks = initial_masks
    for comp in network:
        i, j = comp
        mi = current_masks[i]
        mj = current_masks[j]
        new_masks = current_masks[:]
        new_masks[i] = mi & mj
        new_masks[j] = mi | mj
        current_masks = new_masks
        states.append(current_masks)
    return states

def _odd_even_transposition():
    """
    Generates the odd-even transposition network.
    """
    network = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            network.append((i, i + 1))
    return network

def _greedy_delete(network, initial_masks):
    """
    Attempts to greedily delete comparators from the network.
    Returns the modified network and its states.
    """
    states = _build_states(network, initial_masks)
    
    # Verify initial validity
    if not _is_sorted(states[-1]):
        return network, states

    changed = True
    while changed:
        changed = False
        # Iterate backwards to handle index shifts naturally? 
        # Actually, removing an element shifts indices, so we break and restart 
        # or update states carefully.
        # Removing from the end is O(1) check if we have states.
        # Removing from the middle requires re-simulating the tail.
        
        # Strategy: Iterate from end to start.
        for k in range(len(network) - 1, -1, -1):
            # Check if network[k] can be removed
            # State before is states[k]
            # Tail is network[k+1:]
            
            if k == len(network) - 1:
                # Tail is empty
                if _is_sorted(states[k]):
                    network.pop(k)
                    states.pop()
                    changed = True
                    break
            else:
                # Simulate tail
                tail = network[k+1:]
                final_masks = _simulate_tail(states[k], tail)
                if _is_sorted(final_masks):
                    network.pop(k)
                    # Rebuild states is expensive. 
                    # But we break and restart the while loop, so we rebuild there?
                    # No, the while loop just continues. We should rebuild states 
                    # or update them.
                    # Rebuilding states is O(L * N * bit_ops). 
                    # For L~100, N=17, this is fast.
                    states = _build_states(network, initial_masks)
                    changed = True
                    break
                    
    return network, states

def construct():
    start_time = time.time()
    # Safety margin: 80s deadline to avoid 90s kill.
    deadline = start_time + 80.0
    
    # 1. Generate initial network
    network = _odd_even_transposition()
    
    # 2. Pre-compute initial masks
    initial_masks = _build_initial_masks()
    
    # 3. Initial Greedy Deletion
    network, _ = _greedy_delete(network, initial_masks)
    
    # 4. Local Search / Hill Climbing
    # Try to insert a comparator and then delete to see if we can improve.
    # Or try swapping comparators?
    # Insertion-Deletion is a good heuristic.
    
    best_network = list(network)
    best_len = len(network)
    
    # Pre-compute all possible pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Randomized search
    random.seed(42) # For reproducibility
    
    while time.time() < deadline - 2.0:
        # Pick a random starting point from the current best
        current_net = list(best_network)
        
        # Try a random insertion
        pos = random.randint(0, len(current_net))
        comp = random.choice(all_pairs)
        current_net.insert(pos, comp)
        
        # Check if this insertion keeps the network valid (it should, unless it breaks sorting? 
        # Actually, inserting a comparator into a valid sorting network keeps it valid?
        # No, it might not sort correctly if the comparator is in the wrong place?
        # Wait, if a network sorts all inputs, adding a comparator (i,j) means:
        # min(a_i, a_j) -> i, max(a_i, a_j) -> j.
        # If the previous output was sorted, is the new output sorted?
        # Not necessarily. The comparator might "unsort" it if i > j? No, i < j.
        # If the sequence was sorted, a_i <= a_j. So min=a_i, max=a_j. No change.
        # So inserting a comparator into a valid sorting network results in a valid sorting network.
        # So we don't need to check validity after insertion.
        
        # Now try to delete from this new network
        candidate_net, _ = _greedy_delete(current_net, initial_masks)
        
        if len(candidate_net) < best_len:
            best_network = candidate_net
            best_len = len(best_network)
            
    # Final verification
    states = _build_states(best_network, initial_masks)
    if _is_sorted(states[-1]):
        return best_network
    else:
        # Fallback
        return _odd_even_transposition()

if __name__ == "__main__":
    # Quick self-test
    net = construct()
    print(f"Found network with {len(net)} comparators")
