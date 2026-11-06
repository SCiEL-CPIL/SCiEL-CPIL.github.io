import random
import time
import math

N = 18
TOTAL_INPUTS = 1 << N

def _bitonic_sort_network(n):
    """
    Generate a bitonic sort network for n wires.
    For non-power-of-2, we split into powers of 2 and merge.
    Uses the standard bitonic merge approach.
    """
    if n <= 1:
        return []
    
    # Find the largest power of 2 <= n
    p = 1
    while p * 2 <= n:
        p *= 2
    
    if p == n:
        return _bitonic_power_of_2(n)
    
    # For non-power-of-2: split into p and n-p, sort both, then merge
    # This is a valid sorting network construction
    # Actually, let's just use a simpler approach: 
    # Generate bitonic for the next power of 2 and use only the needed wires
    # But that wastes comparators.
    
    # Better: use the fact that we can construct a sorting network for any n
    # by recursive splitting. Let's use a different approach:
    # Start with a known-good network and optimize.
    # For now, let's generate a reasonable network using a different method.
    
    # Use the "AKS-like" recursive approach or just a good heuristic
    # Let's generate using a simple recursive divide-and-conquer:
    return _recursive_sort_network(n)

def _bitonic_power_of_2(n):
    """Generate bitonic sort for n = power of 2."""
    network = []
    _bitonic_helper(n, 0, 1, network)
    return network

def _bitonic_helper(n, offset, direction, network):
    """Recursively generate bitonic sort comparators."""
    if n <= 1:
        return
    
    half = n // 2
    
    # Sort first half ascending, second half descending (or vice versa)
    _bitonic_helper(half, offset, 1, network)
    _bitonic_helper(half, offset + half, -1, network)
    
    # Merge the two halves
    _bitonic_merge(half, offset, direction, network)

def _bitonic_merge(half, offset, direction, network):
    """Merge two bitonic sequences of length 'half' starting at 'offset'."""
    for k in range(half, 0, -1):
        for i in range(0, k):
            a = offset + i
            b = offset + i + k
            if direction == 1:
                network.append((a, b))
            else:
                network.append((b, a))
                # Actually, compare-exchange is symmetric in effect for sorting
                # min goes to first wire, max to second
                # So we always do (min_wire, max_wire)
                # The direction affects which way the bitonic sequence goes
                # Let me reconsider...
                # For bitonic merge, we compare (i, i+k) for k from half down to 1
                # The direction determines the initial bitonic order
                # The comparators are always (smaller_index, larger_index)
                network[-1] = (a, b)  # Always (a, b) where a < b

def _recursive_sort_network(n):
    """
    Generate a sorting network using recursive divide and conquer.
    Split into two halves, sort each, then merge using a merge network.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    n1 = n // 2
    n2 = n - n1
    
    left = _recursive_sort_network(n1)
    right = _recursive_sort_network(n2)
    right = [(i + n1, j + n1) for i, j in right]
    
    # Merge: use a simple merge network
    # For merging two sorted lists of length n1 and n2
    # We can use the "odd-even merge" or a simple approach
    # Simple approach: compare all pairs that could be out of order
    # This is expensive but correct. Let's use a better merge.
    
    # Actually, let's use a different strategy:
    # Generate a network that is known to work, then optimize.
    # For now, let's use a simple but effective construction.
    
    # Use the "AKS" style: recursively sort, then do a merge phase
    # For simplicity, let's use a different seed: 
    # Start with a network that sorts by repeatedly comparing all pairs
    # in a specific order (like a bubble sort network), then optimize.
    
    # Let's just use a modified odd-even transposition as seed but with
    # a different structure that's easier to optimize.
    
    # Actually, let me implement a proper bitonic sort for any n by padding
    # to next power of 2, generating the network, then removing comparators
    # that only involve padded wires.
    
    p = 1
    while p < n:
        p *= 2
    
    # Generate bitonic sort for p wires
    full_network = _bitonic_power_of_2(p)
    
    # Keep only comparators where both wires are < n
    network = [(i, j) for i, j in full_network if i < n and j < n]
    
    return network

def _verify_network(network, inputs_mask):
    """
    Verify that the network sorts all 2^N Boolean inputs.
    inputs_mask: numpy array or list of length 2^N, each element is an N-bit mask.
    Returns True if all inputs are sorted.
    """
    # Use a faster verification: simulate on all inputs
    # For N=18, 2^18 = 262144 inputs
    # We'll simulate the network on all inputs simultaneously
    
    # Represent each input as an integer (bitmask)
    # After sorting, the bitmask should have all 1s before all 0s
    # i.e., the sorted bitmask should be of the form (1 << k) - 1 for some k
    
    states = list(range(TOTAL_INPUTS))
    
    for i, j in network:
        mask_i = 1 << i
        mask_j = 1 << j
        
        for idx in range(TOTAL_INPUTS):
            s = states[idx]
            bit_i = (s >> i) & 1
            bit_j = (s >> j) & 1
            if bit_i > bit_j:
                # Swap: set bit_i to 0, bit_j to 1
                s = s & ~mask_i
                s = s | mask_j
            states[idx] = s
    
    # Check if all states are sorted
    for s in states:
        # A sorted state has the form: some number of 1s followed by 0s
        # i.e., s == (1 << k) - 1 for some k in [0, N]
        if s != 0 and s != (1 << N) - 1:
            # Check if s is of the form (1 << k) - 1
            # (1 << k) - 1 has the property that s & (s + 1) == 0
            if s & (s + 1) != 0:
                return False
    return True

def _verify_network_fast(network):
    """
    Fast verification using the insight that we only need to check
    a subset of inputs, or use a smarter simulation.
    
    Actually, for Boolean inputs, the sorted output must be of the form
    11...100...0. So we can check: after processing, for each input,
    the output should have all 1s in the lower bits and 0s in the upper bits.
    
    Let's use a different approach: simulate the network symbolically
    or use the fact that we can check all 2^N inputs efficiently.
    """
    # Precompute the effect of the network on all inputs
    # Use array-based simulation for speed
    
    # states[i] = the bitmask of input i after processing
    states = list(range(TOTAL_INPUTS))
    
    for i, j in network:
        mask_i = 1 << i
        mask_j = 1 << j
        
        # For each state, if bit_i > bit_j, swap them
        for idx in range(TOTAL_INPUTS):
            s = states[idx]
            if ((s >> i) & 1) > ((s >> j) & 1):
                states[idx] = s ^ mask_i ^ mask_j
    
    # Check sortedness
    for s in states:
        if s & (s + 1) != 0:
            return False
    return True

def _verify_network_vectorized(network):
    """
    Even faster verification using bit manipulation on arrays.
    """
    # Represent all 2^N inputs as a list of integers
    # Simulate the network
    
    states = list(range(TOTAL_INPUTS))
    
    for i, j in network:
        mask_i = 1 << i
        mask_j = 1 << j
        
        for idx in range(TOTAL_INPUTS):
            s = states[idx]
            bit_i = (s >> i) & 1
            bit_j = (s >> j) & 1
            if bit_i != bit_j:
                # If bit_i=1, bit_j=0: swap -> bit_i=0, bit_j=1
                # If bit_i=0, bit_j=1: no swap needed (already correct)
                if bit_i == 1:
                    states[idx] = s ^ mask_i ^ mask_j
    
    # Check sortedness: s & (s+1) == 0
    for s in states:
        if s & (s + 1) != 0:
            return False
    return True

def _is_sorted_state(s):
    """Check if a bitmask represents a sorted state (1s then 0s)."""
    return (s & (s + 1)) == 0

def _verify_network_optimized(network):
    """
    Optimized verification.
    Key insight: we can early-exit if we find an unsorted state.
    """
    states = list(range(TOTAL_INPUTS))
    
    for i, j in network:
        mask_i = 1 << i
        mask_j = 1 << j
        
        for idx in range(TOTAL_INPUTS):
            s = states[idx]
            bit_i = (s >> i) & 1
            bit_j = (s >> j) & 1
            if bit_i == 1 and bit_j == 0:
                states[idx] = s ^ mask_i ^ mask_j
    
    for s in states:
        if s & (s + 1) != 0:
            return False
    return True

def _remove_comparator(network, idx):
    """Return a new network with the comparator at idx removed."""
    return network[:idx] + network[idx+1:]

def _swap_comparators(network, i, j):
    """Return a new network with comparators at positions i and j swapped."""
    new = list(network)
    new[i], new[j] = new[j], new[i]
    return new

def _move_comparator(network, src, dst):
    """Return a new network with comparator at src moved to dst."""
    new = list(network)
    comp = new.pop(src)
    new.insert(dst, comp)
    return new

def _insert_comparator(network, idx, comp):
    """Return a new network with comp inserted at position idx."""
    new = list(network)
    new.insert(idx, comp)
    return new

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    random.seed(42)
    
    # Generate initial network
    # Try bitonic sort first
    initial_network = _recursive_sort_network(N)
    
    # Verify it's correct
    if not _verify_network_optimized(initial_network):
        # Fallback to odd-even transposition
        initial_network = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                initial_network.append((i, i + 1))
    
    best_network = list(initial_network)
    best_size = len(best_network)
    
    # Aggressive optimization loop
    current = list(initial_network)
    
    # Phase 1: Greedy removal
    # Try removing comparators in multiple orders
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        # Shuffle the network
        shuffled = list(current)
        random.shuffle(shuffled)
        
        # Try removing from end to start
        improved = False
        for idx in range(len(shuffled) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            
            test = _remove_comparator(shuffled, idx)
            if _verify_network_optimized(test):
                shuffled = test
                improved = True
                if len(shuffled) < best_size:
                    best_size = len(shuffled)
                    best_network = list(shuffled)
        
        if improved:
            current = shuffled
        else:
            # Try different removal order
            shuffled2 = list(current)
            random.shuffle(shuffled2)
            improved2 = False
            for idx in range(len(shuffled2)):
                if time.time() > deadline - 10:
                    break
                test = _remove_comparator(shuffled2, idx)
                if _verify_network_optimized(test):
                    shuffled2 = test
                    improved2 = True
                    if len(shuffled2) < best_size:
                        best_size = len(shuffled2)
                        best_network = list(shuffled2)
            if improved2:
                current = shuffled2
    
    # Phase 2: Local search with swaps and moves
    for iteration in range(1000):
        if time.time() > deadline - 5:
            break
        
        # Randomly try a perturbation
        strategy = random.choice(['swap', 'move', 'insert_remove'])
        
        if strategy == 'swap' and len(current) >= 2:
            i = random.randint(0, len(current) - 2)
            test = _swap_comparators(current, i, i + 1)
            if _verify_network_optimized(test):
                current = test
                if len(current) < best_size:
                    best_size = len(current)
                    best_network = list(current)
        
        elif strategy == 'move' and len(current) >= 3:
            src = random.randint(0, len(current) - 1)
            dst = random.randint(0, len(current) - 1)
            if src != dst:
                test = _move_comparator(current, src, dst)
                if _verify_network_optimized(test):
                    current = test
                    if len(current) < best_size:
                        best_size = len(current)
                        best_network = list(current)
        
        elif strategy == 'insert_remove':
            # Insert a random comparator, then try to remove others
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            idx = random.randint(0, len(current))
            test = _insert_comparator(current, idx, (i, j))
            if _verify_network_optimized(test):
                # Try to remove a different comparator
                for r in range(len(test)):
                    if time.time() > deadline - 5:
                        break
                    test2 = _remove_comparator(test, r)
                    if _verify_network_optimized(test2):
                        current = test2
                        if len(current) < best_size:
                            best_size = len(current)
                            best_network = list(current)
                        break
                else:
                    current = test
    
    # Final: try more aggressive removal with multiple restarts
    for restart in range(10):
        if time.time() > deadline - 2:
            break
        
        current = list(best_network)
        random.shuffle(current)
        
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            test = _remove_comparator(current, idx)
            if _verify_network_optimized(test):
                current = test
                if len(current) < best_size:
                    best_size = len(current)
                    best_network = list(current)
    
    return best_network
