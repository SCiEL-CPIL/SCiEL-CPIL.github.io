import time
import random

N = 17
NUM_INPUTS = 1 << N

def _generate_bitonic_16():
    """Generates a bitonic sorting network for 16 wires."""
    comparators = []
    
    def bitonic_sort(k, n, rev=False):
        if k <= 1:
            return
        bitonic_sort(k // 2, n, rev)
        bitonic_sort(k // 2, n + k // 2, not rev)
        bitonic_merge(k, n, rev)

    def bitonic_merge(k, n, rev=False):
        if k <= 1:
            return
        for i in range(n, n + k - 1):
            j = n + k - (i - n + 1)
            if i == j:
                continue
            if (i < j) != rev:
                comparators.append((i, j))
            else:
                comparators.append((j, i))
        bitonic_merge(k // 2, n, rev)
        bitonic_merge(k // 2, n + k // 2, rev)

    # Sort 16 wires (0 to 15)
    bitonic_sort(16, 0)
    return comparators

def _generate_merge_17():
    """Generates comparators to merge wire 16 into sorted wires 0-15."""
    comparators = []
    for i in range(15, -1, -1):
        comparators.append((i, 16))
    return comparators

def _verify_network_bitmask(net):
    """
    Verifies if the network sorts all 2^17 inputs using bitmask representation.
    Each input is a 17-bit integer. Wire k corresponds to bit k.
    A sorted state has all 0s in lower bits and 1s in higher bits.
    """
    # Start with all possible inputs
    inputs = list(range(NUM_INPUTS))
    
    for i, j in net:
        mask_i = 1 << i
        mask_j = 1 << j
        
        # For each input, if bit i is 1 and bit j is 0, swap them
        # This means: if input has (i=1, j=0), swap to (i=0, j=1)
        # New value = old_value XOR (mask_i | mask_j) when condition met
        for idx in range(len(inputs)):
            val = inputs[idx]
            if (val & mask_i) and not (val & mask_j):
                inputs[idx] = val ^ mask_i ^ mask_j
    
    # Check if all inputs are sorted
    # Sorted means: for each input, if we look at the bits from low to high,
    # we should see some 0s followed by some 1s
    for val in inputs:
        # Find the transition point: all bits below t should be 0, all above should be 1
        # Actually, sorted means non-decreasing, so bits should be 0...01...1
        # Check: once we see a 1, we shouldn't see a 0 again
        seen_one = False
        for bit in range(N):
            if val & (1 << bit):
                seen_one = True
            else:
                if seen_one:
                    return False  # Found a 0 after a 1, not sorted
    
    return True

def _verify_network(net):
    """Wrapper for backward compatibility."""
    return _verify_network_bitmask(net)

def construct():
    # Initial network
    net_16 = _generate_bitonic_16()
    merge_17 = _generate_merge_17()
    current_net = net_16 + merge_17
    
    # Verify initial (should be True)
    if not _verify_network(current_net):
        # Fallback to odd-even if something went wrong
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    best_net = current_net[:]
    best_count = len(best_net)
    
    # Local Search
    deadline = time.time() + 85.0
    random.seed(42)
    
    while time.time() < deadline:
        if not current_net:
            break
            
        # Try different mutation strategies
        strategy = random.random()
        
        if strategy < 0.5:
            # Try removing a random comparator
            idx = random.randint(0, len(current_net) - 1)
            new_net = current_net[:idx] + current_net[idx+1:]
        elif strategy < 0.8 and len(current_net) >= 2:
            # Try swapping two comparators
            idx1 = random.randint(0, len(current_net) - 1)
            idx2 = random.randint(0, len(current_net) - 1)
            if idx1 != idx2:
                new_net = current_net[:]
                new_net[idx1], new_net[idx2] = new_net[idx2], new_net[idx1]
            else:
                continue
        else:
            # Try replacing a comparator with a different wire pair
            idx = random.randint(0, len(current_net) - 1)
            i, j = current_net[idx]
            # Generate a new random comparator
            new_i = random.randint(0, N - 2)
            new_j = random.randint(new_i + 1, N - 1)
            new_net = current_net[:]
            new_net[idx] = (new_i, new_j)
        
        if _verify_network(new_net):
            current_net = new_net
            if len(current_net) < best_count:
                best_net = current_net[:]
                best_count = len(best_net)
        else:
            pass
            
    return best_net
