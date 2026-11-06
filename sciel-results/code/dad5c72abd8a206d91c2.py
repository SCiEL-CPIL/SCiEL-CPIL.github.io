import time
import random
import math

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 55  # Safety margin

# Precompute the initial state for each wire as a Python integer bitmask.
# Bit k of wire i is 1 if input k has a 1 on wire i.
# Input k is an integer from 0 to 2^N - 1.
# For wire i, the bits set are those k where (k >> i) & 1 == 1.
def _get_initial_wire_masks():
    masks = []
    for i in range(N):
        mask = 0
        for k in range(1 << N):
            if (k >> i) & 1:
                mask |= (1 << k)
        masks.append(mask)
    return masks

INITIAL_MASKS = _get_initial_wire_masks()

def _odd_even_transposition(n):
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)]

def _simulate(net):
    """
    Simulate the network using Python integer masks.
    Returns the final state of the wires as a list of integers.
    """
    wires = INITIAL_MASKS[:]
    for i, j in net:
        # min goes to i, max goes to j
        # min(a, b) corresponds to a & b
        # max(a, b) corresponds to a | b
        wires[i], wires[j] = wires[i] & wires[j], wires[i] | wires[j]
    return wires

def _is_sorted(wires):
    """
    Check if the wires are sorted for all boolean inputs.
    Wire i must be a subset of Wire i+1 (i.e., wires[i] & ~wires[i+1] == 0)
    OR equivalently, (wires[i] & wires[i+1]) == wires[i]
    """
    for i in range(N - 1):
        if (wires[i] & wires[i+1]) != wires[i]:
            return False
    return True

def _verify_full(net):
    """
    Verify that the comparator network sorts all 2^N boolean inputs.
    """
    if len(net) > MAX_COMPARATORS:
        return False
    if len(net) == 0:
        return True
    wires = _simulate(net)
    return _is_sorted(wires)

def _prune(net):
    """
    Iteratively prune redundant comparators.
    A comparator (i, j) is redundant if wires[i] is already a subset of wires[j]
    before the comparator is applied.
    """
    changed = True
    while changed:
        changed = False
        # We need the state before each comparator.
        # Re-simulating from scratch for each check is O(L^2).
        for k in range(len(net) - 1, -1, -1):
            # Simulate up to k
            wires = INITIAL_MASKS[:]
            for idx in range(k):
                i, j = net[idx]
                wires[i], wires[j] = wires[i] & wires[j], wires[i] | wires[j]
            
            i, j = net[k]
            # Check if wires[i] is subset of wires[j]
            if (wires[i] & wires[j]) == wires[i]:
                net = net[:k] + net[k+1:]
                changed = True
                break
    return net

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with odd-even transposition network
    net = _odd_even_transposition(N)
    
    # Phase 1: Structural Pruning
    if time.time() < deadline - 10:
        net = _prune(net)
    
    best_net = net[:]
    best_count = len(net)
    
    # Phase 2: Simulated Annealing
    if time.time() < deadline - 5:
        T = 1.0
        min_T = 0.01
        cooling = 0.999
        current_count = len(net)
        
        while time.time() < deadline - 2:
            if T < min_T:
                # Restart from best
                net = best_net[:]
                current_count = best_count
                T = 1.0
            
            # Generate move
            r = random.random()
            if r < 0.4:
                # Remove
                if len(net) == 0:
                    break
                idx = random.randint(0, len(net) - 1)
                new_net = net[:idx] + net[idx+1:]
            elif r < 0.7:
                # Insert
                idx = random.randint(0, len(net))
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                new_net = net[:idx] + [(i, j)] + net[idx:]
            else:
                # Mutate
                if len(net) == 0:
                    continue
                idx = random.randint(0, len(net) - 1)
                i, j = net[idx]
                ni = random.randint(0, N - 2)
                nj = random.randint(ni + 1, N - 1)
                if (ni, nj) == (i, j):
                    continue
                new_net = net[:]
                new_net[idx] = (ni, nj)
            
            # Verify
            if not _verify_full(new_net):
                continue
                
            new_count = len(new_net)
            delta = new_count - current_count
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                net = new_net
                current_count = new_count
                if current_count < best_count:
                    best_net = net[:]
                    best_count = current_count
                    
            T *= cooling
    
    # Final validation
    if not _verify_full(best_net):
        best_net = _odd_even_transposition(N)
        
    return best_net
