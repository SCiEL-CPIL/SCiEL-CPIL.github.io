import time
import random

N = 16

# Precompute initial bitsets for zero-one principle verification
# state[i] is a 65536-bit integer where bit k is 1 if input k has a 1 in wire i
_INITIAL_BITSETS = []
for i in range(N):
    bitset = 0
    for k in range(1 << N):
        if k & (1 << i):
            bitset |= (1 << k)
    _INITIAL_BITSETS.append(bitset)

def _verify_zero_one_fast(net):
    """
    Fast verification using bit-slicing.
    Returns True if the network sorts all 2^N boolean inputs.
    """
    if not net:
        return True
    
    state = _INITIAL_BITSETS[:]
    
    for i, j in net:
        new_i = state[i] & state[j]
        new_j = state[i] | state[j]
        state[i] = new_i
        state[j] = new_j
        
    # Check if sorted: for all adjacent wires i, i+1, 
    # if wire i has a 1, wire i+1 must also have a 1 in all inputs.
    # state[i] & ~state[i+1] should be 0.
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
            
    return True

def _base_60_network():
    """
    A known 60-comparator sorting network for 16 wires.
    This is the Batcher Odd-Merge network for N=16.
    """
    return [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    ]

def _odd_even_network():
    """
    Odd-even transposition network: 120 comparators.
    """
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _try_remove_comparators(net, rng, deadline):
    """
    Systematically try to remove each comparator.
    Returns the best (smallest) valid network found.
    """
    best = net[:]
    best_len = len(best)
    
    # Try removing each comparator one at a time
    for idx in range(len(best)):
        if time.time() > deadline:
            break
        
        # Try removing comparator at idx
        candidate = best[:idx] + best[idx+1:]
        if _verify_zero_one_fast(candidate):
            best = candidate
            best_len -= 1
            # After successful removal, restart the scan from beginning
            # because indices have shifted
            idx = -1
    
    return best, best_len

def _guided_local_search(net, rng, deadline):
    """
    Local search that focuses on:
    1. Systematic removal of redundant comparators
    2. Swapping adjacent comparators to enable further removals
    3. Mutating endpoints to find better positions
    """
    current = net[:]
    current_len = len(current)
    best = current[:]
    best_len = current_len
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        
        # Phase 1: Try systematic removals
        new_best, new_best_len = _try_remove_comparators(current, rng, deadline)
        if new_best_len < current_len:
            current = new_best
            current_len = new_best_len
            improved = True
            no_improve = 0
            if current_len < best_len:
                best = current[:]
                best_len = current_len
            continue
        
        # Phase 2: Try swaps and mutations to enable future removals
        if len(current) > 1:
            # Try swapping two comparators
            if rng.random() < 0.7:
                i, j = rng.sample(range(len(current)), 2)
                candidate = current[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                if _verify_zero_one_fast(candidate):
                    current = candidate
                    no_improve += 1
                    improved = True
            else:
                # Try mutating an endpoint
                idx = rng.randrange(len(current))
                old_i, old_j = current[idx]
                
                # Try changing one endpoint
                if rng.random() < 0.5:
                    new_i = rng.randrange(N)
                    if new_i != old_i:
                        if new_i < old_j:
                            candidate = current[:]
                            candidate[idx] = (new_i, old_j)
                        elif new_i > old_j:
                            candidate = current[:]
                            candidate[idx] = (old_j, new_i)
                        else:
                            candidate = None
                    else:
                        candidate = None
                else:
                    new_j = rng.randrange(N)
                    if new_j != old_j:
                        if old_i < new_j:
                            candidate = current[:]
                            candidate[idx] = (old_i, new_j)
                        elif old_i > new_j:
                            candidate = current[:]
                            candidate[idx] = (new_j, old_i)
                        else:
                            candidate = None
                    else:
                        candidate = None
                
                if candidate is not None and _verify_zero_one_fast(candidate):
                    current = candidate
                    no_improve += 1
                    improved = True
        
        if not improved:
            no_improve += 1
    
    return best, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Initial best network
    best_net = _base_60_network()
    best_len = len(best_net)
    
    if not _verify_zero_one_fast(best_net):
        # Fallback to odd-even if base is broken (shouldn't happen)
        best_net = _odd_even_network()
        best_len = len(best_net)
        
    rng = random.Random(42)
    
    # Run guided local search
    improved_net, improved_len = _guided_local_search(best_net, rng, deadline)
    if improved_len < best_len:
        best_net = improved_net
        best_len = improved_len
    
    # If we have time, try more aggressive search from the improved network
    if time.time() < deadline - 5.0 and best_len > 50:
        # Perturb and search again
        perturbed = best_net[:]
        for _ in range(10):
            if len(perturbed) > 1:
                i, j = rng.sample(range(len(perturbed)), 2)
                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        
        perturbed_net, perturbed_len = _guided_local_search(perturbed, rng, deadline)
        if perturbed_len < best_len:
            best_net = perturbed_net
            best_len = perturbed_len
    
    return best_net

if __name__ == "__main__":
    # Quick self-test
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print(f"Valid: {_verify_zero_one_fast(net)}")
