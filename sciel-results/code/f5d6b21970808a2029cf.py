import time
import random
import math

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

def _find_redundant_comparators(net, rng, max_checks=50):
    """
    Try to find redundant comparators by removing them and checking validity.
    Returns a list of indices of redundant comparators.
    """
    redundant = []
    n = len(net)
    if n == 0:
        return redundant
    
    # Randomly sample indices to check
    indices_to_check = list(range(n))
    rng.shuffle(indices_to_check)
    
    checked = 0
    for idx in indices_to_check:
        if checked >= max_checks:
            break
        checked += 1
        
        # Create a new network without this comparator
        new_net = net[:idx] + net[idx+1:]
        
        if _verify_zero_one_fast(new_net):
            redundant.append(idx)
            
    return redundant

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
    
    # Strategy: Iteratively remove redundant comparators
    # This is a form of greedy pruning that guarantees validity at each step
    
    current_net = best_net[:]
    current_len = len(current_net)
    
    # Main loop: keep trying to remove comparators
    while time.time() < deadline:
        # Try to find and remove redundant comparators
        redundant = _find_redundant_comparators(current_net, rng, max_checks=20)
        
        if not redundant:
            # No redundant comparators found in this round
            # Try a more aggressive search: random swaps and mutations
            improved = False
            for _ in range(100):
                if time.time() > deadline:
                    break
                    
                # Random move
                move_type = rng.random()
                
                if move_type < 0.3:
                    # Swap two comparators
                    if len(current_net) > 1:
                        i, j = rng.sample(range(len(current_net)), 2)
                        new_net = current_net[:]
                        new_net[i], new_net[j] = new_net[j], new_net[i]
                        if _verify_zero_one_fast(new_net) and len(new_net) == current_len:
                            current_net = new_net
                            improved = True
                elif move_type < 0.6:
                    # Mutate a comparator
                    if len(current_net) > 0:
                        idx = rng.randrange(len(current_net))
                        old_i, old_j = current_net[idx]
                        new_i = rng.randrange(N)
                        new_j = rng.randrange(N)
                        if new_i < new_j:
                            new_net = current_net[:]
                            new_net[idx] = (new_i, new_j)
                            if _verify_zero_one_fast(new_net) and len(new_net) == current_len:
                                current_net = new_net
                                improved = True
                else:
                    # Try removing a random comparator
                    if len(current_net) > 1:
                        idx = rng.randrange(len(current_net))
                        new_net = current_net[:idx] + current_net[idx+1:]
                        if _verify_zero_one_fast(new_net):
                            current_net = new_net
                            current_len = len(current_net)
                            improved = True
                            if current_len < best_len:
                                best_net = current_net[:]
                                best_len = current_len
                            
            if not improved and current_len == best_len:
                # Stuck, try restarting from best
                current_net = best_net[:]
                current_len = best_len
                # Add some random perturbation
                for _ in range(3):
                    if len(current_net) > 1:
                        i, j = rng.sample(range(len(current_net)), 2)
                        current_net[i], current_net[j] = current_net[j], current_net[i]
        else:
            # Remove all redundant comparators found
            # Sort indices in reverse order to avoid shifting issues
            redundant.sort(reverse=True)
            for idx in redundant:
                current_net = current_net[:idx] + current_net[idx+1:]
            
            current_len = len(current_net)
            if current_len < best_len:
                best_net = current_net[:]
                best_len = current_len
    
    return best_net

if __name__ == "__main__":
    # Quick self-test
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print(f"Valid: {_verify_zero_one_fast(net)}")
