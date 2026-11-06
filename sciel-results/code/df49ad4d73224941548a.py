import time
import random
import math

N = 16
NUM_INPUTS = 1 << N  # 65536
ALL_ONES = (1 << NUM_INPUTS) - 1

# Precompute initial bitsets for zero-one principle verification
# state[i] is a 65536-bit integer where bit k is 1 if input k has a 1 in wire i
_INITIAL_BITSETS = []
for i in range(N):
    bitset = 0
    for k in range(NUM_INPUTS):
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
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
            
    return True

def _compute_states(net):
    """
    Computes the state of the wires before each comparator.
    Returns a list 'states' where states[k] is the list of N wire states
    before comparator k is applied. states[0] is the initial state.
    Also returns the final state.
    """
    if not net:
        return [_INITIAL_BITSETS[:]], _INITIAL_BITSETS[:]
    
    states = [_INITIAL_BITSETS[:]]
    current_state = _INITIAL_BITSETS[:]
    
    for i, j in net:
        next_state = current_state[:]
        next_state[i] = current_state[i] & current_state[j]
        next_state[j] = current_state[i] | current_state[j]
        states.append(next_state)
        current_state = next_state
        
    return states, current_state

def _remove_redundant(net):
    """
    Removes all comparators that are no-ops in the current network.
    Returns the new network.
    """
    if not net:
        return []
        
    states, _ = _compute_states(net)
    to_remove = set()
    
    for k, (i, j) in enumerate(net):
        # Check if wire i <= wire j for all inputs before this comparator
        if states[k][i] & ~states[k][j] == 0:
            to_remove.add(k)
            
    if not to_remove:
        return net
        
    new_net = [net[k] for k in range(len(net)) if k not in to_remove]
    return new_net

def _prune_network(net, deadline):
    """
    Repeatedly remove redundant comparators until none are left.
    """
    current_net = net[:]
    changed = True
    while changed and time.time() < deadline:
        new_net = _remove_redundant(current_net)
        if len(new_net) < len(current_net):
            current_net = new_net
            changed = True
        else:
            changed = False
            
    return current_net

def _base_60_network():
    """
    A known 60-comparator sorting network for 16 wires.
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

def _bitonic_sort_network(n):
    """
    Bitonic sort network for n wires (n must be power of 2).
    """
    if n <= 1:
        return []
    
    comparators = []
    
    def bitonic_merge(size, start, direction):
        if size <= 1:
            return
        half = size // 2
        for i in range(half):
            a = start + i
            b = start + i + half
            if direction:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
        bitonic_merge(half, start, direction)
        bitonic_merge(half, start + half, direction)
        
    def bitonic_sort(size, start, direction):
        if size <= 1:
            return
        half = size // 2
        bitonic_sort(half, start, True)
        bitonic_sort(half, start + half, False)
        bitonic_merge(size, start, direction)
        
    bitonic_sort(n, 0, True)
    return comparators

def _local_search_prune(net, deadline, rng):
    """
    Local search that tries to swap/move comparators to enable further pruning.
    Uses simulated annealing-like acceptance.
    """
    current_net = net[:]
    best_net = net[:]
    best_len = len(best_net)
    
    # First, prune the initial net
    current_net = _prune_network(current_net, deadline)
    if len(current_net) < best_len:
        best_net = current_net[:]
        best_len = len(current_net)
        
    no_improve = 0
    max_no_improve = 500
    temperature = 1.0
    cooling = 0.999
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Generate a random perturbation
        move_type = rng.choice(['swap', 'move', 'insert', 'remove'])
        
        new_net = current_net[:]
        
        if move_type == 'swap' and len(new_net) > 1:
            i, j = rng.sample(range(len(new_net)), 2)
            new_net[i], new_net[j] = new_net[j], new_net[i]
        elif move_type == 'move' and len(new_net) > 1:
            i = rng.randrange(len(new_net))
            j = rng.randrange(len(new_net))
            item = new_net.pop(i)
            new_net.insert(j, item)
        elif move_type == 'insert':
            # Insert a random comparator
            i, j = rng.sample(range(N), 2)
            if i > j: i, j = j, i
            pos = rng.randrange(len(new_net) + 1)
            new_net.insert(pos, (i, j))
        elif move_type == 'remove' and len(new_net) > 1:
            idx = rng.randrange(len(new_net))
            new_net.pop(idx)
        else:
            no_improve += 1
            continue
            
        # Verify and prune
        if _verify_zero_one_fast(new_net):
            pruned_net = _prune_network(new_net, deadline)
            new_len = len(pruned_net)
            
            if new_len < best_len:
                best_net = pruned_net[:]
                best_len = new_len
                current_net = pruned_net[:]
                no_improve = 0
                temperature = 1.0  # Reset temperature on improvement
            elif new_len == best_len:
                # Accept equal length to move around plateau
                if rng.random() < 0.3:
                    current_net = pruned_net[:]
                no_improve += 1
            else:
                # Accept worse solution with probability based on temperature
                delta = new_len - best_len
                prob = math.exp(-delta / max(temperature, 0.01))
                if rng.random() < prob:
                    current_net = pruned_net[:]
                no_improve += 1
        else:
            no_improve += 1
            
        temperature *= cooling
        
    return best_net

def _multi_seed_search(deadline, rng):
    """
    Try multiple seed networks and local search on each.
    """
    seeds = []
    
    # 1. Base 60 network
    base_net = _base_60_network()
    if _verify_zero_one_fast(base_net):
        seeds.append(('base60', base_net))
        
    # 2. Odd-even transposition
    oe_net = _odd_even_network()
    if _verify_zero_one_fast(oe_net):
        seeds.append(('odd_even', oe_net))
        
    # 3. Bitonic sort
    bitonic_net = _bitonic_sort_network(N)
    if _verify_zero_one_fast(bitonic_net):
        seeds.append(('bitonic', bitonic_net))
        
    # 4. Shell sort network
    shell_net = []
    gaps = [8, 4, 2, 1]
    for gap in gaps:
        for i in range(gap, N):
            shell_net.append((i - gap, i))
    if _verify_zero_one_fast(shell_net):
        seeds.append(('shell', shell_net))
        
    if not seeds:
        return _odd_even_network()
        
    best_net = None
    best_len = float('inf')
    
    # Phase 1: Prune each seed and keep the best
    for name, seed in seeds:
        if time.time() > deadline - 5:
            break
        pruned = _prune_network(seed, deadline)
        if len(pruned) < best_len:
            best_net = pruned
            best_len = len(pruned)
            
    if best_net is None:
        best_net = seeds[0][1]
        best_len = len(best_net)
        
    # Phase 2: Local search on the best pruned network
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        best_net = _local_search_prune(best_net, deadline, rng)
        
    # Final verification
    if not _verify_zero_one_fast(best_net):
        # Fallback to a known valid one
        for name, seed in seeds:
            if _verify_zero_one_fast(seed):
                return seed
        return _odd_even_network()
        
    return best_net

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    rng = random.Random(42)
    
    net = _multi_seed_search(deadline, rng)
    
    # Final verification
    if not _verify_zero_one_fast(net):
        return _odd_even_network()
        
    return net

if __name__ == "__main__":
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print(f"Valid: {_verify_zero_one_fast(net)}")
