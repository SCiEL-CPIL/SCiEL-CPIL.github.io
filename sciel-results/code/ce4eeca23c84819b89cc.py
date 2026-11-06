import time
import random
import math

N = 16
ALL_ONES = (1 << (1 << N)) - 1  # 65536 ones

# Pre-compute initial states for speed
_initial_states = None

def _get_initial_states():
    global _initial_states
    if _initial_states is None:
        _initial_states = []
        for i in range(N):
            val = 0
            period = 1 << (i + 1)
            half = 1 << i
            period_val = ((1 << half) - 1) << half
            num_periods = (1 << N) // period
            shift = 0
            for _ in range(num_periods):
                val |= period_val << shift
                shift += period
            _initial_states.append(val)
    return _initial_states

def fast_verify(network):
    """Verify a sorting network using bit-parallel zero-one principle."""
    state = _get_initial_states()[:]
    
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def odd_even_network():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def batcher_network(n):
    """Generate Batcher's bitonic sorting network for n wires (n must be power of 2)."""
    if n == 2:
        return [(0, 1)]
    
    # Recursive construction
    # Stage 1: Sort first half and second half in opposite directions
    # Actually, standard recursive definition:
    # Batcher(n) = Batcher(n/2) [parallel] Batcher(n/2) [parallel] Merge(n/2, n/2)
    # But simpler to implement iteratively or recursively with directions.
    
    def batcher_recursive(k, direction=1):
        # k is log2(n)
        if k == 0:
            return []
        
        # Divide into two halves
        half = 1 << (k - 1)
        
        # Recursively sort the two halves
        # Left half: indices 0 to half-1
        # Right half: indices half to 2*half-1
        # We need to generate comparators for the left and right halves.
        # The direction of the second half is reversed for bitonic merge.
        
        # Get comparators for left half (indices 0..half-1)
        left_comps = batcher_recursive(k - 1, 1)
        # Map indices: 0..half-1 are already correct for left half
        
        # Get comparators for right half (indices half..2*half-1)
        # We need to generate a network for half wires, but with reversed direction?
        # Actually, Batcher's algorithm:
        # 1. Sort first half ascending.
        # 2. Sort second half descending.
        # 3. Merge the two bitonic sequences.
        
        # Let's implement a standard iterative bitonic sort for clarity and correctness.
        # It's easier to generate the network by simulating the stages.
        
        pass
    
    # Iterative Batcher's Bitonic Sort Network Generation
    # Stages: log2(n) stages.
    # At stage k (1 to log2(n)), we have blocks of size 2^k.
    # Inside each block, we compare elements i and i + 2^(k-1) for i in 0..2^(k-1)-1.
    # The direction of comparison depends on the bitonic sequence.
    
    comps = []
    k = 1
    while (1 << k) <= n:
        # For each block of size 2^k
        block_size = 1 << k
        half_block = 1 << (k - 1)
        
        # Determine the direction of the bitonic sequence for each block.
        # In a standard bitonic sort, the top-level is ascending.
        # Sub-blocks alternate.
        # A simpler way: Just generate all comparators for the "merge" steps.
        # Actually, the standard construction is:
        # For k = 1 to log2(n):
        #   For i = 0 to n-1 step 2^k:
        #     For j = 0 to 2^(k-1)-1:
        #       Compare (i + j, i + j + 2^(k-1))
        #       The direction is determined by whether the block is "up" or "down".
        #       For the outermost loop, it's "up".
        #       For the next level, it alternates.
        
        # Let's use a recursive helper that returns the list of comparators for a block of size 2^k.
        # This is safer.
        
        def gen_bitonic(k, start_idx, direction):
            # direction: 1 for ascending, -1 for descending
            if k == 0:
                return []
            
            half = 1 << (k - 1)
            comps_left = gen_bitonic(k - 1, start_idx, direction)
            comps_right = gen_bitonic(k - 1, start_idx + half, -direction)
            
            # Merge phase
            comps_merge = []
            # Compare i and i + half/2? No.
            # In the merge of two bitonic sequences of length half,
            # we compare (start_idx + i, start_idx + i + half) for i in 0..half-1?
            # No, the merge is done in log2(half) steps.
            # Actually, the recursive definition of Batcher is:
            # Batcher(2^k) = Batcher(2^(k-1)) || Batcher(2^(k-1)) [reversed] || Merge(2^(k-1), 2^(k-1))
            
            # Let's stick to the iterative stage-based generation which is easier to get right for network construction.
            # Reference: Batcher's Odd-Even-Merge Sort Network
            # For n=16:
            # Stage 1 (k=1): Compare (0,1), (2,3), (4,5)... (14,15)
            # Stage 2 (k=2): Compare (0,2), (1,3), (4,6), (5,7)...
            # Stage 3 (k=3): Compare (0,4), (1,5), (2,6), (3,7)...
            # Stage 4 (k=4): Compare (0,8), (1,9), (2,10), (3,11)...
            
            # Wait, the standard bitonic sort network has comparators in specific patterns.
            # Let's use a known correct generator.
            pass
        
        break # Fallback to a simpler valid network if this gets too complex
    
    # Fallback: Just return a large valid network (Odd-Even) if Batcher generation is complex.
    # But Batcher is 80 comparators, which is a good seed.
    # Let's hardcode the Batcher network for N=16 to be safe and fast.
    # Source: https://en.wikipedia.org/wiki/Bitonic_sorter
    # For 16 wires, the comparators are:
    # Stage 1: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15)
    # Stage 2: (0,2), (1,3), (4,6), (5,7), (8,10), (9,11), (12,14), (13,15)
    # Stage 3: (0,4), (1,5), (2,6), (3,7), (8,12), (9,13), (10,14), (11,15)
    # Stage 4: (0,8), (1,9), (2,10), (3,11), (4,12), (5,13), (6,14), (7,15)
    # Stage 5: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15)
    # Stage 6: (0,2), (1,3), (4,6), (5,7), (8,10), (9,11), (12,14), (13,15)
    # Stage 7: (0,4), (1,5), (2,6), (3,7), (8,12), (9,13), (10,14), (11,15)
    # Stage 8: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15)
    # Wait, that's 8 stages? No, log2(16)=4.
    # The number of comparators is 80.
    
    # Correct Batcher Network for N=16 (80 comparators):
    b = []
    # Stage 1: 8 comparators
    for i in range(0, 16, 2):
        b.append((i, i+1))
    # Stage 2: 8 comparators
    for i in range(0, 16, 4):
        b.append((i, i+2))
        b.append((i+1, i+3))
    # Stage 3: 8 comparators
    for i in range(0, 16, 8):
        b.append((i, i+4))
        b.append((i+1, i+5))
        b.append((i+2, i+6))
        b.append((i+3, i+7))
    # Stage 4: 8 comparators
    b.append((0, 8))
    b.append((1, 9))
    b.append((2, 10))
    b.append((3, 11))
    b.append((4, 12))
    b.append((5, 13))
    b.append((6, 14))
    b.append((7, 15))
    # Stage 5: 8 comparators (Repeat of Stage 1 pattern but on the merged halves? No, it's the merge step)
    # Actually, the bitonic sort network for 16 is composed of 4 stages of "bitonic merge".
    # The total count is 80.
    
    # Let's just use a known list of 80 comparators for Batcher.
    # If I can't guarantee the list is correct, I'll use Odd-Even (120) and a Random-Valid (150) as seeds.
    # Random-Valid: Generate 200 random comparators, verify, if invalid, add more?
    # Better: Use the "Knuth" network or just rely on Odd-Even and Batcher.
    # I will use Odd-Even and a "Shell" network.
    
    # Shell Network:
    # Gaps: 8, 4, 2, 1
    # For each gap g, compare (i, i+g) for all valid i.
    # This is a valid sorting network.
    shell = []
    for g in [8, 4, 2, 1]:
        for i in range(0, 16 - g):
            shell.append((i, i + g))
    
    # Check if shell is valid
    if fast_verify(shell):
        return shell
    else:
        return odd_even_network()

def greedy_remove(net, time_limit=None, order='reverse'):
    """Aggressively try to remove each comparator; return minimized net."""
    if order == 'reverse':
        indices = range(len(net) - 1, -1, -1)
    elif order == 'forward':
        indices = range(len(net))
    else: # random
        indices = list(range(len(net)))
        random.shuffle(indices)
        
    changed = True
    while changed:
        changed = False
        for idx in indices:
            if time_limit and time.time() > time_limit:
                return net
            if idx >= len(net):
                continue
            candidate = net[:idx] + net[idx+1:]
            if fast_verify(candidate):
                net = candidate
                changed = True
                # Restart the loop with the new length
                if order == 'reverse':
                    indices = range(len(net) - 1, -1, -1)
                elif order == 'forward':
                    indices = range(len(net))
                else:
                    indices = list(range(len(net)))
                    random.shuffle(indices)
                break
    return net

def simulated_annealing(net, time_limit, start_temp=1.0, end_temp=0.1, cooling=0.999):
    """Simulated annealing to minimize comparator count."""
    current_net = list(net)
    current_cost = len(current_net)
    best_net = list(current_net)
    best_cost = current_cost
    
    temp = start_temp
    while time.time() < time_limit and temp > end_temp:
        # Generate a neighbor
        op = random.random()
        if op < 0.5 and len(current_net) > 0:
            # Remove a random comparator
            idx = random.randint(0, len(current_net) - 1)
            neighbor = current_net[:idx] + current_net[idx+1:]
            neighbor_cost = len(neighbor)
        elif op < 0.8 and len(current_net) > 1:
            # Swap two comparators
            idx1 = random.randint(0, len(current_net) - 1)
            idx2 = random.randint(0, len(current_net) - 1)
            if idx1 != idx2:
                neighbor = list(current_net)
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                neighbor_cost = len(neighbor)
            else:
                continue
        else:
            # Insert a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(current_net))
            neighbor = current_net[:pos] + [(i, j)] + current_net[pos:]
            neighbor_cost = len(neighbor)
        
        # Verify neighbor
        if not fast_verify(neighbor):
            # Reject invalid
            temp *= cooling
            continue
            
        # Accept or reject
        delta = neighbor_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_net = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_net = list(current_net)
                best_cost = current_cost
                
        temp *= cooling
        
    return best_net, best_cost

def construct():
    deadline = time.time() + 85  # 85s to leave 5s margin
    start_time = time.time()
    
    seeds = []
    # Seed 1: Odd-Even
    seeds.append(odd_even_network())
    # Seed 2: Shell Sort Network
    shell_net = batcher_network(16) # This actually returns shell or odd-even
    if len(shell_net) < len(seeds[0]):
        seeds.append(shell_net)
    # Seed 3: Random Valid Network (large, then reduced)
    # Generate a network with 150 random comparators
    random_net = []
    for _ in range(150):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        random_net.append((i, j))
    # It's likely invalid. We can't use it as a seed if it's invalid.
    # So we skip this or try to make it valid by adding comparators?
    # Better: Just use the two known valid seeds.
    
    best_net = None
    best_count = float('inf')
    
    # Phase 1: Greedy Reduction on each seed
    for seed in seeds:
        if time.time() > deadline - 20:
            break
        reduced = greedy_remove(seed, deadline - 20, order='reverse')
        if len(reduced) < best_count:
            best_count = len(reduced)
            best_net = list(reduced)
            
        # Also try forward order
        reduced_fwd = greedy_remove(seed, deadline - 20, order='forward')
        if len(reduced_fwd) < best_count:
            best_count = len(reduced_fwd)
            best_net = list(reduced_fwd)
            
    if best_net is None:
        best_net = odd_even_network()
        best_count = len(best_net)
        
    # Phase 2: Simulated Annealing from the best greedy result
    if time.time() < deadline - 5:
        sa_result, sa_cost = simulated_annealing(best_net, deadline - 5)
        if sa_cost < best_count:
            best_count = sa_cost
            best_net = sa_result
            
    # Phase 3: If time remains, try SA from other seeds
    if time.time() < deadline - 5 and len(seeds) > 1:
        for seed in seeds[1:]:
            if time.time() > deadline - 2:
                break
            reduced = greedy_remove(seed, deadline - 5, order='random')
            if len(reduced) < best_count:
                best_count = len(reduced)
                best_net = list(reduced)
                
            sa_result, sa_cost = simulated_annealing(reduced, deadline - 2)
            if sa_cost < best_count:
                best_count = sa_cost
                best_net = sa_result
                
    return best_net
