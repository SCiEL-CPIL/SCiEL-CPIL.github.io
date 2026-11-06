import time
import random
import math

N = 17
NUM_INPUTS = 1 << N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    # 1. Generate the seed network: Odd-Even Transposition
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best_network = seed_network[:]
    
    # 2. Verification Function using bit-vector simulation
    # Precompute the initial state masks for all 2^N inputs
    # state[j] is a bitmask of length 2^N, where bit k is 1 if input k has a 1 on wire j
    initial_state = [0] * N
    for j in range(N):
        mask = 0
        for k in range(NUM_INPUTS):
            if (k >> j) & 1:
                mask |= (1 << k)
        initial_state[j] = mask

    def is_valid(network):
        state = initial_state[:]
        for i, j in network:
            new_i = state[i] & state[j]
            new_j = state[i] | state[j]
            state[i] = new_i
            state[j] = new_j
        
        # Check if sorted: state[k] should be a subset of state[k+1]
        # i.e., state[k] & ~state[k+1] should be 0
        for k in range(N - 1):
            if (state[k] & ~state[k+1]) != 0:
                return False
        return True

    if not is_valid(best_network):
        return best_network

    random.seed(42)
    
    current_network = best_network[:]
    current_cost = len(current_network)
    best_cost = current_cost
    
    # Simulated Annealing parameters
    T = 5.0  # Higher initial temperature to encourage exploration
    T_min = 0.01
    cooling_rate = 0.995
    
    def try_remove_all(current_net):
        """Aggressively try to remove any redundant comparators."""
        removed = True
        while removed:
            if time.time() > deadline:
                break
            removed = False
            # Try removals in random order
            indices = list(range(len(current_net)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline:
                    break
                # Skip if we've already checked this index in this pass? 
                # No, we break out of the inner loop if we remove one, 
                # then restart the outer loop with a new shuffle.
                candidate = current_net[:idx] + current_net[idx+1:]
                if is_valid(candidate):
                    current_net = candidate
                    removed = True
                    break
        return current_net

    def grow_and_shrink(current_net, num_adds):
        """Add random comparators, then aggressively remove redundancies."""
        candidate = current_net[:]
        for _ in range(num_adds):
            new_i = random.randint(0, N - 1)
            new_j = random.randint(0, N - 1)
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            if new_i == new_j:
                continue
            pos = random.randint(0, len(candidate))
            candidate.insert(pos, (new_i, new_j))
        
        if not is_valid(candidate):
            return None
        
        optimized = try_remove_all(candidate)
        return optimized

    # Main Search Loop
    while time.time() < deadline:
        # 1. Aggressively reduce current network
        optimized = try_remove_all(current_network)
        if len(optimized) < current_cost:
            current_network = optimized
            current_cost = len(optimized)
            if current_cost < best_cost:
                best_network = current_network[:]
                best_cost = current_cost
        
        if time.time() > deadline:
            break

        # 2. Mutation Phase
        # Strategy: Grow and Shrink with varying growth sizes
        # Smaller growth (1) for fine-tuning, larger (2-4) for escaping minima
        r = random.random()
        if r < 0.5:
            num_adds = 1
        elif r < 0.8:
            num_adds = 2
        elif r < 0.95:
            num_adds = 3
        else:
            num_adds = 4
            
        new_net = grow_and_shrink(current_network, num_adds)
        
        if new_net is not None:
            new_cost = len(new_net)
            delta = new_cost - current_cost
            
            # Acceptance criterion
            if delta <= 0:
                accept = True
            else:
                # Accept worse solutions with probability exp(-delta/T)
                # If T is high, we accept more.
                accept_prob = math.exp(-delta / T)
                accept = random.random() < accept_prob
            
            if accept:
                current_network = new_net
                current_cost = new_cost
                if current_cost < best_cost:
                    best_network = current_network[:]
                    best_cost = current_cost
        
        # Cool down
        T = max(T * cooling_rate, T_min)

    return best_network
# EVOLVE-BLOCK-END
