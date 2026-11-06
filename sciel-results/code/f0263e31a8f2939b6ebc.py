import time
import random
import math

N = 17
NUM_INPUTS = 1 << N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # 1. Generate the seed network: Odd-Even Transposition
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best_network = seed_network[:]
    
    # 2. Verification Function using bit-vector simulation
    initial_state = [0] * N
    for j in range(N):
        mask = 0
        for k in range(NUM_INPUTS):
            if (k >> j) & 1:
                mask |= (1 << k)
        initial_state[j] = mask

    # Pre-compute the sorted state for quick comparison
    # After sorting, wire k should have exactly the inputs where the top k bits are all 1
    # i.e., state[k] should equal the mask of inputs where bits k..N-1 are all 1
    sorted_state = [0] * N
    for k in range(N):
        mask = 0
        for inp in range(NUM_INPUTS):
            # Check if bits k through N-1 are all 1
            all_ones = True
            for bit in range(k, N):
                if (inp >> bit) & 1 == 0:
                    all_ones = False
                    break
            if all_ones:
                mask |= (1 << inp)
        sorted_state[k] = mask

    def is_valid(network):
        state = initial_state[:]
        for i, j in network:
            new_i = state[i] & state[j]
            new_j = state[i] | state[j]
            state[i] = new_i
            state[j] = new_j
        
        for k in range(N):
            if state[k] != sorted_state[k]:
                return False
        return True

    if not is_valid(best_network):
        return best_network
    
    random.seed(42)
    
    # 3. Local Search with Simulated Annealing
    current_network = best_network[:]
    current_cost = len(current_network)
    best_cost = current_cost
    
    # Simulated annealing parameters
    T = 3.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.997
    
    def try_remove_all(current_net):
        """Aggressively try to remove any redundant comparators."""
        removed = True
        while removed:
            if time.time() > deadline:
                break
            removed = False
            # Try removals in random order to escape local minima
            indices = list(range(len(current_net)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline:
                    break
                candidate = current_net[:idx] + current_net[idx+1:]
                if is_valid(candidate):
                    current_net = candidate
                    removed = True
                    break
        return current_net

    def is_valid_fast(network, base_state=None):
        """Fast verification that can reuse state from a previous valid network."""
        if base_state is None:
            state = initial_state[:]
        else:
            state = base_state[:]
        
        for i, j in network:
            new_i = state[i] & state[j]
            new_j = state[i] | state[j]
            state[i] = new_i
            state[j] = new_j
        
        for k in range(N):
            if state[k] != sorted_state[k]:
                return False, None
        return True, state

    def add_and_optimize(current_net):
        """Add a random comparator, then aggressively remove redundancies."""
        new_i = random.randint(0, N - 1)
        new_j = random.randint(0, N - 1)
        if new_i > new_j:
            new_i, new_j = new_j, new_i
        if new_i == new_j:
            return current_net
        
        pos = random.randint(0, len(current_net))
        candidate = current_net[:]
        candidate.insert(pos, (new_i, new_j))
        
        if not is_valid(candidate):
            return current_net
        
        # Aggressively remove redundancies from the enlarged network
        optimized = try_remove_all(candidate)
        return optimized

    def add_targeted(current_net):
        """Add a comparator between non-adjacent wires (longer distance), then optimize."""
        dist = random.choice([2, 3, 4, 5, 6, 7, 8])
        new_i = random.randint(0, N - 1 - dist)
        new_j = new_i + dist
        
        pos = random.randint(0, len(current_net))
        candidate = current_net[:]
        candidate.insert(pos, (new_i, new_j))
        
        if not is_valid(candidate):
            return current_net
        
        optimized = try_remove_all(candidate)
        return optimized

    # Simulated annealing loop
    while time.time() < deadline:
        # Phase 1: Aggressively remove
        optimized = try_remove_all(current_network)
        if len(optimized) < current_cost:
            current_network = optimized
            current_cost = len(optimized)
            if current_cost < best_cost:
                best_network = current_network[:]
                best_cost = current_cost
        
        if time.time() > deadline:
            break
        
        # Phase 2: Mutate - try different strategies
        strategy = random.random()
        
        if strategy < 0.3:
            # Add and optimize (most likely to find improvements)
            new_net = add_and_optimize(current_network)
            new_cost = len(new_net)
            
            # Simulated annealing acceptance criterion
            if new_cost <= current_cost:
                current_network = new_net
                current_cost = new_cost
                if current_cost < best_cost:
                    best_network = current_network[:]
                    best_cost = current_cost
            else:
                # Accept worse solution with probability exp(-delta/T)
                delta = new_cost - current_cost
                if random.random() < math.exp(-delta / T):
                    current_network = new_net
                    current_cost = new_cost
        
        elif strategy < 0.45:
            # Move a comparator to a different position
            if len(current_network) > 1:
                i = random.randint(0, len(current_network) - 1)
                j = random.randint(0, len(current_network) - 1)
                if i != j:
                    candidate = current_network[:]
                    comp = candidate.pop(i)
                    candidate.insert(j, comp)
                    valid, _ = is_valid_fast(candidate)
                    if valid:
                        optimized = try_remove_all(candidate)
                        new_cost = len(optimized)
                        
                        if new_cost <= current_cost:
                            current_network = optimized
                            current_cost = new_cost
                            if current_cost < best_cost:
                                best_network = current_network[:]
                                best_cost = current_cost
                        else:
                            delta = new_cost - current_cost
                            if random.random() < math.exp(-delta / T):
                                current_network = optimized
                                current_cost = new_cost
        
        elif strategy < 0.6:
            # Swap two comparators
            if len(current_network) > 1:
                i = random.randint(0, len(current_network) - 1)
                j = random.randint(0, len(current_network) - 1)
                if i != j:
                    candidate = current_network[:]
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    valid, _ = is_valid_fast(candidate)
                    if valid:
                        # Try to remove redundancies after swap
                        optimized = try_remove_all(candidate)
                        new_cost = len(optimized)
                        
                        if new_cost <= current_cost:
                            current_network = optimized
                            current_cost = new_cost
                            if current_cost < best_cost:
                                best_network = current_network[:]
                                best_cost = current_cost
                        else:
                            delta = new_cost - current_cost
                            if random.random() < math.exp(-delta / T):
                                current_network = optimized
                                current_cost = new_cost
        
        elif strategy < 0.75:
            # Replace one comparator
            if len(current_network) > 0:
                i = random.randint(0, len(current_network) - 1)
                new_i = random.randint(0, N - 1)
                new_j = random.randint(0, N - 1)
                if new_i > new_j:
                    new_i, new_j = new_j, new_i
                if new_i != new_j:
                    candidate = current_network[:]
                    candidate[i] = (new_i, new_j)
                    valid, _ = is_valid_fast(candidate)
                    if valid:
                        optimized = try_remove_all(candidate)
                        new_cost = len(optimized)
                        
                        if new_cost <= current_cost:
                            current_network = optimized
                            current_cost = new_cost
                            if current_cost < best_cost:
                                best_network = current_network[:]
                                best_cost = current_cost
                        else:
                            delta = new_cost - current_cost
                            if random.random() < math.exp(-delta / T):
                                current_network = optimized
                                current_cost = new_cost
        
        elif strategy < 0.9:
            # Targeted mutation: try adding comparators between non-adjacent wires
            new_net = add_targeted(current_network)
            new_cost = len(new_net)
            
            if new_cost <= current_cost:
                current_network = new_net
                current_cost = new_cost
                if current_cost < best_cost:
                    best_network = current_network[:]
                    best_cost = current_cost
            else:
                delta = new_cost - current_cost
                if random.random() < math.exp(-delta / T):
                    current_network = new_net
                    current_cost = new_cost
        
        else:
            # Random restart: try a completely different mutation
            # Add 2-3 comparators at random positions, then optimize
            additions = random.randint(2, 3)
            candidate = current_network[:]
            for _ in range(additions):
                new_i = random.randint(0, N - 1)
                new_j = random.randint(0, N - 1)
                if new_i > new_j:
                    new_i, new_j = new_j, new_i
                if new_i != new_j:
                    pos = random.randint(0, len(candidate))
                    candidate.insert(pos, (new_i, new_j))
            
            valid, _ = is_valid_fast(candidate)
            if valid:
                optimized = try_remove_all(candidate)
                new_cost = len(optimized)
                
                if new_cost <= current_cost:
                    current_network = optimized
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_network = current_network[:]
                        best_cost = current_cost
                else:
                    delta = new_cost - current_cost
                    if random.random() < math.exp(-delta / T):
                        current_network = optimized
                        current_cost = new_cost
        
        # Cool down
        T = max(T * cooling_rate, T_min)

    # Final aggressive removal pass
    if time.time() < deadline - 2:
        final_optimized = try_remove_all(best_network)
        if len(final_optimized) < best_cost:
            best_network = final_optimized
            best_cost = len(final_optimized)

    return best_network
# EVOLVE-BLOCK-END
