import time
import random
import math

N = 17
NUM_INPUTS = 1 << N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

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

    def is_valid(network):
        state = initial_state[:]
        for i, j in network:
            new_i = state[i] & state[j]
            new_j = state[i] | state[j]
            state[i] = new_i
            state[j] = new_j
        
        for k in range(N - 1):
            if (state[k] & ~state[k+1]) != 0:
                return False
        return True

    if not is_valid(best_network):
        return best_network

    random.seed(123)
    
    # 3. Simulated Annealing with aggressive optimization
    current_network = best_network[:]
    
    def try_remove_all(current_net):
        """Aggressively try to remove any redundant comparators."""
        removed = True
        while removed:
            if time.time() > deadline:
                break
            removed = False
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

    # Simulated annealing parameters
    initial_temp = 10.0
    cooling_rate = 0.995
    min_temp = 0.01
    
    temperature = initial_temp
    current_network = try_remove_all(current_network)
    
    while time.time() < deadline and temperature > min_temp:
        if time.time() > deadline:
            break
        
        # Generate a neighbor
        neighbor = current_network[:]
        strategy = random.random()
        
        if strategy < 0.3:
            # Add a random comparator
            new_i = random.randint(0, N - 1)
            new_j = random.randint(0, N - 1)
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            if new_i != new_j:
                pos = random.randint(0, len(neighbor))
                neighbor.insert(pos, (new_i, new_j))
        
        elif strategy < 0.5:
            # Remove a random comparator
            if len(neighbor) > 1:
                idx = random.randint(0, len(neighbor) - 1)
                neighbor.pop(idx)
        
        elif strategy < 0.7:
            # Move a comparator
            if len(neighbor) > 1:
                i = random.randint(0, len(neighbor) - 1)
                j = random.randint(0, len(neighbor) - 1)
                if i != j:
                    comp = neighbor.pop(i)
                    neighbor.insert(j, comp)
        
        elif strategy < 0.85:
            # Swap two comparators
            if len(neighbor) > 1:
                i = random.randint(0, len(neighbor) - 1)
                j = random.randint(0, len(neighbor) - 1)
                if i != j:
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        else:
            # Replace a comparator
            if len(neighbor) > 0:
                i = random.randint(0, len(neighbor) - 1)
                new_i = random.randint(0, N - 1)
                new_j = random.randint(0, N - 1)
                if new_i > new_j:
                    new_i, new_j = new_j, new_i
                if new_i != new_j:
                    neighbor[i] = (new_i, new_j)
        
        # Check validity and optimize
        if not is_valid(neighbor):
            # Invalid neighbor: only accept if we can fix it by removing
            # For now, reject invalid neighbors
            temperature *= cooling_rate
            continue
        
        neighbor = try_remove_all(neighbor)
        
        # Acceptance decision
        current_cost = len(current_network)
        neighbor_cost = len(neighbor)
        delta = neighbor_cost - current_cost
        
        if delta <= 0:
            # Accept improvement
            current_network = neighbor
            if len(current_network) < len(best_network):
                best_network = current_network[:]
        else:
            # Accept worse solution with probability exp(-delta/T)
            if temperature > 0:
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    current_network = neighbor
        
        temperature *= cooling_rate
    
    # Final cleanup
    if time.time() < deadline:
        final_network = try_remove_all(current_network)
        if len(final_network) < len(best_network):
            best_network = final_network[:]
    
    return best_network
# EVOLVE-BLOCK-END
