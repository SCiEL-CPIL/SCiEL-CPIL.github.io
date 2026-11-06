import time
import random
import math

N = 17
NUM_INPUTS = 1 << N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s limit
    
    # 1. Generate a better seed network: Odd-Even Transposition with some optimization
    # Start with odd-even transposition
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best_network = seed_network[:]
    
    # 2. Verification Function using bit-vector simulation
    # Pre-compute initial state masks for all 17 wires
    initial_state = [0] * N
    for j in range(N):
        mask = 0
        for k in range(NUM_INPUTS):
            if (k >> j) & 1:
                mask |= (1 << k)
        initial_state[j] = mask
    
    def is_valid(network):
        state = [initial_state[k] for k in range(N)]
        for i, j in network:
            si = state[i]
            sj = state[j]
            new_i = si & sj
            new_j = si | sj
            state[i] = new_i
            state[j] = new_j
            # Early exit: if a wire becomes all-0s or all-1s, it's trivially sorted
            # with respect to any other wire in that direction
            # (all-0s is subset of everything; all-1s is superset of everything)
        
        # Check sorted: wire k <= wire k+1 means state[k] is subset of state[k+1]
        for k in range(N - 1):
            if (state[k] & ~state[k+1]) != 0:
                return False
        return True

    if not is_valid(best_network):
        return best_network
    
    random.seed(42)
    
    # 3. Simulated Annealing with aggressive optimization
    current_network = best_network[:]
    
    def try_remove_all(current_net, max_rounds=3):
        """Aggressively try to remove any redundant comparators."""
        removed = True
        rounds = 0
        while removed and rounds < max_rounds:
            if time.time() > deadline:
                break
            removed = False
            rounds += 1
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
    
    # Simulated annealing parameters - tuned for better exploration
    initial_temp = 6.0
    cooling_rate = 0.9985
    min_temp = 0.003
    
    temperature = initial_temp
    current_network = try_remove_all(current_network, max_rounds=8)
    
    while time.time() < deadline and temperature > min_temp:
        if time.time() > deadline:
            break
        
        # Generate a neighbor
        neighbor = current_network[:]
        strategy = random.random()
        n = len(neighbor)
        
        if strategy < 0.20:
            # Add 1-3 comparators at once for larger structural changes
            num_add = random.choice([1, 1, 2, 2, 3])
            pos = random.randint(0, len(neighbor))
            for _ in range(num_add):
                new_i = random.randint(0, N - 2)
                gap = random.choice([1, 1, 1, 2, 2, 3, 4, 5])
                new_j = min(N - 1, new_i + gap)
                if new_i < new_j:
                    neighbor.insert(pos, (new_i, new_j))
                    pos += 1
        
        elif strategy < 0.35:
            # Remove a random comparator
            if n > 1:
                idx = random.randint(0, n - 1)
                neighbor.pop(idx)
        
        elif strategy < 0.50:
            # Move a comparator to a different position
            if n > 2:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    comp = neighbor.pop(i)
                    if i < j:
                        j -= 1
                    neighbor.insert(j, comp)
        
        elif strategy < 0.65:
            # Swap two comparators
            if n > 1:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        elif strategy < 0.80:
            # Replace a comparator with a nearby wire pair
            if n > 0:
                i = random.randint(0, n - 1)
                old_i, old_j = neighbor[i]
                new_i = max(0, min(N - 2, old_i + random.randint(-2, 2)))
                new_j = max(new_i + 1, min(N - 1, old_j + random.randint(-2, 2)))
                if new_i < new_j:
                    neighbor[i] = (new_i, new_j)
        
        else:
            # Reorder a small segment of the network
            if n > 3:
                start = random.randint(0, n - 4)
                end = random.randint(start + 2, min(start + 5, n - 1))
                segment = neighbor[start:end+1]
                random.shuffle(segment)
                neighbor[start:end+1] = segment
        
        # Check validity
        if not is_valid(neighbor):
            temperature *= cooling_rate
            continue
        
        # Light cleanup: one round of removal to save time
        if time.time() < deadline:
            neighbor = try_remove_all(neighbor, max_rounds=1)
        
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
    
    # Final cleanup with more aggressive removal - multiple passes
    if time.time() < deadline:
        # Try multiple rounds of aggressive removal
        for _ in range(5):
            if time.time() > deadline:
                break
            final_network = try_remove_all(current_network, max_rounds=8)
            if len(final_network) < len(best_network):
                best_network = final_network[:]
                current_network = best_network[:]
    
    return best_network
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
