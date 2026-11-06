import time
import random
import math

N = 17
NUM_INPUTS = 1 << N

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # 1. Generate the seed network: Odd-Even Transposition
    # This uses N*(N+1)/2 comparators = 17*18/2 = 153
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best_network = seed_network[:]
    
    # 2. Verification Function using bit-vector simulation
    # State is a list of N integers, each representing a bitmask of NUM_INPUTS bits.
    # Bit k in state[j] is 1 if the k-th Boolean input has a 1 on wire j.
    initial_state = [0] * N
    for j in range(N):
        mask = 0
        for k in range(NUM_INPUTS):
            if (k >> j) & 1:
                mask |= (1 << k)
        initial_state[j] = mask

    def is_valid(network):
        # Copy state
        state = initial_state[:]
        # Track which wires are "settled" (all zeros or all ones)
        # A wire is settled if its mask is 0 or NUM_INPUTS-1
        all_ones = (1 << NUM_INPUTS) - 1
        
        for i, j in network:
            si = state[i]
            sj = state[j]
            # Early exit: if both wires are identical, no change needed
            if si == sj:
                continue
            new_i = si & sj
            new_j = si | sj
            state[i] = new_i
            state[j] = new_j
        
        # Check if sorted: state[k] should be a subset of state[k+1]
        # i.e., state[k] & ~state[k+1] == 0
        for k in range(N - 1):
            if (state[k] & ~state[k+1]) != 0:
                return False
        return True

    # Verify seed (should be valid)
    if not is_valid(best_network):
        return best_network
    
    random.seed(42)
    
    # 3. Local Search with Simulated Annealing
    current_network = best_network[:]
    current_cost = len(current_network)
    best_cost = current_cost
    
    # Simulated annealing parameters - tuned for better exploration
    T = 3.0  # Initial temperature - higher for better escape from local minima
    T_min = 0.003
    cooling_rate = 0.999  # Slower cooling for longer exploration

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

    def add_and_optimize(current_net):
        """Add a random comparator, then aggressively remove redundancies."""
        new_i = random.randint(0, N - 1)
        new_j = random.randint(0, N - 1)
        if new_i > new_j:
            new_i, new_j = new_j, new_i
        if new_i == new_j:
            return current_net
        
        # Prefer inserting at positions that may be more useful
        # Insert near the beginning, middle, or end with some bias
        r = random.random()
        if r < 0.3:
            pos = random.randint(0, max(1, len(current_net) // 3))
        elif r < 0.7:
            pos = random.randint(len(current_net) // 3, 2 * len(current_net) // 3)
        else:
            pos = random.randint(2 * len(current_net) // 3, len(current_net))
        
        candidate = current_net[:]
        candidate.insert(pos, (new_i, new_j))
        
        if not is_valid(candidate):
            return current_net
        
        # Aggressively remove redundancies from the enlarged network
        optimized = try_remove_all(candidate)
        return optimized

    def add_targeted(current_net):
        """Add a comparator between non-adjacent wires (longer distance), then optimize."""
        # Weight toward medium distances which are often more useful
        dist_weights = [1, 3, 4, 5, 4, 3, 2, 2, 1, 1, 1]  # distances 2-12
        dist = random.choices(range(2, min(N, 13)), weights=dist_weights[:min(N-1, 11)])[0]
        if dist >= N:
            dist = N - 1
        new_i = random.randint(0, N - 1 - dist)
        new_j = new_i + dist
        
        pos = random.randint(0, len(current_net))
        candidate = current_net[:]
        candidate.insert(pos, (new_i, new_j))
        
        if not is_valid(candidate):
            return current_net
        
        optimized = try_remove_all(candidate)
        return optimized

    def replace_and_optimize(current_net):
        """Replace a random comparator with a new one, then optimize."""
        if not current_net:
            return current_net
        idx = random.randint(0, len(current_net) - 1)
        new_i = random.randint(0, N - 1)
        new_j = random.randint(0, N - 1)
        if new_i > new_j:
            new_i, new_j = new_j, new_i
        if new_i == new_j:
            return current_net
            
        candidate = current_net[:]
        candidate[idx] = (new_i, new_j)
        
        if not is_valid(candidate):
            return current_net
        
        optimized = try_remove_all(candidate)
        return optimized

    # Simulated annealing loop
    # Track consecutive failures to adapt temperature
    consecutive_failures = 0
    max_consecutive_failures = 50
    
    while time.time() < deadline:
        # Phase 1: Aggressively remove redundancies
        optimized = try_remove_all(current_network)
        if len(optimized) < current_cost:
            current_network = optimized
            current_cost = len(optimized)
            consecutive_failures = 0
            if current_cost < best_cost:
                best_network = current_network[:]
                best_cost = current_cost

        if time.time() > deadline:
            break
        
        # Phase 2: Mutate
        strategy = random.random()
        accepted = False
        
        if strategy < 0.40:
            # Add and optimize - increased weight
            new_net = add_and_optimize(current_network)
            new_cost = len(new_net)
            
            if new_cost <= current_cost:
                current_network = new_net
                current_cost = new_cost
                accepted = True
                consecutive_failures = 0
                if current_cost < best_cost:
                    best_network = current_network[:]
                    best_cost = current_cost
            else:
                delta = new_cost - current_cost
                if random.random() < math.exp(-delta / T):
                    current_network = new_net
                    current_cost = new_cost
                    accepted = True
                    consecutive_failures = 0
        
        elif strategy < 0.55:
            # Move a comparator to a different position
            if len(current_network) > 1:
                i = random.randint(0, len(current_network) - 1)
                j = random.randint(0, len(current_network) - 1)
                if i != j:
                    candidate = current_network[:]
                    comp = candidate.pop(i)
                    candidate.insert(j, comp)
                    if is_valid(candidate):
                        optimized = try_remove_all(candidate)
                        new_cost = len(optimized)
                        
                        if new_cost <= current_cost:
                            current_network = optimized
                            current_cost = new_cost
                            accepted = True
                            consecutive_failures = 0
                            if current_cost < best_cost:
                                best_network = current_network[:]
                                best_cost = current_cost
                        else:
                            delta = new_cost - current_cost
                            if random.random() < math.exp(-delta / T):
                                current_network = optimized
                                current_cost = new_cost
                                accepted = True
                                consecutive_failures = 0
        
        elif strategy < 0.65:
            # Swap two comparators
            if len(current_network) > 1:
                i = random.randint(0, len(current_network) - 1)
                j = random.randint(0, len(current_network) - 1)
                if i != j:
                    candidate = current_network[:]
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    if is_valid(candidate):
                        optimized = try_remove_all(candidate)
                        new_cost = len(optimized)
                        
                        if new_cost <= current_cost:
                            current_network = optimized
                            current_cost = new_cost
                            accepted = True
                            consecutive_failures = 0
                            if current_cost < best_cost:
                                best_network = current_network[:]
                                best_cost = current_cost
                        else:
                            delta = new_cost - current_cost
                            if random.random() < math.exp(-delta / T):
                                current_network = optimized
                                current_cost = new_cost
                                accepted = True
                                consecutive_failures = 0
        
        elif strategy < 0.75:
            # Replace one comparator
            new_net = replace_and_optimize(current_network)
            new_cost = len(new_net)
            
            if new_cost <= current_cost:
                current_network = new_net
                current_cost = new_cost
                accepted = True
                consecutive_failures = 0
                if current_cost < best_cost:
                    best_network = current_network[:]
                    best_cost = current_cost
            else:
                delta = new_cost - current_cost
                if random.random() < math.exp(-delta / T):
                    current_network = new_net
                    current_cost = new_cost
                    accepted = True
                    consecutive_failures = 0
        
        elif strategy < 0.85:
            # Targeted mutation: try adding comparators between non-adjacent wires
            new_net = add_targeted(current_network)
            new_cost = len(new_net)
            
            if new_cost <= current_cost:
                current_network = new_net
                current_cost = new_cost
                accepted = True
                consecutive_failures = 0
                if current_cost < best_cost:
                    best_network = current_network[:]
                    best_cost = current_cost
            else:
                delta = new_cost - current_cost
                if random.random() < math.exp(-delta / T):
                    current_network = new_net
                    current_cost = new_cost
                    accepted = True
                    consecutive_failures = 0
        
        else:
            # Random restart-like mutation: add 2 random comparators
            additions = random.randint(1, 2)
            candidate = current_network[:]
            for _ in range(additions):
                new_i = random.randint(0, N - 1)
                new_j = random.randint(0, N - 1)
                if new_i > new_j:
                    new_i, new_j = new_j, new_i
                if new_i != new_j:
                    pos = random.randint(0, len(candidate))
                    candidate.insert(pos, (new_i, new_j))
            
            if is_valid(candidate):
                optimized = try_remove_all(candidate)
                new_cost = len(optimized)
                
                if new_cost <= current_cost:
                    current_network = optimized
                    current_cost = new_cost
                    accepted = True
                    consecutive_failures = 0
                    if current_cost < best_cost:
                        best_network = current_network[:]
                        best_cost = current_cost
                else:
                    delta = new_cost - current_cost
                    if random.random() < math.exp(-delta / T):
                        current_network = optimized
                        current_cost = new_cost
                        accepted = True
                        consecutive_failures = 0
        
        if not accepted:
            consecutive_failures += 1
        
        # Cool down
        T = max(T * cooling_rate, T_min)
        
        # If stuck for too long, do a partial restart
        if consecutive_failures >= max_consecutive_failures:
            consecutive_failures = 0
            # Keep the best network but add some random perturbations
            if time.time() > deadline:
                break
            candidate = best_network[:]
            num_adds = random.randint(2, 4)
            for _ in range(num_adds):
                new_i = random.randint(0, N - 1)
                new_j = random.randint(0, N - 1)
                if new_i > new_j:
                    new_i, new_j = new_j, new_i
                if new_i != new_j:
                    pos = random.randint(0, len(candidate))
                    candidate.insert(pos, (new_i, new_j))
            
            if is_valid(candidate):
                optimized = try_remove_all(candidate)
                current_network = optimized
                current_cost = len(optimized)
            else:
                current_network = best_network[:]
                current_cost = best_cost

    return best_network
