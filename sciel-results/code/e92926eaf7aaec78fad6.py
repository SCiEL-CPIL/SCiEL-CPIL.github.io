import time
import random
import math
import numpy as np

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
    
    # 2. Verification Function using NumPy vectorization
    # State shape: (NUM_INPUTS, N)
    # state[input_idx, wire_idx] is 1 if input has 1 on wire
    
    # Generate initial state
    inputs = np.arange(NUM_INPUTS, dtype=np.uint32)
    initial_state = np.empty((NUM_INPUTS, N), dtype=np.uint8)
    for j in range(N):
        initial_state[:, j] = (inputs >> j) & 1

    def is_valid(network):
        # Copy state
        state = initial_state.copy()
        
        for i, j in network:
            # Vectorized compare-exchange
            # min goes to i, max goes to j
            # new_i = state[i] & state[j]
            # new_j = state[i] | state[j]
            new_i = state[:, i] & state[:, j]
            new_j = state[:, i] | state[:, j]
            state[:, i] = new_i
            state[:, j] = new_j
        
        # Check if sorted: state[:, k] <= state[:, k+1] for all k
        # Equivalent to no 1 in k and 0 in k+1
        # i.e., (state[:, k] & ~state[:, k+1]) == 0
        # Or simply: np.all(state[:, :-1] <= state[:, 1:])
        
        if np.any(state[:, :-1] > state[:, 1:]):
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
    
    # Simulated annealing parameters
    T = 2.0  # Initial temperature
    T_min = 0.005
    cooling_rate = 0.998

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
        dist = random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10])
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
    while time.time() < deadline:
        # Phase 1: Aggressively remove redundancies
        optimized = try_remove_all(current_network)
        if len(optimized) < current_cost:
            current_network = optimized
            current_cost = len(optimized)
            if current_cost < best_cost:
                best_network = current_network[:]
                best_cost = current_cost

        if time.time() > deadline:
            break
        
        # Phase 2: Mutate
        strategy = random.random()
        accepted = False
        
        if strategy < 0.35:
            # Add and optimize
            new_net = add_and_optimize(current_network)
            new_cost = len(new_net)
            
            if new_cost <= current_cost:
                current_network = new_net
                current_cost = new_cost
                accepted = True
                if current_cost < best_cost:
                    best_network = current_network[:]
                    best_cost = current_cost
            else:
                delta = new_cost - current_cost
                if random.random() < math.exp(-delta / T):
                    current_network = new_net
                    current_cost = new_cost
                    accepted = True
        
        elif strategy < 0.5:
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
                            if current_cost < best_cost:
                                best_network = current_network[:]
                                best_cost = current_cost
                        else:
                            delta = new_cost - current_cost
                            if random.random() < math.exp(-delta / T):
                                current_network = optimized
                                current_cost = new_cost
                                accepted = True
        
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
                            if current_cost < best_cost:
                                best_network = current_network[:]
                                best_cost = current_cost
                        else:
                            delta = new_cost - current_cost
                            if random.random() < math.exp(-delta / T):
                                current_network = optimized
                                current_cost = new_cost
                                accepted = True
        
        elif strategy < 0.8:
            # Replace one comparator
            new_net = replace_and_optimize(current_network)
            new_cost = len(new_net)
            
            if new_cost <= current_cost:
                current_network = new_net
                current_cost = new_cost
                accepted = True
                if current_cost < best_cost:
                    best_network = current_network[:]
                    best_cost = current_cost
            else:
                delta = new_cost - current_cost
                if random.random() < math.exp(-delta / T):
                    current_network = new_net
                    current_cost = new_cost
                    accepted = True
        
        elif strategy < 0.9:
            # Targeted mutation: try adding comparators between non-adjacent wires
            new_net = add_targeted(current_network)
            new_cost = len(new_net)
            
            if new_cost <= current_cost:
                current_network = new_net
                current_cost = new_cost
                accepted = True
                if current_cost < best_cost:
                    best_network = current_network[:]
                    best_cost = current_cost
            else:
                delta = new_cost - current_cost
                if random.random() < math.exp(-delta / T):
                    current_network = new_net
                    current_cost = new_cost
                    accepted = True
        
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
                    if current_cost < best_cost:
                        best_network = current_network[:]
                        best_cost = current_cost
                else:
                    delta = new_cost - current_cost
                    if random.random() < math.exp(-delta / T):
                        current_network = optimized
                        current_cost = new_cost
                        accepted = True

        # Cool down
        T = max(T * cooling_rate, T_min)

    return best_network
