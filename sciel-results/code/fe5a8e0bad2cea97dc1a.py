import time
import random

N = 17
NUM_INPUTS = 1 << N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin

    # 1. Generate the seed network: Odd-Even Transposition
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best_network = seed_network[:]
    best_cost = len(best_network)
    
    # 2. Precompute initial state masks for verification
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

    random.seed(42)
    
    # 3. Genetic Algorithm with crossover
    MAX_COMPARATORS = 80  # Upper bound for network size
    
    def encode_network(network):
        """Encode network as fixed-length list, padding with (-1, -1)"""
        encoded = list(network)
        while len(encoded) < MAX_COMPARATORS:
            encoded.append((-1, -1))
        return encoded[:MAX_COMPARATORS]
    
    def decode_network(encoded):
        """Decode encoded network back to list of comparators"""
        return [(i, j) for i, j in encoded if i != -1]
    
    def mutate_encoded(encoded):
        """Apply random mutations to encoded network"""
        mutated = encoded[:]
        for _ in range(random.randint(1, 3)):
            pos = random.randint(0, len(mutated) - 1)
            if mutated[pos] == (-1, -1):
                # Replace dummy with random comparator
                new_i = random.randint(0, N - 1)
                new_j = random.randint(0, N - 1)
                if new_i > new_j:
                    new_i, new_j = new_j, new_i
                if new_i != new_j:
                    mutated[pos] = (new_i, new_j)
            else:
                # Replace existing comparator
                new_i = random.randint(0, N - 1)
                new_j = random.randint(0, N - 1)
                if new_i > new_j:
                    new_i, new_j = new_j, new_i
                if new_i != new_j:
                    mutated[pos] = (new_i, new_j)
                else:
                    mutated[pos] = (-1, -1)
        return mutated
    
    def crossover(parent1, parent2):
        """Uniform crossover between two encoded networks"""
        child = []
        for p1, p2 in zip(parent1, parent2):
            if random.random() < 0.5:
                child.append(p1)
            else:
                child.append(p2)
        return child
    
    def greedy_remove_redundant(network):
        """Aggressively remove redundant comparators"""
        removed = True
        while removed:
            if time.time() > deadline:
                break
            removed = False
            for idx in range(len(network)):
                if time.time() > deadline:
                    break
                candidate = network[:idx] + network[idx+1:]
                if is_valid(candidate):
                    network = candidate
                    removed = True
                    break
        return network
    
    # Initialize population
    POP_SIZE = 30
    population = [encode_network(best_network)]
    
    # Add diverse initial members
    for _ in range(POP_SIZE - 1):
        if random.random() < 0.3:
            # Mutate seed
            encoded = encode_network(best_network)
            encoded = mutate_encoded(encoded)
        else:
            # Random network
            encoded = [(-1, -1)] * MAX_COMPARATORS
            num_comparators = random.randint(60, 80)
            for _ in range(num_comparators):
                pos = random.randint(0, MAX_COMPARATORS - 1)
                if encoded[pos] == (-1, -1):
                    new_i = random.randint(0, N - 1)
                    new_j = random.randint(0, N - 1)
                    if new_i > new_j:
                        new_i, new_j = new_j, new_i
                    if new_i != new_j:
                        encoded[pos] = (new_i, new_j)
        population.append(encoded)
    
    # Evaluate initial population
    def evaluate_encoded(encoded):
        network = decode_network(encoded)
        if not network:
            return float('inf')
        if is_valid(network):
            return len(network)
        return float('inf')
    
    population_costs = [evaluate_encoded(ind) for ind in population]
    
    # Genetic algorithm loop
    GENERATION_SIZE = POP_SIZE
    while time.time() < deadline:
        # Selection: tournament
        def tournament_select():
            candidates = random.sample(range(POP_SIZE), 3)
            best_idx = min(candidates, key=lambda i: population_costs[i])
            return population[best_idx][:], population_costs[best_idx]
        
        # Create next generation
        new_population = []
        new_costs = []
        
        # Elitism: keep best individual
        best_idx = min(range(POP_SIZE), key=lambda i: population_costs[i])
        new_population.append(population[best_idx][:])
        new_costs.append(population_costs[best_idx])
        
        # Fill rest with offspring
        while len(new_population) < POP_SIZE and time.time() < deadline:
            parent1, _ = tournament_select()
            parent2, _ = tournament_select()
            
            child = crossover(parent1, parent2)
            child = mutate_encoded(child)
            
            # Try to improve by removing redundancies
            network = decode_network(child)
            if network and is_valid(network):
                optimized = greedy_remove_redundant(network)
                if len(optimized) < len(network):
                    network = optimized
                    child = encode_network(network)
            
            cost = evaluate_encoded(child)
            new_population.append(child)
            new_costs.append(cost)
        
        population = new_population
        population_costs = new_costs
        
        # Update best
        gen_best_idx = min(range(POP_SIZE), key=lambda i: population_costs[i])
        if population_costs[gen_best_idx] < best_cost:
            best_network = decode_network(population[gen_best_idx])
            best_cost = population_costs[gen_best_idx]
    
    return best_network
# EVOLVE-BLOCK-END
