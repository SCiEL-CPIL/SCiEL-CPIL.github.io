"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136

# Precompute all Boolean inputs
def _make_all_inputs():
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    inputs = ((indices >> bit_positions) & 1).astype(np.uint8)
    return inputs

_ALL_INPUTS = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = _make_all_inputs()
    return _ALL_INPUTS

def simulate_network(comparators, inputs=None):
    if inputs is None:
        inputs = _get_all_inputs()
    state = inputs.copy()
    for (i, j) in comparators:
        mi = state[:, i] & state[:, j]
        mj = state[:, i] | state[:, j]
        state[:, i] = mi
        state[:, j] = mj
    return state

def is_sorted_network(comparators):
    inputs = _get_all_inputs()
    output = simulate_network(comparators, inputs)
    for k in range(N - 1):
        if np.any(output[:, k] > output[:, k + 1]):
            return False
    return True

def random_comparator():
    i = random.randint(0, N - 2)
    j = random.randint(i + 1, N - 1)
    return (i, j)

def mutate_network(net, mutation_rate=0.1):
    """
    Mutate a network by performing random operations.
    """
    new_net = list(net)
    num_ops = max(1, int(len(new_net) * mutation_rate))
    
    for _ in range(num_ops):
        if not new_net:
            break
            
        op = random.choice(['remove', 'swap', 'replace', 'insert'])
        
        if op == 'remove':
            if len(new_net) > 1:
                idx = random.randint(0, len(new_net) - 1)
                new_net.pop(idx)
                
        elif op == 'swap':
            if len(new_net) > 1:
                i, j = random.sample(range(len(new_net)), 2)
                new_net[i], new_net[j] = new_net[j], new_net[i]
                
        elif op == 'replace':
            idx = random.randint(0, len(new_net) - 1)
            new_net[idx] = random_comparator()
            
        elif op == 'insert':
            pos = random.randint(0, len(new_net))
            new_net.insert(pos, random_comparator())
            
    return new_net

def crossover(parent1, parent2):
    """
    Uniform crossover: for each position, pick from parent1 or parent2.
    We need to handle different lengths.
    """
    len1 = len(parent1)
    len2 = len(parent2)
    max_len = max(len1, len2)
    
    child = []
    for i in range(max_len):
        if i < len1 and i < len2:
            # Pick one
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        elif i < len1:
            # Only in parent1
            if random.random() < 0.7: # Slightly bias towards keeping
                child.append(parent1[i])
        elif i < len2:
            # Only in parent2
            if random.random() < 0.7:
                child.append(parent2[i])
                
    return child

def genetic_algorithm(seed_net, deadline):
    pop_size = 20
    # Initialize population
    population = [list(seed_net)]
    
    # Add some random mutations to seed
    for _ in range(pop_size - 1):
        population.append(mutate_network(seed_net, mutation_rate=0.2))
        
    # Track best
    best_net = None
    best_score = float('inf')
    
    # Pre-evaluate initial population
    for i, net in enumerate(population):
        if is_sorted_network(net):
            score = len(net)
            if score < best_score:
                best_score = score
                best_net = list(net)
        else:
            # Assign infinite score
            pass
            
    if best_net is None:
        # If seed is broken (shouldn't happen), return seed
        return seed_net

    start_time = time.time()
    
    while time.time() < deadline - 5:
        # Selection: Tournament
        parents = []
        for _ in range(pop_size // 2):
            # Pick 2 random, take the valid one with smaller size
            c1, c2 = random.sample(population, 2)
            s1 = len(c1) if is_sorted_network(c1) else float('inf')
            s2 = len(c2) if is_sorted_network(c2) else float('inf')
            
            if s1 < s2:
                parents.append(c1)
            elif s2 < s1:
                parents.append(c2)
            else:
                parents.append(random.choice([c1, c2]))
                
        # Crossover and Mutation
        new_population = []
        for i in range(0, len(parents) - 1, 2):
            p1 = parents[i]
            p2 = parents[i+1]
            
            child = crossover(p1, p2)
            child = mutate_network(child, mutation_rate=0.1)
            
            # Ensure child is not empty
            if not child:
                child = list(p1)
                
            new_population.append(child)
            
        # Add best individual (elitism)
        # We need to identify the best in current population
        # To save time, we don't re-evaluate everyone, just add the known best
        new_population.append(list(best_net))
        
        # Fill rest with random or mutated best
        while len(new_population) < pop_size:
            new_population.append(mutate_network(best_net, mutation_rate=0.3))
            
        population = new_population
        
        # Evaluate new population and update best
        for net in population:
            if is_sorted_network(net):
                score = len(net)
                if score < best_score:
                    best_score = score
                    best_net = list(net)
                    
    return best_net

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Start with the seed odd-even transposition network
    seed = [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
    
    # Verify seed
    if not is_sorted_network(seed):
        raise ValueError("Seed network is not correct")
        
    best = genetic_algorithm(seed, deadline)
    
    # Final polish: try to remove redundant comparators from the GA result
    if time.time() < deadline - 2:
        i = 0
        while i < len(best):
            if time.time() > deadline - 2:
                break
            candidate = best[:i] + best[i+1:]
            if is_sorted_network(candidate):
                best = candidate
            else:
                i += 1
                
    return best

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
