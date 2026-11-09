"""Size-efficient sorting network search on 16 wires.

Strategy:
  - Build a candidate network as a sequence of parallel layers.
  - Each layer is a set of disjoint comparators (a matching on 16 wires).
  - Use a genetic algorithm to evolve the layer sequence.
  - Fitness is based on the number of zero-one principle violations
    (inputs that are not sorted after passing through the network).
  - To keep within the 90s wall-clock limit, we use a time budget and
    return the best valid network found, or a fallback if none is valid.
"""

import time
import random
import numpy as np

N = 16
TIME_LIMIT = 85.0  # safety margin below 90s

# Precompute all 2^16 zero-one inputs as a bit array for speed.
# We represent each input as a 16-bit integer.
ALL_INPUTS = np.arange(2**N, dtype=np.uint32)

def _apply_comparator_layer(state, layer):
    """Apply a layer of disjoint comparators to a batch of states.
    
    state: array of shape (batch, N) with values 0 or 1 (uint8)
    layer: list of (i, j) pairs, disjoint
    Returns new state array.
    """
    new_state = state.copy()
    for i, j in layer:
        a = state[:, i]
        b = state[:, j]
        new_state[:, i] = np.minimum(a, b)
        new_state[:, j] = np.maximum(a, b)
    return new_state

def _apply_network(state, network):
    """Apply a full network (list of layers) to a batch of states."""
    for layer in network:
        state = _apply_comparator_layer(state, layer)
    return state

def _is_sorted(state):
    """Check if each row of state is sorted non-decreasing."""
    # For 0/1 inputs, sorted means all 0s then all 1s.
    # Equivalent: no 1 appears before a 0 in each row.
    # We can check: for each row, the sequence is non-decreasing.
    diffs = np.diff(state, axis=1)
    # If any diff is -1, it's not sorted.
    return np.all(diffs >= 0, axis=1)

def _count_violations(network, subset=None):
    """Count number of zero-one inputs that are NOT sorted after network."""
    if subset is None:
        subset = ALL_INPUTS
    # Convert subset to state array: each bit is a wire value.
    # Wire i corresponds to bit i (LSB is wire 0).
    bits = np.arange(N, dtype=np.uint32)
    # state shape: (len(subset), N)
    state = ((subset[:, None] >> bits[None, :]) & 1).astype(np.uint8)
    state = _apply_network(state, network)
    sorted_mask = _is_sorted(state)
    return int(np.sum(~sorted_mask))

def _random_layer():
    """Generate a random matching (set of disjoint comparators) on 16 wires."""
    wires = list(range(N))
    random.shuffle(wires)
    layer = []
    for k in range(0, N, 2):
        i, j = wires[k], wires[k+1]
        if i > j:
            i, j = j, i
        layer.append((i, j))
    return layer

def _random_network(num_layers=None):
    """Generate a random network with a random number of layers."""
    if num_layers is None:
        num_layers = random.randint(10, 30)
    return [_random_layer() for _ in range(num_layers)]

def _network_to_comparators(network):
    """Flatten a list of layers into a list of (i,j) pairs."""
    comps = []
    for layer in network:
        comps.extend(layer)
    return comps

def _evaluate_network(network):
    """Evaluate a network: return (num_violations, num_comparators)."""
    num_comps = sum(len(layer) for layer in network)
    # Use a subset for speed during evolution
    violations = _count_violations(network, subset=ALL_INPUTS)
    return violations, num_comps

def _fitness(network):
    """Fitness: lower is better. Combine violations and comparator count."""
    violations, num_comps = _evaluate_network(network)
    # Penalize violations heavily
    return violations * 10000 + num_comps

def _crossover(net1, net2):
    """Uniform crossover between two networks (lists of layers)."""
    child = []
    # Pad to same length
    max_len = max(len(net1), len(net2))
    for i in range(max_len):
        if i < len(net1) and random.random() < 0.5:
            child.append(net1[i])
        elif i < len(net2):
            child.append(net2[i])
        else:
            child.append(net1[i])
    return child

def _mutate(network, mutation_rate=0.3):
    """Mutate a network: add, remove, or replace layers."""
    child = [layer[:] for layer in network]
    for i in range(len(child)):
        if random.random() < mutation_rate:
            child[i] = _random_layer()
    # Randomly add a layer
    if random.random() < 0.2:
        pos = random.randint(0, len(child))
        child.insert(pos, _random_layer())
    # Randomly remove a layer
    if len(child) > 1 and random.random() < 0.2:
        pos = random.randint(0, len(child) - 1)
        child.pop(pos)
    return child

def _ga_search(time_limit):
    """Genetic algorithm search for a valid sorting network."""
    start_time = time.time()
    
    # Initialize population
    pop_size = 30
    population = [_random_network(random.randint(10, 25)) for _ in range(pop_size)]
    
    best_network = None
    best_fitness = float('inf')
    
    # Evaluate initial population
    for net in population:
        fit = _fitness(net)
        if fit < best_fitness:
            best_fitness = fit
            best_network = net
    
    generation = 0
    while time.time() - start_time < time_limit:
        generation += 1
        
        # Select parents (tournament selection)
        new_population = []
        for _ in range(pop_size):
            # Tournament of 3
            candidates = random.sample(population, 3)
            winners = min(candidates, key=lambda x: _fitness(x))
            # Clone or crossover
            if random.random() < 0.3 and len(population) > 1:
                other = random.choice(population)
                child = _crossover(winners, other)
            else:
                child = [layer[:] for layer in winners]
            child = _mutate(child)
            new_population.append(child)
        
        # Evaluate new population and keep best
        for net in new_population:
            fit = _fitness(net)
            if fit < best_fitness:
                best_fitness = fit
                best_network = net
        
        # Replace population with new one, but keep the best
        population = new_population
        # Ensure best is in population
        if best_network is not None:
            population[0] = best_network
    
    return best_network, best_fitness

def _verify_full(network):
    """Full verification of network against all 65536 inputs."""
    violations = _count_violations(network, subset=ALL_INPUTS)
    return violations == 0

def construct():
    """Construct a sorting network for 16 wires.
    
    Returns a list of (i, j) comparator pairs.
    """
    start_time = time.time()
    
    # First, try to find a valid network via GA
    best_network, best_fitness = _ga_search(TIME_LIMIT - 5.0)
    
    if best_network is not None:
        # Verify the best network
        if _verify_full(best_network):
            return _network_to_comparators(best_network)
    
    # Fallback: odd-even transposition network (120 comparators, guaranteed correct)
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
