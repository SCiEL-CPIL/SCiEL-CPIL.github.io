"""Improved program for size-efficient sorting networks on 17 wires."""
import random
import time

N = 17
MAX_COMPARATORS = 4096
TIME_LIMIT = 55.0  # Leave some buffer for 60s limit

# EVOLVE-BLOCK-START

def _verify_network_bitmask(layers):
    """Verify the network sorts all 2^N binary inputs using bitmask representation.
    
    Each input is represented as an integer where bit i indicates wire i.
    A comparator (i, j) with i < j swaps bits if bit i is 1 and bit j is 0.
    """
    n = N
    for mask in range(1 << n):
        bits = mask
        for layer in layers:
            new_bits = bits
            for i, j in layer:
                bi = (bits >> i) & 1
                bj = (bits >> j) & 1
                if bi and not bj:
                    # Set bit i to 0, bit j to 1
                    new_bits = (new_bits & ~(1 << i)) | (1 << j)
            bits = new_bits
        # Check if sorted: all 0s before all 1s
        # Count the number of 1s
        num_ones = bin(bits).count('1')
        # Sorted means the lowest num_ones bits are 0 and the rest are 1
        # Actually, sorted ascending means: wire 0 has smallest, wire N-1 has largest
        # So all 0s should be on the left (lower indices), all 1s on the right
        # Expected sorted value: (1 << n) - (1 << num_ones) if we want 1s on top
        # Wait, let me think again. If we have k ones, sorted means:
        # wires 0 to n-k-1 are 0, wires n-k to n-1 are 1
        # So the sorted bitmask should be: ((1 << n) - 1) ^ ((1 << (n - num_ones)) - 1)
        # = (1 << n) - (1 << (n - num_ones))
        expected = (1 << n) - (1 << (n - num_ones))
        if bits != expected:
            return False
    return True

def _count_comparators(layers):
    return sum(len(layer) for layer in layers)

def _get_depth(layers):
    return len(layers)

def _generate_random_layer():
    """Generate a random layer of disjoint comparators."""
    wires = list(range(N))
    random.shuffle(wires)
    layer = []
    i = 0
    while i < N - 1:
        a = wires[i]
        b = wires[i + 1]
        if a > b:
            a, b = b, a
        layer.append((a, b))
        i += 2
    return layer

def _generate_seed_network():
    """Generate the odd-even transposition network as layers."""
    layers = []
    for phase in range(N):
        layer = []
        for i in range(phase % 2, N - 1, 2):
            layer.append((i, i + 1))
        layers.append(layer)
    return layers

def _mutate_network(layers, mutation_rate=0.3):
    """Mutate a network by modifying some layers."""
    new_layers = [list(layer) for layer in layers]
    for i in range(len(new_layers)):
        if random.random() < mutation_rate:
            if random.random() < 0.5:
                # Replace layer with a new random layer
                new_layers[i] = _generate_random_layer()
            else:
                # Modify a single comparator in the layer
                if new_layers[i]:
                    idx = random.randint(0, len(new_layers[i]) - 1)
                    # Remove old comparator
                    new_layers[i].pop(idx)
                    # Add a new random comparator ensuring disjointness
                    used_wires = set()
                    for a, b in new_layers[i]:
                        used_wires.add(a)
                        used_wires.add(b)
                    available = [w for w in range(N) if w not in used_wires]
                    if len(available) >= 2:
                        a, b = random.sample(available, 2)
                        if a > b:
                            a, b = b, a
                        new_layers[i].append((a, b))
                    else:
                        # Just regenerate the layer
                        new_layers[i] = _generate_random_layer()
    return [tuple(layer) for layer in new_layers]

def _crossover_network(layers1, layers2):
    """Crossover two networks."""
    if len(layers1) != len(layers2):
        # Pad the shorter one
        shorter, longer = (layers1, layers2) if len(layers1) < len(layers2) else (layers2, layers1)
        while len(shorter) < len(longer):
            shorter.append(_generate_random_layer())
        layers1, layers2 = (shorter, longer) if len(layers1) <= len(layers2) else (longer, shorter)
    
    new_layers = []
    for i in range(len(layers1)):
        if random.random() < 0.5:
            new_layers.append(list(layers1[i]))
        else:
            new_layers.append(list(layers2[i]))
    return [tuple(layer) for layer in new_layers]

def _optimize_network(initial_layers, time_limit=50.0):
    """Optimize the network using a simple evolutionary approach."""
    start_time = time.time()
    
    # Start with the seed network
    best_network = [list(layer) for layer in initial_layers]
    best_count = _count_comparators(best_network)
    
    # Verify the seed network
    if not _verify_network_bitmask(best_network):
        # Should not happen, but just in case
        return initial_layers
    
    # Create a population
    population_size = 10
    population = [best_network]
    for _ in range(population_size - 1):
        mutated = _mutate_network(best_network, mutation_rate=0.5)
        population.append(mutated)
    
    best_overall = best_network
    best_overall_count = best_count
    
    while time.time() - start_time < time_limit:
        # Evaluate population
        valid_population = []
        for net in population:
            if _verify_network_bitmask(net):
                count = _count_comparators(net)
                valid_population.append((count, net))
        
        if not valid_population:
            # No valid networks, restart from best
            population = [best_network]
            for _ in range(population_size - 1):
                mutated = _mutate_network(best_network, mutation_rate=0.5)
                population.append(mutated)
            continue
        
        # Sort by count
        valid_population.sort(key=lambda x: x[0])
        best_count, best_net = valid_population[0]
        
        if best_count < best_overall_count:
            best_overall = [list(layer) for layer in best_net]
            best_overall_count = best_count
            best_network = [list(layer) for layer in best_net]
        
        # Create next generation
        new_population = []
        # Elitism: keep the best
        new_population.append([list(layer) for layer in best_net])
        
        # Crossover and mutation
        while len(new_population) < population_size:
            if len(valid_population) >= 2 and random.random() < 0.7:
                # Crossover
                idx1 = random.randint(0, min(len(valid_population) - 1, 4))
                idx2 = random.randint(0, min(len(valid_population) - 1, 4))
                child = _crossover_network(valid_population[idx1][1], valid_population[idx2][1])
                child = _mutate_network(child, mutation_rate=0.3)
            else:
                # Mutation
                parent = valid_population[random.randint(0, min(len(valid_population) - 1, 4))][1]
                child = _mutate_network(parent, mutation_rate=0.5)
            
            new_population.append(child)
        
        population = new_population
    
    return best_overall

def construct():
    # Generate the seed network
    seed_layers = _generate_seed_network()
    
    # Verify it's correct
    if not _verify_network_bitmask(seed_layers):
        # Should not happen
        return [(i, j) for layer in seed_layers for i, j in layer]
    
    # Try to optimize
    try:
        optimized_layers = _optimize_network(seed_layers, time_limit=TIME_LIMIT)
        
        # Verify the optimized network
        if _verify_network_bitmask(optimized_layers):
            # Convert layers to flat list of comparators
            comparators = []
            for layer in optimized_layers:
                for comp in layer:
                    comparators.append(comp)
            
            # Check constraint
            if len(comparators) <= MAX_COMPARATORS:
                return comparators
    except Exception:
        pass
    
    # Fallback to seed network
    comparators = []
    for layer in seed_layers:
        for comp in layer:
            comparators.append(comp)
    return comparators

# EVOLVE-BLOCK-END
