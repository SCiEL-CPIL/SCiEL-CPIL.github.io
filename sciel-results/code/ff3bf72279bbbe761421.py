"""Depth-based genetic algorithm for sorting networks on 17 wires."""
import numpy as np
import random
import sys
import time

N = 17
MAX_COMPARATORS = 4096
TIME_LIMIT = 55  # seconds, leaving buffer for the 60s limit

def _verify_network_fast(comparators):
    """
    Verify a sorting network using the Zero-One Principle.
    Uses a compact representation: for each wire, maintain a bitmask of which
    of the 2^N inputs have a 1 on that wire.
    Returns True if the network sorts all 2^N boolean inputs.
    """
    num_inputs = 1 << N
    # masks[i] is a bitmask: bit k is 1 iff input k has value 1 on wire i
    # Initially, for input k, wire j has value 1 iff (k >> j) & 1
    # So masks[j] has bits set at positions k where (k>>j)&1 = 1
    masks = [0] * N
    for j in range(N):
        # Build mask for wire j: all inputs k where bit j of k is 1
        mask = 0
        for k in range(num_inputs):
            if (k >> j) & 1:
                mask |= (1 << k)
        masks[j] = mask
    
    for (i, j) in comparators:
        a = masks[i]
        b = masks[j]
        # min: 1 only where both are 1 -> a & b
        # max: 1 where at least one is 1 -> a | b
        masks[i] = a & b
        masks[j] = a | b
    
    # Check sorted: for each input k, wire 0 >= wire 1 >= ... >= wire N-1
    # Equivalently: for each adjacent pair (i, i+1), no input k has wire i = 0 and wire i+1 = 1
    # i.e., masks[i] should be a superset of masks[i+1] (bitwise: masks[i] | masks[i+1] == masks[i])
    for i in range(N - 1):
        if (masks[i] | masks[i+1]) != masks[i]:
            return False
    return True

def _verify_network(comparators):
    """Compatibility wrapper."""
    return _verify_network_fast(comparators)

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _flatten_layers(layers):
    """Flatten a list of layers (each layer is a list of comparators) into a single list."""
    comps = []
    for layer in layers:
        comps.extend(layer)
    return comps

def _layers_to_comparators(layers):
    """Convert layers representation to flat comparator list."""
    return _flatten_layers(layers)

def _comparators_to_layers(comparators):
    """
    Convert a flat list of comparators to a layered representation.
    Greedy: assign each comparator to the earliest layer where it doesn't conflict.
    """
    layers = []
    for comp in comparators:
        placed = False
        for layer_idx, layer in enumerate(layers):
            # Check if comp conflicts with any comparator in this layer
            conflict = False
            for (i, j) in layer:
                if i == comp[0] or i == comp[1] or j == comp[0] or j == comp[1]:
                    conflict = True
                    break
            if not conflict:
                layer.append(comp)
                placed = True
                break
        if not placed:
            layers.append([comp])
    return layers

def _generate_random_layers(max_depth=12, max_per_layer=8):
    """Generate a random layered network."""
    depth = random.randint(8, max_depth)
    layers = []
    for _ in range(depth):
        # Generate a random set of non-conflicting comparators
        wires_used = set()
        layer = []
        max_comp = random.randint(2, max_per_layer)
        for _ in range(max_comp):
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            if i not in wires_used and j not in wires_used:
                layer.append((i, j))
                wires_used.add(i)
                wires_used.add(j)
        if layer:
            layers.append(layer)
    return layers

def _crossover_layers(parent1, parent2):
    """Crossover: take a random prefix of layers from parent1 and suffix from parent2."""
    if not parent1 or not parent2:
        return list(parent1) if parent1 else list(parent2)
    # Pad to same length for crossover
    max_len = max(len(parent1), len(parent2))
    point = random.randint(0, max_len)
    child = []
    for i in range(max_len):
        if i < point:
            child.append(list(parent1[i]) if i < len(parent1) else [])
        else:
            child.append(list(parent2[i]) if i < len(parent2) else [])
    # Remove empty layers
    child = [layer for layer in child if layer]
    return child

def _mutate_layers(layers, mutation_rate=0.1):
    """Mutate a layered network."""
    layers = [list(layer) for layer in layers]
    n_mutations = max(1, int(len(layers) * mutation_rate) + random.randint(0, 2))
    for _ in range(n_mutations):
        op = random.choice(['add_comp', 'remove_comp', 'move_comp', 'add_layer', 'remove_layer'])
        if op == 'add_comp':
            # Add a comparator to a random layer
            if layers:
                layer_idx = random.randint(0, len(layers) - 1)
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                # Check conflict
                conflict = False
                for (a, b) in layers[layer_idx]:
                    if i in (a, b) or j in (a, b):
                        conflict = True
                        break
                if not conflict:
                    layers[layer_idx].append((i, j))
        elif op == 'remove_comp':
            # Remove a random comparator from a random layer
            non_empty = [i for i, layer in enumerate(layers) if layer]
            if non_empty:
                layer_idx = random.choice(non_empty)
                comp_idx = random.randint(0, len(layers[layer_idx]) - 1)
                layers[layer_idx].pop(comp_idx)
                if not layers[layer_idx]:
                    layers.pop(layer_idx)
        elif op == 'move_comp':
            # Move a comparator from one layer to another
            non_empty = [i for i, layer in enumerate(layers) if layer]
            if len(non_empty) >= 2:
                src_idx = random.choice(non_empty)
                comp_idx = random.randint(0, len(layers[src_idx]) - 1)
                comp = layers[src_idx].pop(comp_idx)
                # Try to place in another layer
                dst_candidates = [i for i in range(len(layers)) if i != src_idx]
                random.shuffle(dst_candidates)
                for dst_idx in dst_candidates:
                    conflict = False
                    for (a, b) in layers[dst_idx]:
                        if comp[0] in (a, b) or comp[1] in (a, b):
                            conflict = True
                            break
                    if not conflict:
                        layers[dst_idx].append(comp)
                        break
                else:
                    # Put back if no place found
                    layers[src_idx].append(comp)
        elif op == 'add_layer':
            # Add a new empty layer with one comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(layers))
            layers.insert(pos, [(i, j)])
        elif op == 'remove_layer':
            if len(layers) > 1:
                idx = random.randint(0, len(layers) - 1)
                layers.pop(idx)
    return layers

def _count_comparators(layers):
    """Count total comparators in a layered network."""
    return sum(len(layer) for layer in layers)

def _evaluate_fitness_layers(layers):
    """
    Fitness: lower is better.
    If the network is valid, fitness = number of comparators.
    If invalid, fitness = a large penalty.
    """
    comps = _flatten_layers(layers)
    if len(comps) > MAX_COMPARATORS:
        return MAX_COMPARATORS + 1000
    if _verify_network_fast(comps):
        return len(comps)
    else:
        return MAX_COMPARATORS + 1000

def _local_search_reduce(layers, time_budget=5.0):
    """
    Greedy local search: try to remove each comparator and keep it if the network
    remains valid. Repeat until no more improvements or time budget exceeded.
    """
    start_time = time.time()
    layers = [list(layer) for layer in layers]
    improved = True
    while improved and (time.time() - start_time) < time_budget:
        improved = False
        # Try removing each comparator
        for layer_idx in range(len(layers)):
            if (time.time() - start_time) >= time_budget:
                break
            for comp_idx in range(len(layers[layer_idx])):
                if (time.time() - start_time) >= time_budget:
                    break
                # Try removing this comparator
                saved = layers[layer_idx].pop(comp_idx)
                comps = _flatten_layers(layers)
                if _verify_network_fast(comps):
                    # Keep it removed
                    improved = True
                    if not layers[layer_idx]:
                        layers.pop(layer_idx)
                        layer_idx -= 1
                    if layer_idx < 0:
                        break
                else:
                    # Put it back
                    layers[layer_idx].insert(comp_idx, saved)
                # Adjust index if we removed a layer
                if not layers[layer_idx] if layer_idx < len(layers) else True:
                    if layer_idx >= len(layers):
                        break
    return layers

def _genetic_algorithm():
    """
    Run a depth-based genetic algorithm to find a small sorting network.
    """
    start_time = time.time()
    
    # Initialize population with layered networks
    pop_size = 20
    population = []
    
    # Start with the seed network converted to layers
    seed_comps = _odd_even_transposition()
    seed_layers = _comparators_to_layers(seed_comps)
    population.append(seed_layers)
    
    # Add more seed-derived networks with random pruning
    for _ in range(5):
        pruned = []
        for comp in seed_comps:
            if random.random() > 0.15:  # Keep 85% of comparators
                pruned.append(comp)
        if pruned:
            population.append(_comparators_to_layers(pruned))
    
    # Add random layered networks
    while len(population) < pop_size:
        population.append(_generate_random_layers())
    
    # Evaluate initial population
    fitnesses = [_evaluate_fitness_layers(ind) for ind in population]
    
    best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
    best_fitness = fitnesses[best_idx]
    best_layers = population[best_idx]
    best_comps = _flatten_layers(best_layers)
    
    print(f"Initial best: {best_fitness} comparators", file=sys.stderr)
    
    # If initial best is valid, do local search to reduce
    if best_fitness < MAX_COMPARATORS:
        best_layers = _local_search_reduce(best_layers, time_budget=min(10.0, TIME_LIMIT - (time.time() - start_time)))
        best_comps = _flatten_layers(best_layers)
        best_fitness = _count_comparators(best_layers)
        if _verify_network_fast(best_comps):
            print(f"After local search: {best_fitness} comparators", file=sys.stderr)
        else:
            # Revert if local search broke it
            best_layers = population[best_idx]
            best_comps = _flatten_layers(best_layers)
            best_fitness = fitnesses[best_idx]
    
    # Genetic algorithm loop
    generation = 0
    while time.time() - start_time < TIME_LIMIT:
        generation += 1
        remaining_time = TIME_LIMIT - (time.time() - start_time)
        if remaining_time <= 0:
            break
        
        # Selection
        sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
        elite_size = max(1, pop_size // 4)
        new_population = [population[i] for i in sorted_indices[:elite_size]]
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() - start_time >= TIME_LIMIT:
                break
            if random.random() < 0.4 and len(population) >= 2:
                # Crossover
                i1 = random.choice(sorted_indices[:elite_size])
                i2 = random.choice(sorted_indices[:elite_size])
                child = _crossover_layers(population[i1], population[i2])
            else:
                # Mutation
                i1 = random.choice(sorted_indices[:elite_size])
                child = _mutate_layers(population[i1], mutation_rate=0.15)
            
            if child:
                new_population.append(child)
        
        population = new_population
        fitnesses = [_evaluate_fitness_layers(ind) for ind in population]
        
        # Update best
        gen_best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] < best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_layers = population[gen_best_idx]
            best_comps = _flatten_layers(best_layers)
            print(f"Gen {generation}: new best {best_fitness} comparators", file=sys.stderr)
            
            # Try local search on new best
            if best_fitness < MAX_COMPARATORS:
                remaining = TIME_LIMIT - (time.time() - start_time)
                if remaining > 5:
                    improved_layers = _local_search_reduce(best_layers, time_budget=min(8.0, remaining * 0.3))
                    improved_comps = _flatten_layers(improved_layers)
                    if _verify_network_fast(improved_comps) and len(improved_comps) < best_fitness:
                        best_fitness = len(improved_comps)
                        best_layers = improved_layers
                        best_comps = improved_comps
                        print(f"Local search improved to: {best_fitness} comparators", file=sys.stderr)
    
    return best_comps

def construct():
    """
    Construct a sorting network for 17 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    # Try the genetic algorithm
    try:
        network = _genetic_algorithm()
        # Verify the result
        if _verify_network_fast(network):
            return network
    except Exception as e:
        print(f"GA failed: {e}", file=sys.stderr)
    
    # Fallback to odd-even transposition
    return _odd_even_transposition()
