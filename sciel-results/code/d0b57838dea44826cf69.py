"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np
import time

N = 17

def _apply_comparator(state, i, j):
    """Apply a compare-exchange to a state vector (in-place)."""
    if state[i] > state[j]:
        state[i], state[j] = state[j], state[i]
    return state

def _evaluate_network_layers(layers, samples=None, sample_size=1024):
    """
    Evaluate a sorting network (list of layers) using the zero-one principle with random Boolean inputs.
    Returns True if the network sorts all sampled inputs.
    """
    if samples is None:
        samples = np.random.randint(0, 2, size=(sample_size, N))
    
    for sample in samples:
        state = sample.copy()
        for layer in layers:
            for i, j in layer:
                _apply_comparator(state, i, j)
        # Check if sorted
        if not np.all(state[:-1] <= state[1:]):
            return False
    return True

def _evaluate_network_full_layers(layers):
    """
    Evaluate a sorting network over ALL 2^17 Boolean inputs (zero-one principle).
    Returns True if the network sorts all inputs.
    """
    for mask in range(1 << N):
        state = np.array([(mask >> i) & 1 for i in range(N)], dtype=np.int8)
        for layer in layers:
            for i, j in layer:
                _apply_comparator(state, i, j)
        if not np.all(state[:-1] <= state[1:]):
            return False
    return True

def _flat_to_layers(flat_comparators):
    """Convert a flat list of comparators into a list of layers (matchings)."""
    if not flat_comparators:
        return []
    
    layers = []
    # We use a greedy approach to assign comparators to the earliest possible layer
    # where both wires are free.
    # To make it deterministic and simple, we just group them by their index in the flat list?
    # No, that doesn't guarantee parallelism.
    # Standard method: iterate through comparators, assign to first layer where i and j are not used.
    
    # However, for the GA, we want to maintain the structure.
    # The seed is already parallel.
    # For arbitrary flat lists, we need to reconstruct layers.
    # This function is primarily for initializing the population from the seed.
    
    # A better way for the GA is to represent the network directly as layers.
    # If we receive a flat list from crossover/mutation that might not be valid layers,
    # we should probably just work with layers throughout.
    
    # Let's assume the input is a valid sequence of comparators.
    # We will assign each comparator to a layer index.
    # Layer 0: first comparator.
    # Layer k: next comparator if wires are free in layer k-1? No, that's not right.
    
    # Actually, the easiest way to initialize from the seed (which is already layered)
    # is to just parse it.
    # But for general flat lists, we can just put each comparator in its own layer?
    # That would be depth = len(comparators).
    # Or we can try to pack them.
    
    # Let's implement a simple packer:
    used_in_layer = {}
    for comp in flat_comparators:
        placed = False
        for l_idx in range(len(layers)):
            i, j = comp
            if i not in used_in_layer.get(l_idx, set()) and j not in used_in_layer.get(l_idx, set()):
                layers[l_idx].append(comp)
                used_in_layer[l_idx].add(i)
                used_in_layer[l_idx].add(j)
                placed = True
                break
        if not placed:
            layers.append([comp])
            used_in_layer[len(layers)-1] = {comp[0], comp[1]}
            
    return layers

def _layers_to_flat(layers):
    """Convert layers to a flat list of comparators."""
    flat = []
    for layer in layers:
        flat.extend(layer)
    return flat

def _generate_seed_layers():
    """Generate the odd-even transposition network as layers."""
    layers = []
    for phase in range(N):
        layer = []
        for i in range(phase % 2, N - 1, 2):
            layer.append((i, i + 1))
        layers.append(layer)
    return layers

def _mutate_layers(layers, mutation_rate=0.2):
    """
    Apply mutations to the layer-based network.
    """
    new_layers = [list(layer) for layer in layers]
    n_comps = sum(len(l) for l in new_layers)
    
    num_mutations = max(1, int(mutation_rate * n_comps))
    
    for _ in range(num_mutations):
        op = random.choice(['remove', 'insert', 'move', 'swap_pos'])
        
        if op == 'remove' and n_comps > 0:
            # Pick a random comparator
            l_idx = random.randint(0, len(new_layers) - 1)
            if not new_layers[l_idx]:
                continue
            c_idx = random.randint(0, len(new_layers[l_idx]) - 1)
            new_layers[l_idx].pop(c_idx)
            n_comps -= 1
            # Remove empty layers? Maybe keep them for structure, or remove to save depth.
            # Let's remove empty layers to encourage compactness.
            if not new_layers[l_idx]:
                new_layers.pop(l_idx)
                
        elif op == 'insert' and n_comps < 200: # Cap size
            # Pick a random wire pair
            i, j = random.sample(range(N), 2)
            if i > j: i, j = j, i
            # Try to insert into a random layer where it fits
            inserted = False
            if new_layers:
                # Try random layer
                l_idx = random.randint(0, len(new_layers) - 1)
                if _can_add_to_layer(new_layers[l_idx], i, j):
                    new_layers[l_idx].append((i, j))
                    inserted = True
                    n_comps += 1
            if not inserted:
                # Add to a new layer at the end
                new_layers.append([(i, j)])
                n_comps += 1
                
        elif op == 'move':
            # Move a comparator to a different layer
            if n_comps > 1 and len(new_layers) > 1:
                l_idx = random.randint(0, len(new_layers) - 1)
                if not new_layers[l_idx]:
                    continue
                c_idx = random.randint(0, len(new_layers[l_idx]) - 1)
                comp = new_layers[l_idx][c_idx]
                # Try to move to a different layer
                target_l_idx = random.randint(0, len(new_layers) - 1)
                if target_l_idx != l_idx:
                    # Temporarily remove
                    new_layers[l_idx].pop(c_idx)
                    if _can_add_to_layer(new_layers[target_l_idx], comp[0], comp[1]):
                        new_layers[target_l_idx].append(comp)
                    else:
                        # Put it back
                        new_layers[l_idx].append(comp)
                    # Clean up empty source layer
                    if not new_layers[l_idx]:
                         new_layers.pop(l_idx)
                         # Note: indices might have shifted, but we are done with this op
                        
        elif op == 'swap_pos':
            # Swap two comparators in the same layer or adjacent layers?
            # Let's just swap positions within a layer (changes order, but comparators in a layer are parallel so order doesn't matter for execution, but might matter for our representation if we flatten).
            # Actually, swapping within a layer is a no-op for the network function.
            # Swapping between layers is the same as 'move' if we just swap the lists?
            # Let's implement 'shift': move a comparator to an earlier or later layer if possible.
            if n_comps > 0:
                l_idx = random.randint(0, len(new_layers) - 1)
                if not new_layers[l_idx]:
                    continue
                c_idx = random.randint(0, len(new_layers[l_idx]) - 1)
                comp = new_layers[l_idx][c_idx]
                direction = random.choice([-1, 1])
                target_l_idx = l_idx + direction
                if 0 <= target_l_idx < len(new_layers):
                    new_layers[l_idx].pop(c_idx)
                    if _can_add_to_layer(new_layers[target_l_idx], comp[0], comp[1]):
                        new_layers[target_l_idx].append(comp)
                    else:
                        new_layers[l_idx].append(comp)
                    if not new_layers[l_idx]:
                         new_layers.pop(l_idx)

    # Clean up empty layers
    new_layers = [l for l in new_layers if l]
    
    return new_layers

def _can_add_to_layer(layer, i, j):
    """Check if comparator (i, j) can be added to the layer without conflict."""
    for a, b in layer:
        if a == i or a == j or b == i or b == j:
            return False
    return True

def _crossover_layers(parent1, parent2):
    """
    Crossover: take a prefix of layers from parent1 and suffix from parent2.
    """
    if not parent1 or not parent2:
        return [list(l) for l in (parent1 or parent2)]
    
    # Determine split point in terms of layers
    # We can't just split layers arbitrarily because a layer is a set of parallel ops.
    # We must take whole layers.
    max_len = max(len(parent1), len(parent2))
    split = random.randint(0, max_len)
    
    child = []
    for i in range(len(parent1)):
        if i < split:
            child.append(list(parent1[i]))
        else:
            break
            
    for i in range(len(parent2)):
        if i >= split:
            child.append(list(parent2[i]))
        else:
            continue
            
    # If split was 0, we take all from parent2.
    # If split > len(parent1), we take all from parent1 and then rest from parent2?
    # The logic above:
    # parent1 layers 0..split-1
    # parent2 layers split..end
    # This assumes layer indices align.
    
    return child

def _local_search_layers(layers, max_iters=20, sample_size=512):
    """
    Local search: try removing comparators.
    """
    current_layers = [list(l) for l in layers]
    changed = True
    iters = 0
    while changed and iters < max_iters:
        changed = False
        iters += 1
        # Try removing each comparator from each layer
        for l_idx in range(len(current_layers) - 1, -1, -1):
            if not current_layers[l_idx]:
                continue
            for c_idx in range(len(current_layers[l_idx]) - 1, -1, -1):
                # Create candidate
                candidate_layers = [list(l) for l in current_layers]
                candidate_layers[l_idx].pop(c_idx)
                # Remove empty layer
                if not candidate_layers[l_idx]:
                    candidate_layers.pop(l_idx)
                
                # Quick check
                if _evaluate_network_layers(candidate_layers, sample_size=sample_size):
                    current_layers = candidate_layers
                    changed = True
                    break
            if changed:
                break
    return current_layers

def construct():
    """
    Construct a sorting network using a genetic algorithm with layer representation.
    Returns a list of (i, j) comparator pairs.
    """
    start_time = time.time()
    max_time = 55  # Leave some buffer
    
    # Initialize with seed network layers
    best_layers = _generate_seed_layers()
    best_flat = _layers_to_flat(best_layers)
    best_len = len(best_flat)
    
    # Population
    pop_size = 20
    population = [best_layers]
    for _ in range(pop_size - 1):
        mutated = _mutate_layers(best_layers, mutation_rate=0.2)
        population.append(mutated)
    
    # Genetic algorithm loop
    generation = 0
    while time.time() - start_time < max_time:
        generation += 1
        
        # Evaluate and select
        scored = []
        for comp_layers in population:
            # Quick evaluation with random samples
            if _evaluate_network_layers(comp_layers, sample_size=512):
                flat = _layers_to_flat(comp_layers)
                score = len(flat) # We want to minimize, so smaller is better
            else:
                score = 10000
            scored.append((score, comp_layers))
        
        scored.sort(key=lambda x: x[0])  # Best first (smallest count)
        
        # Update best
        if scored[0][0] < 10000:  # Valid
            best_comp_layers = scored[0][1]
            best_flat_candidate = _layers_to_flat(best_comp_layers)
            if len(best_flat_candidate) < best_len:
                # Full verification
                if _evaluate_network_full_layers(best_comp_layers):
                    best_layers = best_comp_layers
                    best_flat = best_flat_candidate
                    best_len = len(best_flat)
        
        # Create next generation
        new_population = []
        
        # Elitism: keep best few
        for i in range(min(5, len(scored))):
            if scored[i][0] < 10000:
                new_population.append(scored[i][1])
        
        # Crossover and mutation
        while len(new_population) < pop_size:
            if len(new_population) < 2:
                parent1 = best_layers
                parent2 = best_layers
            else:
                # Tournament selection
                candidates = random.sample(scored, min(3, len(scored)))
                candidates.sort(key=lambda x: x[0])
                parent1 = candidates[0][1]
                parent2 = candidates[1][1] if len(candidates) > 1 else best_layers
            
            child = _crossover_layers(parent1, parent2)
            child = _mutate_layers(child, mutation_rate=0.15)
            new_population.append(child)
        
        population = new_population
        
        # Local search on best candidate occasionally
        if generation % 5 == 0 and time.time() - start_time < max_time - 10:
            local_result = _local_search_layers(best_layers, max_iters=20)
            local_flat = _layers_to_flat(local_result)
            if len(local_flat) < best_len:
                if _evaluate_network_full_layers(local_result):
                    best_layers = local_result
                    best_flat = local_flat
                    best_len = len(best_flat)
    
    return best_flat

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
