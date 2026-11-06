"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import numpy as np

N = 17
MAX_COMPARATORS = 4096
TIME_LIMIT = 55.0  # Leave some buffer for 60s limit

# EVOLVE-BLOCK-START

def _apply_comparator(arr, i, j):
    """Apply a single comparator to an array."""
    if arr[i] > arr[j]:
        arr[i], arr[j] = arr[j], arr[i]

def _verify_network(layers):
    """Verify the network sorts all 2^N binary inputs using the zero-one principle."""
    n = N
    total = 1 << n
    for mask in range(total):
        arr = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.uint8)
        for layer in layers:
            for i, j in layer:
                if arr[i] > arr[j]:
                    arr[i], arr[j] = arr[j], arr[i]
        if not np.all(arr[:-1] <= arr[1:]):
            return False
    return True

def _verify_network_fast(layers):
    """Faster verification using vectorized numpy operations."""
    n = N
    total = 1 << n
    
    # Precompute all input vectors as a 2D array: shape (total, n)
    masks = np.array([[ (mask >> i) & 1 for i in range(n)] for mask in range(total)], dtype=np.uint8)
    
    # Process each layer
    for layer in layers:
        # For each comparator in the layer, apply to all rows simultaneously
        for i, j in layer:
            # Find rows where arr[i] > arr[j]
            swap_mask = masks[:, i] > masks[:, j]
            if np.any(swap_mask):
                # Swap values in those rows
                temp = masks[swap_mask, i].copy()
                masks[swap_mask, i] = masks[swap_mask, j]
                masks[swap_mask, j] = temp
    
    # Check if all rows are sorted
    # A row is sorted if arr[k] <= arr[k+1] for all k
    diffs = masks[:, 1:] - masks[:, :-1]
    return np.all(diffs >= 0)

def _count_comparators(layers):
    return sum(len(layer) for layer in layers)

def _get_depth(layers):
    return len(layers)

def _generate_random_layer(used_wires=None):
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

def _generate_layer_from_wires(wire_pairs):
    """Generate a layer from a list of wire pairs, ensuring disjointness."""
    layer = []
    used = set()
    for a, b in wire_pairs:
        if a not in used and b not in used:
            if a > b:
                a, b = b, a
            layer.append((a, b))
            used.add(a)
            used.add(b)
    return layer

def _mutate_layer(layer, mutation_type):
    """Mutate a single layer."""
    if mutation_type == 0:
        # Replace entire layer with random
        return _generate_random_layer()
    elif mutation_type == 1:
        # Remove one comparator
        if not layer:
            return layer
        idx = random.randint(0, len(layer) - 1)
        new_layer = [c for i, c in enumerate(layer) if i != idx]
        return new_layer
    elif mutation_type == 2:
        # Add one comparator
        used_wires = set()
        for a, b in layer:
            used_wires.add(a)
            used_wires.add(b)
        available = [w for w in range(N) if w not in used_wires]
        if len(available) >= 2:
            a, b = random.sample(available, 2)
            if a > b:
                a, b = b, a
            return layer + [(a, b)]
        return layer
    elif mutation_type == 3:
        # Swap endpoints of a comparator
        if not layer:
            return layer
        idx = random.randint(0, len(layer) - 1)
        a, b = layer[idx]
        # Try to find another comparator to swap with
        new_layer = list(layer)
        for j in range(len(new_layer)):
            if j == idx:
                continue
            c, d = new_layer[j]
            # Check if we can swap b and c
            if b not in [x for x, y in new_layer if x != a and y != a] and \
               c not in [x for x, y in new_layer if x != a and y != a]:
                # This is complex, just skip for now
                pass
        return new_layer
    else:
        return layer

def _mutate_network(layers, mutation_rate=0.3):
    """Mutate a network by modifying some layers."""
    new_layers = [list(layer) for layer in layers]
    for i in range(len(new_layers)):
        if random.random() < mutation_rate:
            mut_type = random.randint(0, 3)
            new_layers[i] = _mutate_layer(new_layers[i], mut_type)
    return [tuple(layer) for layer in new_layers]

def _optimize_network(initial_layers, time_limit=50.0):
    """Optimize the network using a hill-climbing approach with layer-level mutations."""
    start_time = time.time()
    
    # Start with the seed network
    best_network = [list(layer) for layer in initial_layers]
    best_count = _count_comparators(best_network)
    
    # Verify the seed network
    if not _verify_network_fast(best_network):
        return initial_layers
    
    best_overall = [list(layer) for layer in best_network]
    best_overall_count = best_count
    
    # Local search: try removing comparators from random layers
    while time.time() - start_time < time_limit:
        # Try to improve by removing one comparator from a random layer
        improved = False
        num_layers = len(best_network)
        
        # Try removing comparators from each layer
        for layer_idx in range(num_layers):
            layer = best_network[layer_idx]
            for comp_idx in range(len(layer)):
                # Create a candidate with this comparator removed
                candidate = [list(l) for l in best_network]
                candidate[layer_idx] = [c for i, c in enumerate(layer) if i != comp_idx]
                
                # Verify
                if _verify_network_fast(candidate):
                    count = _count_comparators(candidate)
                    if count < best_overall_count:
                        best_overall = candidate
                        best_overall_count = count
                        best_network = candidate
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Try adding a comparator to a random layer (to enable further removals)
            layer_idx = random.randint(0, num_layers - 1)
            layer = best_network[layer_idx]
            used_wires = set()
            for a, b in layer:
                used_wires.add(a)
                used_wires.add(b)
            available = [w for w in range(N) if w not in used_wires]
            if len(available) >= 2:
                a, b = random.sample(available, 2)
                if a > b:
                    a, b = b, a
                
                candidate = [list(l) for l in best_network]
                candidate[layer_idx] = layer + [(a, b)]
                
                if _verify_network_fast(candidate):
                    # This might enable future improvements, keep it temporarily
                    # But don't update best if count increased
                    pass
        
        # Occasionally try a larger mutation
        if random.random() < 0.1:
            candidate = _mutate_network(best_network, mutation_rate=0.2)
            if _verify_network_fast(candidate):
                count = _count_comparators(candidate)
                if count < best_overall_count:
                    best_overall = [list(l) for l in candidate]
                    best_overall_count = count
                    best_network = [list(l) for l in candidate]
    
    return best_overall

def construct():
    # Generate the seed network
    seed_layers = _generate_seed_network()
    
    # Verify it's correct
    if not _verify_network_fast(seed_layers):
        # Should not happen
        return [(i, j) for layer in seed_layers for i, j in layer]
    
    # Try to optimize
    try:
        optimized_layers = _optimize_network(seed_layers, time_limit=TIME_LIMIT)
        
        # Verify the optimized network
        if _verify_network_fast(optimized_layers):
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
