"""Improved sorting network search for 17 wires."""
import time
import numpy as np

N = 17
MAX_COMPARATORS = 4096

def _verify(network):
    """Verify sorting network over all 2^N Boolean inputs using zero-one principle."""
    total = 1 << N
    inputs = np.empty((total, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (np.arange(total, dtype=np.uint32) >> i) & 1
    
    for (i, j) in network:
        a = inputs[:, i].copy()
        b = inputs[:, j].copy()
        inputs[:, i] = np.minimum(a, b)
        inputs[:, j] = np.maximum(a, b)
    
    for k in range(N - 1):
        if np.any(inputs[:, k] > inputs[:, k + 1]):
            return False
    return True

def _odd_even_transposition(n):
    """Generate odd-even transposition sorting network for n wires."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def _try_remove(network, idx):
    """Try removing comparator at index idx, return (success, new_network)."""
    candidate = network[:idx] + network[idx+1:]
    if _verify(candidate):
        return True, candidate
    return False, None

def _count_comparators(network):
    """Count total comparators."""
    return len(network)

def _compute_depth(network):
    """Compute parallel depth of the network."""
    if not network:
        return 0
    # Track the earliest layer each wire is "free" after
    wire_free = [0] * N
    depth = 0
    for (i, j) in network:
        d = max(wire_free[i], wire_free[j]) + 1
        depth = max(depth, d)
        wire_free[i] = d
        wire_free[j] = d
    return depth

def _layered_network(network):
    """Convert flat network to list of layers (each layer is a set of non-conflicting comparators)."""
    layers = []
    i = 0
    while i < len(network):
        layer = []
        used_in_layer = set()
        while i < len(network):
            comp = network[i]
            if comp[0] not in used_in_layer and comp[1] not in used_in_layer:
                layer.append(comp)
                used_in_layer.add(comp[0])
                used_in_layer.add(comp[1])
                i += 1
            else:
                break
        layers.append(layer)
    return layers

def _from_layers(layers):
    """Convert layers back to flat network."""
    net = []
    for layer in layers:
        net.extend(layer)
    return net

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Start with odd-even transposition network
    network = _odd_even_transposition(N)
    
    # Verify it's correct
    if not _verify(network):
        return network
    
    best_network = list(network)
    best_count = len(network)
    
    # Phase 1: Greedy removal - repeatedly remove redundant comparators
    improved = True
    while improved:
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return best_network
            
            success, candidate = _try_remove(network, idx)
            if success:
                network = candidate
                improved = True
                best_network = list(network)
                best_count = len(network)
    
    # Phase 2: Local search with random order - try removing comparators in random order
    if time.time() < deadline:
        rng = np.random.default_rng(42)
        # Use a random permutation approach to avoid bias
        while time.time() < deadline:
            if len(network) <= best_count:
                break
            # Generate random order of indices
            order = rng.permutation(len(network))
            found_removal = False
            for pos in order:
                if time.time() > deadline:
                    return best_network
                candidate = network[:pos] + network[pos+1:]
                if _verify(candidate):
                    network = candidate
                    best_network = list(network)
                    best_count = len(network)
                    found_removal = True
                    break
            if not found_removal:
                # No more single removals possible, break
                break
    
    # Phase 3: Layer-based local search - represent as layers and try mutations
    if time.time() < deadline:
        # Convert to layer representation
        def to_layers(net):
            """Convert flat network to list of layers (each layer is a set of non-conflicting comparators)."""
            layers = []
            used = set()
            i = 0
            while i < len(net):
                layer = []
                used_in_layer = set()
                while i < len(net):
                    comp = net[i]
                    if comp[0] not in used_in_layer and comp[1] not in used_in_layer:
                        layer.append(comp)
                        used_in_layer.add(comp[0])
                        used_in_layer.add(comp[1])
                        i += 1
                    else:
                        break
                layers.append(layer)
            return layers
        
        def from_layers(layers):
            """Convert layers back to flat network."""
            net = []
            for layer in layers:
                net.extend(layer)
            return net
        
        # Try to reduce depth by merging layers where possible
        layers = to_layers(network)
        
        # Try swapping comparators between adjacent layers to enable removals
        for _pass in range(3):
            if time.time() > deadline:
                break
            for l in range(len(layers) - 1):
                if time.time() > deadline:
                    break
                # Try moving a comparator from layer l to layer l+1 or vice versa
                for i in range(len(layers[l])):
                    if time.time() > deadline:
                        break
                    comp = layers[l][i]
                    # Try removing from layer l and adding to layer l+1
                    new_layer_l = layers[l][:i] + layers[l][i+1:]
                    # Check if comp can fit in layer l+1
                    can_fit = True
                    for c2 in layers[l+1]:
                        if comp[0] in c2 or comp[1] in c2:
                            can_fit = False
                            break
                    if can_fit:
                        new_layer_l1 = layers[l+1] + [comp]
                        # Reconstruct network
                        new_layers = layers[:l] + [new_layer_l] + [new_layer_l1] + layers[l+2:]
                        candidate = from_layers(new_layers)
                        if len(candidate) < len(network):
                            if _verify(candidate):
                                network = candidate
                                best_network = list(network)
                                best_count = len(network)
                                layers = to_layers(network)
                                break
                if time.time() > deadline:
                    break
    
    # Phase 4: Final greedy removal pass
    if time.time() < deadline:
        improved = True
        while improved:
            improved = False
            for idx in range(len(network) - 1, -1, -1):
                if time.time() > deadline:
                    break
                success, candidate = _try_remove(network, idx)
                if success:
                    network = candidate
                    improved = True
                    best_network = list(network)
                    best_count = len(network)
            if time.time() > deadline:
                break
    
    # Phase 5: Genetic algorithm approach with layered representation
    if time.time() < deadline:
        # Use genetic algorithm to explore different layer configurations
        def fitness(net):
            """Fitness: lower is better (fewer comparators, then lower depth)."""
            if not _verify(net):
                return float('inf')
            return len(net) * 1000 + _compute_depth(net)
        
        def crossover(parent1, parent2):
            """Crossover two networks by taking layers from each."""
            layers1 = to_layers(parent1)
            layers2 = to_layers(parent2)
            child_layers = []
            for i in range(max(len(layers1), len(layers2))):
                if i < len(layers1) and i < len(layers2):
                    # Randomly choose from parent1 or parent2
                    if np.random.random() < 0.5:
                        child_layers.append(layers1[i])
                    else:
                        child_layers.append(layers2[i])
                elif i < len(layers1):
                    child_layers.append(layers1[i])
                else:
                    child_layers.append(layers2[i])
            return from_layers(child_layers)
        
        def mutation(net, rate=0.1):
            """Mutate a network by randomly removing or swapping comparators."""
            layers = to_layers(net)
            for l in range(len(layers)):
                if np.random.random() < rate:
                    # Randomly remove a comparator from this layer
                    if layers[l]:
                        idx = np.random.randint(len(layers[l]))
                        layers[l].pop(idx)
            # Try to add a random comparator if it fits
            if np.random.random() < rate:
                i, j = np.random.choice(N, 2, replace=False)
                if i > j:
                    i, j = j, i
                # Find a layer where it can fit
                for l in range(len(layers)):
                    can_fit = True
                    for c2 in layers[l]:
                        if i in c2 or j in c2:
                            can_fit = False
                            break
                    if can_fit:
                        layers[l].append((i, j))
                        break
            return from_layers(layers)
        
        # Initialize population
        population = [best_network]
        # Add some mutated versions
        for _ in range(5):
            mutated = mutation(best_network, rate=0.2)
            if _verify(mutated):
                population.append(mutated)
        
        # Run genetic algorithm
        ga_start = time.time()
        ga_deadline = min(deadline, ga_start + 20.0)
        
        for gen in range(50):
            if time.time() > ga_deadline:
                break
            
            # Evaluate fitness
            population = [ind for ind in population if _verify(ind)]
            if not population:
                population = [best_network]
            
            population.sort(key=fitness)
            
            # Keep top performers
            elites = population[:max(1, len(population)//2)]
            
            # Generate new generation
            new_pop = list(elites)
            while len(new_pop) < len(population):
                # Crossover
                p1 = population[np.random.randint(len(population))]
                p2 = population[np.random.randint(len(population))]
                child = crossover(p1, p2)
                # Mutation
                child = mutation(child, rate=0.15)
                if _verify(child):
                    new_pop.append(child)
            
            population = new_pop
        
        # Update best
        for ind in population:
            if len(ind) < best_count:
                if _verify(ind):
                    best_network = list(ind)
                    best_count = len(ind)
    
    # Phase 6: Final intensive local search
    if time.time() < deadline:
        # Try all possible single removals in multiple random orders
        rng = np.random.default_rng(123)
        for _round in range(10):
            if time.time() > deadline:
                break
            if len(network) <= best_count:
                break
            order = rng.permutation(len(network))
            found = False
            for pos in order:
                if time.time() > deadline:
                    break
                candidate = network[:pos] + network[pos+1:]
                if _verify(candidate):
                    network = candidate
                    best_network = list(network)
                    best_count = len(network)
                    found = True
                    break
            if not found:
                break
    
    return best_network
# EVOLVE-BLOCK-END
