import time
import random
from itertools import combinations

N = 16

# EVOLVE-BLOCK-START
def _apply_comparator(state, i, j):
    """Apply a compare-exchange to a bitmask state.
    state is a bitmask where bit k = 1 means wire k holds a 1.
    Comparator (i, j) with i < j: min goes to i, max to j.
    If bit i is 0 and bit j is 1, swap them.
    """
    bi = (state >> i) & 1
    bj = (state >> j) & 1
    if bi == 0 and bj == 1:
        # Swap: set bit i to 1, bit j to 0
        state |= (1 << i)
        state &= ~(1 << j)
    return state

def _apply_network(state, network):
    """Apply a list of comparators to a bitmask state."""
    for i, j in network:
        state = _apply_comparator(state, i, j)
    return state

def _is_sorted(state):
    """Check if a bitmask state is sorted (all 0s then all 1s)."""
    if state == 0:
        return True
    if state == (1 << N) - 1:
        return True
    # Find the position of the lowest set bit
    low = (state & -state).bit_length() - 1
    # All bits from low to N-1 must be 1, and bits 0..low-1 must be 0
    mask = ((1 << N) - 1) ^ ((1 << low) - 1)
    return state == mask

def _count_violations(network):
    """Count the number of inputs for which the network fails to sort.
    Returns the count of violations (0 means perfect sorting).
    """
    violations = 0
    for mask in range(1 << N):
        if not _is_sorted(_apply_network(mask, network)):
            violations += 1
    return violations

def _verify(network):
    """Verify that the network sorts all inputs. Returns True if valid."""
    return _count_violations(network) == 0

def _random_layer():
    """Generate a random layer of disjoint comparators.
    A layer is a random permutation of the 16 wires, paired as (0,1), (2,3), ..., (14,15).
    """
    wires = list(range(N))
    random.shuffle(wires)
    layer = []
    for k in range(N // 2):
        i, j = wires[2*k], wires[2*k+1]
        if i > j:
            i, j = j, i
        layer.append((i, j))
    return layer

def _random_network(num_layers):
    """Generate a random network with a fixed number of layers."""
    network = []
    for _ in range(num_layers):
        network.extend(_random_layer())
    return network

def _remove_redundant_comparators(network):
    """Remove redundant comparators from the network.
    A comparator is redundant if removing it does not increase the violation count.
    We use a greedy approach: try removing each comparator and keep it if it causes violations.
    """
    if not network:
        return network
    
    # First, check if the network is valid
    base_violations = _count_violations(network)
    if base_violations > 0:
        # If the network is invalid, we can't safely remove comparators without potentially making it worse
        # But we can still try to find a valid sub-network? No, that's hard.
        # For now, return the original if invalid, or try to repair?
        # Actually, if it's invalid, the GA should be trying to fix it.
        # Redundancy removal is only safe on valid networks.
        return network

    kept = []
    for idx, comp in enumerate(network):
        # Create a new network without this comparator
        candidate = network[:idx] + network[idx+1:]
        # Check if it's still valid
        if _count_violations(candidate) == 0:
            # It's redundant, so don't keep it
            # Update network to the candidate for subsequent checks?
            # No, we are checking against the original or the current kept?
            # Standard greedy: check against the current best kept so far?
            # Or check against the original?
            # If we check against original, we might keep a comparator that is redundant only because of another redundant one.
            # Better: check against the current 'kept' list.
            # But 'kept' is being built.
            # Let's just check if the candidate (original minus this one) is valid.
            # If yes, skip adding it.
            # If no, add it.
            pass
        else:
            kept.append(comp)
    
    return kept

def construct():
    """Construct a sorting network for 16 wires.
    Returns a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = 85.0  # Leave 5s margin for 90s limit
    
    # Known good networks for fallback
    # 1. Odd-Even Transposition (120 comparators)
    oet_network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # 2. Batcher's Odd-Even Merge Sort (119 comparators)
    # Implementing a correct OMS is tricky, but let's try a simple recursive one if time permits, 
    # or just use OET as the primary fallback since it's simple and known.
    
    # Strategy: Genetic Algorithm
    # Population of networks, each represented as a list of layers (or just a flat list of comparators).
    # Fitness: negative violation count. If violations == 0, fitness is -len(network) (prefer shorter).
    
    pop_size = 50
    num_layers_init = 10  # Start with 10 layers (80 comparators)
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        net = _random_network(num_layers_init)
        population.append(net)
    
    best_network = None
    best_fitness = float('inf')  # We minimize violations, then length
    
    def get_fitness(net):
        v = _count_violations(net)
        if v == 0:
            return len(net)
        else:
            return v * 1000 + len(net)  # Penalize violations heavily
    
    # Evaluate initial population
    for net in population:
        f = get_fitness(net)
        if f < best_fitness:
            best_fitness = f
            best_network = net[:]
    
    # Genetic Algorithm Loop
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Selection: Tournament selection
        def select():
            a, b = random.sample(population, 2)
            return get_fitness(a) < get_fitness(b) and a or b
        
        new_population = []
        
        # Elitism: Keep the best individual
        best_net = min(population, key=get_fitness)
        new_population.append(best_net[:])
        
        # Crossover and Mutation
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
                
            parent1 = select()
            parent2 = select()
            
            # Crossover: Slice crossover on the flat list of comparators
            # Ensure the child has the same number of layers/comparators?
            # Or allow variable length?
            # Let's stick to fixed length for simplicity, or truncate/pad.
            # Since layers are disjoint, a flat list of comparators doesn't guarantee disjointness in layers if we just slice.
            # But the comparator (i,j) is just a pair. Disjointness is a property of the *layer*, not the flat list.
            # The evaluator doesn't care about layers, just the sequence.
            # So we can treat the network as a sequence of comparators.
            
            if len(parent1) != len(parent2):
                # Pad the shorter one with random comparators or truncate?
                # Let's just truncate to the shorter length or pad to longer.
                # Truncating is safer for reducing size.
                max_len = min(len(parent1), len(parent2))
                p1 = parent1[:max_len]
                p2 = parent2[:max_len]
            else:
                p1 = parent1
                p2 = parent2
            
            cross_point = random.randint(1, len(p1) - 1) if len(p1) > 1 else 0
            child = p1[:cross_point] + p2[cross_point:]
            
            # Mutation:
            # 1. Change a comparator to a random one
            # 2. Remove a comparator
            # 3. Add a comparator
            # 4. Swap two comparators
            
            if random.random() < 0.1 and len(child) > 0:
                # Remove a random comparator
                idx = random.randint(0, len(child) - 1)
                child.pop(idx)
            elif random.random() < 0.1 and len(child) < 120:
                # Add a random comparator
                i, j = random.sample(range(N), 2)
                if i > j: i, j = j, i
                idx = random.randint(0, len(child))
                child.insert(idx, (i, j))
            elif random.random() < 0.3 and len(child) > 0:
                # Change a random comparator
                idx = random.randint(0, len(child) - 1)
                i, j = random.sample(range(N), 2)
                if i > j: i, j = j, i
                child[idx] = (i, j)
            elif random.random() < 0.2 and len(child) > 1:
                # Swap two comparators
                i, j = random.sample(range(len(child)), 2)
                child[i], child[j] = child[j], child[i]
            
            new_population.append(child)
        
        population = new_population
        
        # Evaluate new population and update best
        for net in population:
            f = get_fitness(net)
            if f < best_fitness:
                best_fitness = f
                best_network = net[:]
        
        # If we found a valid network, try to optimize its length
        if best_fitness < 1000:  # Means 0 violations
            # Try to remove redundant comparators
            if time.time() < deadline - 5:
                optimized = _remove_redundant_comparators(best_network)
                if len(optimized) < len(best_network):
                    # Verify the optimized one is still valid (should be, but just in case)
                    if _verify(optimized):
                        best_network = optimized
                        best_fitness = len(best_network)
            
            # If we have a very good network, we can stop early if we hit a known optimum or time is low
            if best_fitness <= 60:
                # We can't do better than 60 if that's the reference, but maybe 57?
                # Let's keep searching for a bit longer if time allows.
                pass

    # Final check: if best_network is valid, return it. Otherwise, return OET.
    if best_network is not None and _verify(best_network):
        return best_network
    else:
        return oet_network

# EVOLVE-BLOCK-END
