"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 18

# EVOLVE-BLOCK-START
import math

def _apply_network(network, inputs):
    """Apply a compare-exchange network to a list of inputs."""
    wires = list(inputs)
    for i, j in network:
        if wires[i] > wires[j]:
            wires[i], wires[j] = wires[j], wires[i]
    return wires

def _is_correct(network):
    """Check if the network sorts all 2^N boolean inputs using 0-1 principle."""
    for mask in range(1 << N):
        wires = [(mask >> i) & 1 for i in range(N)]
        for i, j in network:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        # Check if result is sorted
        for i in range(N - 1):
            if wires[i] > wires[i + 1]:
                return False
    return True

def _bitonic_sort_network(n):
    """Generate Batcher's bitonic sort network for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    comps = []
    # Recursively sort first half ascending, second half descending
    left = _bitonic_sort_network(n // 2)
    right = _bitonic_sort_network(n // 2)
    # Offset right half comparators by n//2
    right_offset = [(i + n // 2, j + n // 2) for i, j in right]
    comps.extend(left)
    comps.extend(right_offset)
    
    # Bitonic merge
    comps.extend(_bitonic_merge(n))
    return comps

def _bitonic_merge(n):
    """Generate bitonic merge network for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    comps = []
    logn = int(math.log2(n))
    for k in range(logn, 0, -1):
        step = 1 << k
        for i in range(0, n - step):
            comps.append((i, i + step))
    return comps

def _odd_even_merge(n):
    """Generate odd-even merge network for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    comps = []
    # Recursively merge odd and even subsequences
    odd = _odd_even_merge(n // 2)
    even = _odd_even_merge(n // 2)
    # Offset even subsequence by 1
    even_offset = [(i + 1, j + 1) for i, j in even]
    comps.extend(odd)
    comps.extend(even_offset)
    
    # Cross-comparisons
    for i in range(1, n - 1, 2):
        comps.append((i, i + 1))
    return comps

def _generate_valid_networks():
    """Generate a set of valid sorting networks using constructive heuristics."""
    networks = []
    
    # Odd-even transposition network (seed)
    seed = [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
    networks.append(seed)
    
    # Bitonic sort for 16 wires + handle 2 extra wires
    bitonic_16 = _bitonic_sort_network(16)
    # Add comparators to incorporate wires 16 and 17
    extended = list(bitonic_16)
    # Compare wire 16 with 0-15, then wire 17 with 0-16
    for i in range(16):
        extended.append((i, 16))
    for i in range(16):
        extended.append((i, 17))
    extended.append((16, 17))
    # Re-sort to ensure correctness (add more comparators)
    for i in range(17):
        extended.append((i, 17))
    networks.append(extended)
    
    # Odd-even merge for 16 wires + handle 2 extra wires
    oem_16 = _odd_even_merge(16)
    extended_oem = list(oem_16)
    for i in range(16):
        extended_oem.append((i, 16))
    for i in range(16):
        extended_oem.append((i, 17))
    extended_oem.append((16, 17))
    for i in range(17):
        extended_oem.append((i, 17))
    networks.append(extended_oem)
    
    # Simple insertion sort network
    insertion = []
    for i in range(1, N):
        for j in range(i, 0, -1):
            insertion.append((j - 1, j))
    networks.append(insertion)
    
    return networks

def _topology_preserving_crossover(parent1, parent2):
    """Crossover that preserves topological validity by taking prefix/suffix at safe points."""
    if not parent1 or not parent2:
        return list(parent1) if parent1 else list(parent2)
    
    # Find safe cut points where no wire is "in flight"
    # A safe cut is one where all wires touched before the cut are also touched after
    len1, len2 = len(parent1), len(parent2)
    
    # Try multiple cut points and pick one that seems valid
    best_child = None
    best_score = float('inf')
    
    for cut in range(0, min(len1, len2) + 1, max(1, min(len1, len2) // 10)):
        child = list(parent1[:cut]) + list(parent2[cut:])
        # Quick validity check: ensure no comparator has i >= j
        valid = all(i < j for i, j in child)
        if valid:
            score = len(child)
            if score < best_score:
                best_score = score
                best_child = child
    
    if best_child is None:
        # Fallback: simple prefix/suffix
        cut = random.randint(0, min(len1, len2))
        best_child = list(parent1[:cut]) + list(parent2[cut:])
    
    return best_child

def _topology_preserving_mutation(network, rate=0.1):
    """Mutate while trying to preserve sorting validity."""
    network = list(network)
    
    # Mutation 1: Insert a redundant comparator (safe)
    if random.random() < rate and len(network) < 4000:
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        pos = random.randint(0, len(network))
        network.insert(pos, (i, j))
    
    # Mutation 2: Remove a comparator (risky, but try it)
    if random.random() < rate * 0.5 and len(network) > 10:
        pos = random.randint(0, len(network) - 1)
        network.pop(pos)
    
    # Mutation 3: Swap adjacent comparators that don't conflict
    if random.random() < rate and len(network) >= 2:
        i = random.randint(0, len(network) - 2)
        j = i + 1
        # Only swap if they don't share wires or if swapping is safe
        a1, a2 = network[i]
        b1, b2 = network[j]
        # Check if swapping maintains validity
        if not (a1 == b1 or a1 == b2 or a2 == b1 or a2 == b2):
            network[i], network[j] = network[j], network[i]
    
    # Mutation 4: Move a comparator to a different position
    if random.random() < rate * 0.5 and len(network) >= 2:
        i = random.randint(0, len(network) - 1)
        j = random.randint(0, len(network) - 1)
        if i != j:
            comp = network.pop(i)
            network.insert(j, comp)
    
    return network

def construct():
    """Construct a sorting network using an improved genetic algorithm."""
    start_time = time.time()
    timeout = 55
    
    # Generate initial population of valid networks
    valid_networks = _generate_valid_networks()
    population = [list(net) for net in valid_networks]
    
    # Verify and track best
    best_network = None
    best_size = float('inf')
    
    for net in population:
        if _is_correct(net):
            if len(net) < best_size:
                best_size = len(net)
                best_network = list(net)
    
    # If no valid network found, use seed
    if best_network is None:
        seed = [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
        best_network = seed
        best_size = len(seed)
    
    pop_size = 24
    generation = 0
    
    while time.time() - start_time < timeout:
        generation += 1
        
        # Evaluate and sort population
        scored = []
        for net in population:
            if len(net) < best_size:
                if _is_correct(net):
                    scored.append((len(net), net))
                    best_size = len(net)
                    best_network = list(net)
                else:
                    scored.append((len(net) + 500, net))
            else:
                scored.append((len(net), net))
        
        scored.sort(key=lambda x: x[0])
        population = [net for _, net in scored]
        
        # Create next generation
        next_gen = []
        
        # Elitism: keep top 4
        for net in population[:4]:
            next_gen.append(list(net))
        
        # Generate offspring
        while len(next_gen) < pop_size:
            # Tournament selection
            p1 = random.choice(population[:max(1, len(population) // 2)])
            p2 = random.choice(population[:max(1, len(population) // 2)])
            
            # Crossover
            child = _topology_preserving_crossover(p1, p2)
            
            # Mutation with adaptive rate
            mutation_rate = 0.1 + 0.1 * (generation / max(1, generation + 10))
            child = _topology_preserving_mutation(child, rate=mutation_rate)
            
            next_gen.append(child)
        
        population = next_gen
        
        # Early exit
        if best_size <= 72:
            break
    
    return best_network
# EVOLVE-BLOCK-END
