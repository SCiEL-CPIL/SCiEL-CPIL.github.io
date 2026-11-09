import time
import random
import sys

N = 17
TOTAL_MASKS = 1 << N
DEADLINE_FRACTION = 0.92

_INITIAL_WIRES = None
_EXPECTED_WIRES = None

def _init_wires():
    global _INITIAL_WIRES, _EXPECTED_WIRES
    if _INITIAL_WIRES is not None:
        return
    
    _INITIAL_WIRES = [0] * N
    _EXPECTED_WIRES = [0] * N
    
    # Initial wires: bit k of wire i is 1 if (k >> i) & 1
    for i in range(N):
        w = 0
        for k in range(TOTAL_MASKS):
            if (k >> i) & 1:
                w |= (1 << k)
        _INITIAL_WIRES[i] = w
        
    # Expected wires: bit k of wire i is 1 if popcount(k) >= N-i
    for i in range(N):
        w = 0
        threshold = N - i
        for k in range(TOTAL_MASKS):
            if k.bit_count() >= threshold:
                w |= (1 << k)
        _EXPECTED_WIRES[i] = w

_init_wires()

def _verify_network(net):
    """
    Fast verification using pre-computed big integers.
    Returns True if the network sorts all 17-input sequences.
    """
    wires = _INITIAL_WIRES[:]
    
    for i, j in net:
        wi = wires[i]
        wj = wires[j]
        # min to i, max to j
        # Swap bits where wi=1 and wj=0
        S = wi & ~wj
        wires[i] = wi ^ S
        wires[j] = wj ^ S
        
    # Check against expected
    for i in range(N):
        if wires[i] != _EXPECTED_WIRES[i]:
            return False
    return True

def _build_batcher_odd_merge(n):
    """
    Batcher Odd-Merge sorting network.
    """
    if n <= 1:
        return []
    
    n1 = (n + 1) // 2
    n2 = n // 2
    
    net = []
    net.extend(_build_batcher_odd_merge(n1))
    
    if n2 > 1:
        sub_net = _build_batcher_odd_merge(n2)
        for i, j in sub_net:
            net.append((i + n1, j + n1))
            
    for i in range(n1):
        net.append((i, i + n2))
        
    if n1 > 1:
        sub_even = _build_batcher_odd_merge(n1)
        for i, j in sub_even:
            net.append((2*i, 2*j))
            
    if n2 > 1:
        sub_odd = _build_batcher_odd_merge(n2)
        for i, j in sub_odd:
            net.append((2*i+1, 2*j+1))
            
    return net

def _local_search_greedy(net, deadline):
    """
    Aggressively remove redundant comparators.
    Iterates through the network and removes any comparator that does not break validity.
    """
    current_net = net[:]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Randomize order to avoid getting stuck in a specific local minimum
        indices = list(range(len(current_net)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                return current_net
            # Try removing comparator at idx
            new_net = current_net[:idx] + current_net[idx+1:]
            if _verify_network(new_net):
                current_net = new_net
                improved = True
                break # Restart loop with new size
                
    return current_net

def _perturb_network(net, num_perturbations):
    """
    Apply random structural changes to escape local minima.
    1. Swap two comparators.
    2. Insert a random comparator (temporarily increasing size).
    3. Move a comparator to a different position.
    """
    if not net:
        return net
    
    new_net = net[:]
    size = len(new_net)
    
    for _ in range(num_perturbations):
        move_type = random.random()
        
        if move_type < 0.4 and size >= 2:
            # Swap
            i, j = random.sample(range(size), 2)
            new_net[i], new_net[j] = new_net[j], new_net[i]
            
        elif move_type < 0.7 and size > 0:
            # Move
            i = random.randint(0, size - 1)
            j = random.randint(0, size - 1)
            comp = new_net.pop(i)
            new_net.insert(j, comp)
            
        else:
            # Insert random comparator
            # Choose random wires i < j
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, size)
            new_net.insert(pos, (i, j))
            size += 1
            
    return new_net

def construct():
    start_time = time.time()
    deadline = start_time + 60 * DEADLINE_FRACTION
    
    # Initialize population
    # Start with Batcher Odd-Merge
    base_net = _build_batcher_odd_merge(N)
    
    # Initial greedy reduction
    base_net = _local_search_greedy(base_net, deadline)
    
    population = [base_net]
    best_net = base_net[:]
    best_size = len(base_net)
    
    # If we have time, generate diverse initial population members
    # by perturbing the base and reducing
    max_pop_size = 5
    while len(population) < max_pop_size and time.time() < deadline:
        perturbed = _perturb_network(base_net, random.randint(5, 15))
        # Reduce the perturbed version
        reduced = _local_search_greedy(perturbed, deadline)
        
        # Only keep if it's valid (it should be, as we start from valid and only add/swap/move, 
        # but _local_search_greedy assumes input is valid? No, it just checks validity of removals.
        # If the perturbed network is invalid, _local_search_greedy might return it as-is if no removals work?
        # Actually, _local_search_greedy returns the net. We must ensure it's valid.
        if _verify_network(reduced):
            population.append(reduced)
        else:
            # If invalid, try to repair by adding comparators? Or just discard.
            # Discarding is safer for correctness.
            pass

    # Main loop: Iterated Local Search
    # We cycle through the population, perturbing each member and locally optimizing.
    # If a new member is better than the current best, update best.
    # If a new member is better than the member it replaced (or if we just keep the top K), update population.
    
    # Simple strategy: Keep the best K networks found so far.
    # Here K=5.
    
    while time.time() < deadline:
        # Pick a random member from the population to evolve
        parent_idx = random.randint(0, len(population) - 1)
        parent = population[parent_idx]
        
        # Perturb
        # Increase perturbation strength if we haven't improved in a while?
        # For now, fixed moderate perturbation.
        num_perts = random.randint(3, 10)
        child = _perturb_network(parent, num_perts)
        
        # Local search (greedy removal)
        child = _local_search_greedy(child, deadline)
        
        # Check validity
        if _verify_network(child):
            child_size = len(child)
            
            # Update best
            if child_size < best_size:
                best_net = child[:]
                best_size = child_size
                # Replace the worst member in population with the new best?
                # Or just add to population?
                # Let's keep the population size constant and replace the worst one.
                # Find the worst (largest) network in population
                worst_idx = 0
                worst_size = len(population[0])
                for i in range(1, len(population)):
                    if len(population[i]) > worst_size:
                        worst_size = len(population[i])
                        worst_idx = i
                population[worst_idx] = child[:]
                
            else:
                # If not better than best, but maybe better than the parent or some other member?
                # Replace the worst member if this child is better than the worst.
                worst_idx = 0
                worst_size = len(population[0])
                for i in range(1, len(population)):
                    if len(population[i]) > worst_size:
                        worst_size = len(population[i])
                        worst_idx = i
                
                if child_size < worst_size:
                    population[worst_idx] = child[:]
                elif child_size == worst_size and random.random() < 0.5:
                    # Diversification: replace with equal size with some probability
                    population[worst_idx] = child[:]
                    
    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print("Valid:", _verify_network(net))
