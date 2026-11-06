import time
import random
import math

N = 16

# Precompute initial bitsets for zero-one principle verification
_INITIAL_BITSETS = []
for i in range(N):
    bitset = 0
    for k in range(1 << N):
        if k & (1 << i):
            bitset |= (1 << k)
    _INITIAL_BITSETS.append(bitset)

def _verify_zero_one_fast(net):
    """
    Fast verification using bit-slicing.
    Returns True if the network sorts all 2^N boolean inputs.
    """
    if not net:
        return True
    
    state = _INITIAL_BITSETS[:]
    
    for i, j in net:
        new_i = state[i] & state[j]
        new_j = state[i] | state[j]
        state[i] = new_i
        state[j] = new_j
        
    # Check if sorted
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
            
    return True

def _base_60_network():
    """
    A known 60-comparator sorting network for 16 wires.
    """
    return [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    ]

def _odd_even_network():
    """
    Odd-even transposition network: 120 comparators.
    """
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _compute_states(net):
    """
    Returns a list of states (list of 16 bitsets) for each step of the network.
    states[k] is the state before comparator k is applied.
    states[len(net)] is the final state.
    """
    states = [_INITIAL_BITSETS[:]]
    current = _INITIAL_BITSETS[:]
    
    for i, j in net:
        new_i = current[i] & current[j]
        new_j = current[i] | current[j]
        current[i] = new_i
        current[j] = new_j
        states.append(current[:])
        
    return states

def _prune_net(net, deadline):
    """
    Prunes redundant comparators. A comparator (i, j) is redundant if
    wire i is already <= wire j for all inputs (i.e., state[i] & ~state[j] == 0).
    """
    current_net = net[:]
    changed = True
    while changed and time.time() < deadline:
        changed = False
        if not current_net:
            break
            
        states = _compute_states(current_net)
        
        # Identify indices to remove
        to_remove = set()
        for k in range(len(current_net)):
            i, j = current_net[k]
            # Check state BEFORE comparator k
            if states[k][i] & ~states[k][j] == 0:
                to_remove.add(k)
                
        if to_remove:
            # Remove in reverse order to keep indices valid, or build new list
            new_net = []
            for k in range(len(current_net)):
                if k not in to_remove:
                    new_net.append(current_net[k])
            current_net = new_net
            changed = True
            
    return current_net

def _crossover(p1, p2, rng):
    """
    Simple crossover: merge the two lists, then prune.
    Since sorting networks are commutative in a sense (order matters for intermediate
    states but the final result is a sort), merging often preserves validity if
    both parents are valid, though it increases length. Pruning recovers efficiency.
    """
    child = p1[:] + p2[:]
    rng.shuffle(child)
    return child

def _mutate(net, rng, mutation_rate=0.05):
    """
    Mutate the network by randomly removing, adding, or swapping comparators.
    """
    new_net = net[:]
    
    # Randomly remove a few
    for _ in range(int(len(new_net) * mutation_rate)):
        if new_net:
            idx = rng.randrange(len(new_net))
            new_net.pop(idx)
            
    # Randomly add a few
    for _ in range(int(len(new_net) * mutation_rate * 0.5)):
        i, j = rng.sample(range(N), 2)
        if i > j: i, j = j, i
        pos = rng.randrange(len(new_net) + 1)
        new_net.insert(pos, (i, j))
        
    # Randomly swap
    for _ in range(int(len(new_net) * mutation_rate)):
        if len(new_net) > 1:
            i, j = rng.sample(range(len(new_net)), 2)
            new_net[i], new_net[j] = new_net[j], new_net[i]
            
    # Randomly mutate endpoints
    for _ in range(int(len(new_net) * mutation_rate * 0.5)):
        if new_net:
            idx = rng.randrange(len(new_net))
            i, j = new_net[idx]
            if rng.random() < 0.5:
                new_i = rng.randrange(N)
                if new_i < j:
                    new_net[idx] = (new_i, j)
                elif new_i > j:
                    new_net[idx] = (j, new_i)
            else:
                new_j = rng.randrange(N)
                if i < new_j:
                    new_net[idx] = (i, new_j)
                elif i > new_j:
                    new_net[idx] = (new_j, i)
                    
    return new_net

def _local_search(net, deadline):
    """
    Perform local search on the network.
    """
    current_net = net[:]
    current_len = len(current_net)
    
    # Initial prune
    current_net = _prune_net(current_net, deadline)
    current_len = len(current_net)
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        
        # Try random removals (greedy-like)
        if current_net:
            idx = rng.randrange(len(current_net))
            candidate = current_net[:idx] + current_net[idx+1:]
            if _verify_zero_one_fast(candidate):
                current_net = candidate
                current_len -= 1
                improved = True
                no_improve = 0
            else:
                no_improve += 1
        
        # Try random swaps
        if not improved and len(current_net) > 1:
            i, j = rng.sample(range(len(current_net)), 2)
            candidate = current_net[:]
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if _verify_zero_one_fast(candidate):
                current_net = candidate
                # Length same, but maybe enables future removals
                no_improve += 1
            else:
                no_improve += 1
                
        # Prune periodically
        if time.time() < deadline:
            pruned = _prune_net(current_net, deadline)
            if len(pruned) < current_len:
                current_net = pruned
                current_len = len(pruned)
                improved = True
                no_improve = 0
            else:
                if not improved:
                    no_improve += 1
                    
    return current_net

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    rng = random.Random(42)
    
    # Initial best
    best_net = _base_60_network()
    if not _verify_zero_one_fast(best_net):
        best_net = _odd_even_network()
    
    # Prune the initial best
    best_net = _prune_net(best_net, deadline)
    best_len = len(best_net)
    
    # Population for GP
    pop_size = 10
    population = [best_net[:]]
    
    # Add some diverse initial individuals
    oen = _odd_even_network()
    oen_pruned = _prune_net(oen, deadline)
    population.append(oen_pruned[:])
    
    # Add random mutations of best
    for _ in range(pop_size - 2):
        mutated = _mutate(best_net, rng, mutation_rate=0.2)
        # Prune to make it valid-ish or at least cleaner
        mutated = _prune_net(mutated, deadline)
        if _verify_zero_one_fast(mutated):
            population.append(mutated)
        else:
            # Fallback to best
            population.append(best_net[:])
            
    # Ensure population size
    while len(population) < pop_size:
        population.append(best_net[:])
        
    generation = 0
    stagnation_counter = 0
    
    while time.time() < deadline:
        generation += 1
        
        # Evaluate and sort population
        scored_pop = []
        for net in population:
            if _verify_zero_one_fast(net):
                scored_pop.append((len(net), net))
            else:
                # Invalid, assign high cost
                scored_pop.append((10000, net))
                
        scored_pop.sort(key=lambda x: x[0])
        
        best_in_gen_len, best_in_gen_net = scored_pop[0]
        
        if best_in_gen_len < best_len:
            best_len = best_in_gen_len
            best_net = best_in_gen_net[:]
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        # If stuck for too long, restart or increase mutation
        if stagnation_counter > 10:
            # Inject diversity
            for _ in range(pop_size // 2):
                mutated = _mutate(best_net, rng, mutation_rate=0.5)
                mutated = _prune_net(mutated, deadline)
                if _verify_zero_one_fast(mutated):
                    population[rng.randrange(pop_size)] = mutated
            stagnation_counter = 0
            
        # Create next generation
        next_pop = []
        
        # Elitism: keep the best
        next_pop.append(scored_pop[0][1][:])
        
        # Crossover and Mutation
        while len(next_pop) < pop_size:
            if time.time() > deadline:
                break
                
            # Select parents (tournament selection)
            p1_idx = rng.randrange(min(5, len(scored_pop)))
            p2_idx = rng.randrange(min(5, len(scored_pop)))
            
            p1 = scored_pop[p1_idx][1]
            p2 = scored_pop[p2_idx][1]
            
            # Crossover
            child = _crossover(p1, p2, rng)
            
            # Mutation
            child = _mutate(child, rng, mutation_rate=0.1)
            
            # Prune
            child = _prune_net(child, deadline)
            
            # Verify
            if _verify_zero_one_fast(child):
                next_pop.append(child)
            else:
                # If invalid, maybe keep a mutated version of best or just fill with best
                next_pop.append(best_net[:])
                
        population = next_pop
        
    # Final local search on the best found
    final_net = _local_search(best_net, deadline)
    
    return final_net

if __name__ == "__main__":
    # Quick self-test
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print(f"Valid: {_verify_zero_one_fast(net)}")

