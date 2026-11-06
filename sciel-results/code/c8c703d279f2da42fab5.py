import time
import random

N = 17
NUM_INPUTS = 1 << N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    # 1. Generate the seed network: Odd-Even Transposition
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best_network = seed_network[:]
    
    # 2. Verification Function using bit-vector simulation
    # state[j] is a bitmask of NUM_INPUTS bits, where bit k is 1 if input k has a 1 on wire j
    initial_state = [0] * N
    for j in range(N):
        mask = 0
        for k in range(NUM_INPUTS):
            if (k >> j) & 1:
                mask |= (1 << k)
        initial_state[j] = mask

    def is_valid(network):
        state = initial_state[:]
        for i, j in network:
            new_i = state[i] & state[j]
            new_j = state[i] | state[j]
            state[i] = new_i
            state[j] = new_j
        
        for k in range(N - 1):
            if (state[k] & ~state[k+1]) != 0:
                return False
        return True

    if not is_valid(best_network):
        return best_network

    random.seed(42)
    
    # 3. Local Search
    current_network = best_network[:]
    
    # Precompute all possible comparators
    all_comparators = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to remove a comparator
        # Shuffle indices to avoid bias
        indices = list(range(len(current_network)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            candidate = current_network[:idx] + current_network[idx+1:]
            
            if is_valid(candidate):
                current_network = candidate
                improved = True
                if len(current_network) < len(best_network):
                    best_network = current_network[:]
                break # Restart scan with new size
        
        if not improved:
            # Phase 2: Mutate
            # Strategy:
            # 1. Try to add a comparator and then see if we can remove two (net -1)
            # 2. Or just do a local move (swap/replace)
            
            # Attempt "Add then Remove" (2-opt style)
            # This is more expensive but can escape local minima better
            if random.random() < 0.7:
                # Pick a random comparator to add
                new_comp = random.choice(all_comparators)
                pos = random.randint(0, len(current_network))
                
                candidate = current_network[:]
                candidate.insert(pos, new_comp)
                
                # Now try to remove 2 comparators from this candidate to get back to size-1
                # We only need to check if removing 2 specific ones works, but that's O(N^2) checks * verification cost
                # Instead, we can just accept the add if it leads to a valid state, and rely on the next loop's removal phase.
                # But that just increases size.
                # Better: Try to find a pair of indices (a, b) in candidate such that removing both yields a valid network.
                # This is expensive. Let's limit the search.
                
                # Alternative: Just do a simple mutation that keeps size constant or +1, and let the removal phase handle it.
                # Actually, a better strategy for "Add-Remove" is:
                # 1. Add a comparator.
                # 2. Run a limited removal scan on the new network.
                # 3. If we can remove 2, we have a net improvement of -1.
                
                # Let's try a simpler mutation first: Replace a comparator
                i = random.randint(0, len(current_network) - 1)
                new_comp = random.choice(all_comparators)
                candidate = current_network[:]
                candidate[i] = new_comp
                
                if is_valid(candidate):
                    current_network = candidate
            else:
                # Swap two comparators
                if len(current_network) > 1:
                    i = random.randint(0, len(current_network) - 1)
                    j = random.randint(0, len(current_network) - 1)
                    if i != j:
                        candidate = current_network[:]
                        candidate[i], candidate[j] = candidate[j], candidate[i]
                        if is_valid(candidate):
                            current_network = candidate

    return best_network
# EVOLVE-BLOCK-END
