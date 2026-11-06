import time
import random
import numpy as np

N = 18
TOTAL_INPUTS = 1 << N
DEADLINE = 55.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # --- Bit-Parallel Setup ---
    NUM_WORDS = TOTAL_INPUTS // 64
    indices = np.arange(TOTAL_INPUTS, dtype=np.uint32)
    weights = 1 << np.arange(64, dtype=np.uint64)
    
    initial_wires = [None] * N
    for i in range(N):
        bits_i = (indices >> i) & 1
        bits_i_64 = bits_i.astype(np.uint64)
        packed = (bits_i_64.reshape(NUM_WORDS, 64) * weights).sum(axis=1)
        initial_wires[i] = packed

    def verify_network(network):
        wires = [w.copy() for w in initial_wires]
        for i, j in network:
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj
        for i in range(N - 1):
            if np.any(wires[i] & ~wires[i + 1]):
                return False
        return True

    def find_violation(network):
        wires = [w.copy() for w in initial_wires]
        for i, j in network:
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj
        for i in range(N - 1):
            if np.any(wires[i] & ~wires[i + 1]):
                return (i, i + 1)
        return None

    def greedy_remove(net, time_budget=1.0):
        current = net[:]
        start_greedy = time.time()
        removed = True
        while removed and (time.time() - start_greedy) < time_budget:
            removed = False
            n = len(current)
            if n == 0: break
            # Check all positions in random order? Or just sample?
            # For small N, checking all is fast.
            # Let's try checking all positions for better reduction.
            positions = list(range(n))
            random.shuffle(positions)
            for idx in positions:
                if idx >= len(current): continue
                candidate = current[:]
                candidate.pop(idx)
                if verify_network(candidate):
                    current = candidate
                    removed = True
                    break
        return current

    # --- Initial Network: Odd-Even Transposition ---
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    
    best_network = network[:]
    best_size = len(network)
    
    # Initial Reduction
    network = greedy_remove(network, time_budget=5.0)
    if len(network) < best_size:
        best_network = network[:]
        best_size = len(network)
        
    # --- Local Search with Repair ---
    current = best_network[:]
    random.seed(42)
    
    # Simulated Annealing parameters
    T = 1.0
    cooling = 0.995
    
    while (time.time() - start_time) < DEADLINE - 5:
        n_comp = len(current)
        if n_comp <= 70: break
        
        r = random.random()
        candidate = current[:]
        valid_move = True
        
        if r < 0.4:
            # Move
            if n_comp > 1:
                i = random.randint(0, n_comp - 1)
                j = random.randint(0, n_comp - 1)
                comp = candidate.pop(i)
                if i < j: j -= 1
                j = max(0, min(j, len(candidate)))
                candidate.insert(j, comp)
            else:
                valid_move = False
        elif r < 0.7:
            # Swap
            if n_comp > 1:
                i, j = random.sample(range(n_comp), 2)
                candidate[i], candidate[j] = candidate[j], candidate[i]
            else:
                valid_move = False
        else:
            # Replace
            i = random.randint(0, n_comp - 1)
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u == v:
                valid_move = False
            else:
                if u > v: u, v = v, u
                candidate[i] = (u, v)
                
        if not valid_move:
            continue
            
        if verify_network(candidate):
            # Valid
            # Try to reduce
            candidate = greedy_remove(candidate, time_budget=0.2)
            if len(candidate) < best_size:
                best_size = len(candidate)
                best_network = candidate[:]
                current = candidate
            elif len(candidate) <= len(current):
                current = candidate
        else:
            # Invalid: Try Repair
            inv_pair = find_violation(candidate)
            if inv_pair:
                # Try inserting inv_pair at random positions
                # Focus on positions near where the change happened?
                # For now, random positions.
                positions = list(range(len(candidate) + 1))
                random.shuffle(positions)
                repaired = False
                for pos in positions[:5]: # Limit repair attempts
                    temp = candidate[:]
                    temp.insert(pos, inv_pair)
                    if verify_network(temp):
                        # Repair successful
                        # Reduce
                        temp = greedy_remove(temp, time_budget=0.2)
                        if len(temp) < best_size:
                            best_size = len(temp)
                            best_network = temp[:]
                        # Accept if not too large
                        if len(temp) <= len(current) + 2:
                            current = temp
                        repaired = True
                        break
                
    # Final Polish
    final_net = best_network[:]
    final_net = greedy_remove(final_net, time_budget=5.0)
    if len(final_net) < best_size:
        best_network = final_net[:]
        best_size = len(final_net)
        
    if not verify_network(best_network):
        # Fallback
        fallback = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                fallback.append((i, i + 1))
        return fallback
        
    return best_network
# EVOLVE-BLOCK-END
