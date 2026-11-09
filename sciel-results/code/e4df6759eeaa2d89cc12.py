import numpy as np
import time

N = 18
NUM_INPUTS = 1 << N
BLOCK_SIZE = 64
NUM_BLOCKS = NUM_INPUTS // BLOCK_SIZE

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Pre-compute target states for all 2^18 inputs
    # Target for input k: wire i is 1 if i >= N - popcount(k)
    indices = np.arange(NUM_INPUTS, dtype=np.uint32).reshape(NUM_BLOCKS, BLOCK_SIZE)
    
    # Compute bits and popcounts
    bits = (indices[:, :, None] >> np.arange(N)[None, None, :]) & 1
    popcounts = bits.sum(axis=2)
    
    # Helper to pack 2D bool array (B, 64) into 1D uint64 array (B,)
    def pack_bits_to_u64(arr_2d):
        flat = arr_2d.flatten()
        packed = np.packbits(flat, axis=0, bitorder='little')
        packed = packed.reshape(NUM_BLOCKS, 8)
        return packed.view(np.uint64).flatten()
    
    # Compute targets for each wire
    targets = []
    for i in range(N):
        mask = (popcounts >= (N - i))
        targets.append(pack_bits_to_u64(mask))
    
    # Initialize wires (identity permutation)
    wires = []
    for i in range(N):
        wires.append(pack_bits_to_u64(bits[:, :, i]))
    
    # Function to apply comparator
    def apply_comp(wires, i, j):
        new_wires = wires.copy()
        a = new_wires[i]
        b = new_wires[j]
        new_a = a & b
        new_b = a | b
        new_wires[i] = new_a
        new_wires[j] = new_b
        return new_wires
    
    # Function to calculate error (Hamming distance to target)
    def calc_error(wires):
        err = 0
        for i in range(N):
            diff = wires[i] ^ targets[i]
            try:
                err += np.bitwise_count(diff).sum()
            except AttributeError:
                u8 = diff.view(np.uint8).reshape(NUM_BLOCKS, 8)
                unpacked = np.unpackbits(u8, axis=1, count=8, bitorder='little')
                err += unpacked.sum()
        return err
    
    # Greedy search to find a good starting point
    current_wires = wires
    current_error = calc_error(current_wires)
    network = []
    
    # Greedy phase
    while time.time() < deadline - 10.0:
        best_error = current_error
        best_comp = None
        best_new_wires = None
        
        for i in range(N):
            for j in range(i + 1, N):
                new_wires = apply_comp(current_wires, i, j)
                new_error = calc_error(new_wires)
                if new_error < best_error:
                    best_error = new_error
                    best_comp = (i, j)
                    best_new_wires = new_wires
                    
        if best_comp is None:
            break
            
        current_wires = best_new_wires
        current_error = best_error
        network.append(best_comp)
        
        if current_error == 0:
            break
    
    # If we already found a perfect solution, return it
    if current_error == 0:
        return network
    
    # Genetic algorithm phase
    # Population of networks
    pop_size = 50
    population = [network.copy() for _ in range(pop_size)]
    
    # Add some random networks to population
    for k in range(pop_size // 2):
        random_net = []
        for _ in range(20 + np.random.randint(0, 20)):
            i = np.random.randint(0, N)
            j = np.random.randint(i + 1, N)
            random_net.append((i, j))
        population[k] = random_net
    
    # Fitness function: negative of error
    def fitness(net):
        if not net:
            return 0
        w = wires.copy()
        for i, j in net:
            w = apply_comp(w, i, j)
        return -calc_error(w)
    
    best_net = network.copy()
    best_fit = fitness(best_net)
    
    # Evolution loop
    while time.time() < deadline:
        # Evaluate fitness
        fits = [fitness(net) for net in population]
        
        # Find best
        max_idx = np.argmax(fits)
        if fits[max_idx] > best_fit:
            best_fit = fits[max_idx]
            best_net = population[max_idx].copy()
        
        # Selection: tournament selection
        new_population = []
        for _ in range(pop_size):
            # Tournament of 3
            idxs = np.random.choice(pop_size, 3, replace=False)
            winner = idxs[np.argmax([fits[i] for i in idxs])]
            new_population.append(population[winner].copy())
        
        # Crossover and mutation
        for k in range(pop_size):
            if np.random.rand() < 0.7:  # Crossover probability
                if k + 1 < pop_size:
                    other = population[np.random.randint(0, pop_size)]
                    # Uniform crossover
                    crossover_point = np.random.randint(0, max(len(new_population[k]), len(other), 1))
                    child = new_population[k][:crossover_point] + other[crossover_point:]
                    new_population[k] = child
            
            # Mutation
            if np.random.rand() < 0.5:
                # Insert random comparator
                pos = np.random.randint(0, len(new_population[k]) + 1)
                i = np.random.randint(0, N)
                j = np.random.randint(i + 1, N)
                new_population[k].insert(pos, (i, j))
            
            if np.random.rand() < 0.3:
                # Delete random comparator
                if new_population[k]:
                    pos = np.random.randint(0, len(new_population[k]))
                    new_population[k].pop(pos)
            
            if np.random.rand() < 0.2:
                # Swap two comparators
                if len(new_population[k]) >= 2:
                    i1 = np.random.randint(0, len(new_population[k]))
                    i2 = np.random.randint(0, len(new_population[k]))
                    new_population[k][i1], new_population[k][i2] = new_population[k][i2], new_population[k][i1]
        
        population = new_population
        
        # Elitism: keep best
        fits = [fitness(net) for net in population]
        max_idx = np.argmax(fits)
        if fits[max_idx] > best_fit:
            best_fit = fits[max_idx]
            best_net = population[max_idx].copy()
        population[0] = best_net.copy()
    
    # Return best network found
    if best_fit == 0:  # Perfect
        return best_net
    elif len(best_net) < len(network):
        return best_net
    else:
        return network
