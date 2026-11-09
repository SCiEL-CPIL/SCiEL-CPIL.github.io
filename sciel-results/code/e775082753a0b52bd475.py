def construct():
    start_time = time.time()
    time_limit = 55.0
    
    def odd_even_transposition():
        net = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                net.append((i, i + 1))
        return net
        
    network = odd_even_transposition()
    best_network = network
    best_size = len(network)
    
    if not _verify_network(network):
        raise ValueError("Seed network is invalid")
        
    # 1. Greedy Removal
    # Try to remove comparators. We do this in a loop.
    # To make it faster, we can check removal of multiple comparators at once? No, 1 by 1 is safer.
    # But we can be smarter about which one to try.
    
    greedy_time_limit = time_limit * 0.3
    while time.time() - start_time < greedy_time_limit:
        removed_any = False
        # Try removing from the end first, as they are often redundant
        indices = list(range(len(network) - 1, -1, -1))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > greedy_time_limit:
                break
            if idx >= len(network):
                continue
            new_network = network[:idx] + network[idx+1:]
            if _verify_network(new_network):
                network = new_network
                best_network = network
                best_size = len_size
                removed_any = True
                break
        
        if not removed_any:
            break

    # 2. Local Search / Simulated Annealing
    T = 1.0
    current_size = len(network)
    sa_start = time.time()
    
    while time.time() - start_time < time_limit:
        r = random.random()
        new_network = network[:]
        
        if r < 0.3:
            # Remove
            if len(network) > 0:
                idx = random.randint(0, len(network) - 1)
                new_network = network[:idx] + network[idx+1:]
        elif r < 0.6:
            # Insert
            i, j = random.sample(range(N), 2)
            if i > j: i, j = j, i
            pos = random.randint(0, len(network))
            new_network = network[:pos] + [(i, j)] + network[pos:]
        elif r < 0.8:
            # Swap positions
            if len(network) > 1:
                i, j = random.sample(range(len(network)), 2)
                new_network[i], new_network[j] = new_network[j], new_network[i]
        else:
            # Mutate wires
            if len(network) > 0:
                k = random.randint(0, len(network) - 1)
                i, j = network[k]
                # Pick a new partner for i
                new_j = random.randint(0, N - 1)
                if new_j != i:
                    if i > new_j:
                        new_network[k] = (new_j, i)
                    else:
                        new_network[k] = (i, new_j)

        new_size = len(new_network)
        
        if _verify_network(new_network):
            delta = new_size - current_size
            if delta < 0:
                network = new_network
                best_network = network
                best_size = new_size
                current_size = new_size
                T = max(0.1, T * 0.99)
            elif delta == 0:
                if random.random() < 0.2: # Higher acceptance for same size to explore
                    network = new_network
                    current_size = new_size
            else:
                # delta > 0
                p = np.exp(-delta / T)
                if random.random() < p:
                    network = new_network
                    current_size = new_size
                    T = max(0.1, T * 0.99)
        else:
            # Reject
            pass

    return best_network
