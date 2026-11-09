"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def _apply_network(network, inputs):
    """Apply compare-exchange network to an array of values.
    network: list of (i,j) pairs
    inputs: array-like of N values
    Returns the sorted output as a list.
    """
    vals = list(inputs)
    for i, j in network:
        if vals[i] > vals[j]:
            vals[i], vals[j] = vals[j], vals[i]
    return vals

def _fitness(network):
    """
    Compute fitness: number of zero-one inputs that are NOT correctly sorted.
    Lower is better. 0 means perfect sorting network.
    """
    all_masks = np.arange(1 << N, dtype=np.int32)
    inputs_arr = ((all_masks[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int32)
    
    for i, j in network:
        mask_swap = inputs_arr[:, i] > inputs_arr[:, j]
        if np.any(mask_swap):
            tmp = inputs_arr[mask_swap, i].copy()
            inputs_arr[mask_swap, i] = inputs_arr[mask_swap, j]
            inputs_arr[mask_swap, j] = tmp
    
    diffs = np.diff(inputs_arr, axis=1)
    unsorted_rows = np.any(diffs < 0, axis=1)
    return int(np.sum(unsorted_rows))

def _fitness_sample(network, num_samples=200):
    """
    Fast fitness check on a random subset of inputs.
    Returns number of unsorted samples.
    """
    samples = np.random.randint(0, 1 << N, num_samples, dtype=np.int32)
    inputs_arr = ((samples[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int32)
    
    for i, j in network:
        mask_swap = inputs_arr[:, i] > inputs_arr[:, j]
        if np.any(mask_swap):
            tmp = inputs_arr[mask_swap, i].copy()
            inputs_arr[mask_swap, i] = inputs_arr[mask_swap, j]
            inputs_arr[mask_swap, j] = tmp
    
    diffs = np.diff(inputs_arr, axis=1)
    unsorted_rows = np.any(diffs < 0, axis=1)
    return int(np.sum(unsorted_rows))

def _layers_to_network(layers):
    """Convert list of permutation layers to list of comparator pairs."""
    network = []
    for perm in layers:
        for k in range(0, N, 2):
            i, j = perm[k], perm[k+1]
            if i > j:
                i, j = j, i
            network.append((i, j))
    return network

def _construct_impl():
    start_time = time.time()
    DEADLINE = 85  # seconds, leave 5s margin for safety
    
    random.seed(42)
    
    def random_layer():
        perm = random.sample(range(N), N)
        return perm
    
    pop_size = 50
    population = []
    
    # Initialize with random networks
    for _ in range(pop_size):
        num_layers = random.randint(8, 14)
        layers = [random_layer() for _ in range(num_layers)]
        network = _layers_to_network(layers)
        sample_fit = _fitness_sample(network, 50)
        if sample_fit == 0:
            full_fit = _fitness(network)
            if full_fit == 0:
                population.append((len(network), layers))
            else:
                population.append((1000 + len(network), layers))
        else:
            population.append((1000 + len(network), layers))
            
    best_net = None
    best_len = 10000
    
    # Check initial population
    for fit, layers in population:
        if fit < 1000:
            net = _layers_to_network(layers)
            if _fitness(net) == 0:
                if len(net) < best_len:
                    best_len = len(net)
                    best_net = net
    
    if best_net is None:
        # Fallback to seed network
        best_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        best_len = len(best_net)
        
    generations = 0
    while time.time() - start_time < DEADLINE - 10 and generations < 200:
        generations += 1
        
        population.sort(key=lambda x: x[0])
        parents = population[:10]
        
        new_pop = []
        new_pop.append(parents[0])
        
        while len(new_pop) < pop_size and time.time() - start_time < DEADLINE - 10:
            p1 = random.choice(parents)[1]
            p2 = random.choice(parents)[1]
            
            cut = random.randint(0, max(len(p1), len(p2)))
            child_layers = p1[:cut] + p2[cut:]
            
            for _ in range(random.randint(1, 3)):
                op = random.choice(['replace_layer', 'swap_layers', 'remove_layer', 'add_layer'])
                if op == 'replace_layer' and len(child_layers) > 0:
                    idx = random.randint(0, len(child_layers) - 1)
                    child_layers[idx] = random_layer()
                elif op == 'swap_layers' and len(child_layers) > 1:
                    i, j = random.sample(range(len(child_layers)), 2)
                    child_layers[i], child_layers[j] = child_layers[j], child_layers[i]
                elif op == 'remove_layer' and len(child_layers) > 5:
                    idx = random.randint(0, len(child_layers) - 1)
                    child_layers.pop(idx)
                elif op == 'add_layer' and len(child_layers) < 20:
                    idx = random.randint(0, len(child_layers))
                    child_layers.insert(idx, random_layer())
            
            network = _layers_to_network(child_layers)
            sample_fit = _fitness_sample(network, 100)
            
            if sample_fit == 0:
                full_fit = _fitness(network)
                if full_fit == 0:
                    fitness_val = len(network)
                    if len(network) < best_len:
                        best_len = len(network)
                        best_net = network
                else:
                    fitness_val = 1000 + len(network)
            else:
                fitness_val = 1000 + len(network)
                
            new_pop.append((fitness_val, child_layers))
            
        population = new_pop
        
    # Polish step
    if best_net is not None and best_len < 120:
        polished_net = best_net[:]
        for i in range(len(polished_net)):
            if time.time() - start_time > DEADLINE - 2: break
            candidate = polished_net[:i] + polished_net[i+1:]
            if _fitness(candidate) == 0:
                polished_net = candidate
                best_len = len(polished_net)
        best_net = polished_net
        
    return best_net

def construct():
    try:
        return _construct_impl()
    except Exception:
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-END
