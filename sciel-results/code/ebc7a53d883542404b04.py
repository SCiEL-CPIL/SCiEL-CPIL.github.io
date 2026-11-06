import numpy as np
import time
import random

N = 17
NUM_INPUTS = 1 << N

def _make_all_inputs():
    arr = np.arange(NUM_INPUTS, dtype=np.uint32)
    inputs = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (arr >> i) & 1
    return inputs

def _verify_network(network, inputs):
    state = inputs.copy()
    for i, j in network:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    diffs = state[:, 1:] - state[:, :-1]
    return bool(np.all(diffs >= 0))

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    inputs = _make_all_inputs()
    
    # Start with odd-even transposition network
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    if not _verify_network(network, inputs):
        return network  # Should not happen
    
    best_network = list(network)
    best_count = len(network)
    
    random.seed(42)
    
    while time.time() - start_time < DEADLINE:
        # Greedy removal pass
        changed = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() - start_time > DEADLINE - 3:
                break
            candidate = network[:idx] + network[idx+1:]
            if _verify_network(candidate, inputs):
                network = candidate
                changed = True
        
        if changed:
            if len(network) < best_count:
                best_count = len(network)
                best_network = list(network)
        else:
            # Stuck; try add-remove cycles
            if time.time() - start_time > DEADLINE - 10:
                break
            
            found_improvement = False
            for attempt in range(50):
                if time.time() - start_time > DEADLINE - 3:
                    break
                
                # Add random comparator at random position
                i, j = random.sample(range(N), 2)
                if i > j:
                    i, j = j, i
                pos = random.randint(0, len(network))
                augmented = list(network)
                augmented.insert(pos, (i, j))
                
                # Greedy removal from augmented
                new_net = list(augmented)
                for idx2 in range(len(new_net) - 1, -1, -1):
                    if time.time() - start_time > DEADLINE - 3:
                        break
                    candidate = new_net[:idx2] + new_net[idx2+1:]
                    if _verify_network(candidate, inputs):
                        new_net = candidate
                
                if len(new_net) < len(network):
                    network = new_net
                    if len(network) < best_count:
                        best_count = len(network)
                        best_network = list(network)
                    found_improvement = True
                    break
            if not found_improvement:
                # Try a few random restarts with perturbation
                if time.time() - start_time > DEADLINE - 5:
                    break
                # Perturb: remove a few random comparators and add different ones
                perturbed = list(network)
                for _ in range(3):
                    if perturbed:
                        rem_idx = random.randint(0, len(perturbed) - 1)
                        perturbed.pop(rem_idx)
                        i, j = random.sample(range(N), 2)
                        if i > j:
                            i, j = j, i
                        add_pos = random.randint(0, len(perturbed))
                        perturbed.insert(add_pos, (i, j))
                
                # Greedy removal from perturbed
                new_net = list(perturbed)
                for idx2 in range(len(new_net) - 1, -1, -1):
                    if time.time() - start_time > DEADLINE - 3:
                        break
                    candidate = new_net[:idx2] + new_net[idx2+1:]
                    if _verify_network(candidate, inputs):
                        new_net = candidate
                
                if len(new_net) < len(network):
                    network = new_net
                    if len(network) < best_count:
                        best_count = len(network)
                        best_network = list(network)
                else:
                    # No improvement, break out
                    break
    
    return best_network
