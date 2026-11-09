import numpy as np
import random
import time

N = 17

# Pre-compute the input matrix for the 0-1 principle
# Shape: (2**N, N)
INPUTS = (np.arange(2**N)[:, None] // 2**np.arange(N-1, -1, -1)) % 2

def check_network(comparators):
    """
    Check if the given network sorts all 2^N boolean inputs.
    Returns True if valid, False otherwise.
    """
    if not comparators:
        return N == 1
    
    current = INPUTS.copy()
    
    for i, j in comparators:
        if i > j:
            i, j = j, i
        min_val = np.minimum(current[:, i], current[:, j])
        max_val = np.maximum(current[:, i], current[:, j])
        current[:, i] = min_val
        current[:, j] = max_val
        
    diffs = np.diff(current, axis=1)
    return np.all(diffs >= 0)

def construct():
    seed_network = [(i, i + 1) for phase in range(N)
                    for i in range(phase % 2, N - 1, 2)]
    
    current_network = seed_network[:]
    best_network = current_network[:]
    best_cost = len(current_network)
    
    start_time = time.time()
    timeout = 55.0
    
    def local_search(net):
        improved = True
        while improved and time.time() - start_time < timeout:
            improved = False
            # Try random moves
            # 1. Remove
            if len(net) > 0:
                idx = random.randint(0, len(net) - 1)
                candidate = net[:idx] + net[idx+1:]
                if check_network(candidate):
                    net = candidate
                    improved = True
                    continue
            
            # 2. Move
            if len(net) > 1:
                idx = random.randint(0, len(net) - 1)
                comp = net[idx]
                rest = net[:idx] + net[idx+1:]
                new_idx = random.randint(0, len(rest))
                candidate = rest[:new_idx] + [comp] + rest[new_idx:]
                if check_network(candidate):
                    net = candidate
                    improved = True
                    continue
            
            # 3. Swap
            if len(net) >= 2:
                idx1 = random.randint(0, len(net) - 1)
                idx2 = random.randint(0, len(net) - 1)
                if idx1 != idx2:
                    candidate = net[:]
                    candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
                    if check_network(candidate):
                        net = candidate
                        improved = True
                        continue
                        
            # 4. Insert (only if we are stuck and want to explore larger space? 
            #    Usually we minimize size. But sometimes a "detour" through larger size 
            #    leads to a smaller valid size? Unlikely in this metric. 
            #    However, if we are in an invalid state, inserting might help fix it?
            #    No, removing is usually better for invalid states if we want to get back to valid?
            #    Actually, if we are invalid, we just need *any* valid state.
            #    Inserting makes it harder to be valid? No, more comparators = more sorting power.
            #    So if we are invalid, inserting is a good way to become valid.
            #    If we are valid, inserting makes it worse (larger).
            
            if not improved:
                # If we are invalid, try inserting to fix
                # We don't know if we are invalid without checking, but we can assume
                # if we haven't found a valid move in a while, we might be stuck.
                # For simplicity, let's just allow insert if random chance says so.
                if random.random() < 0.1:
                    i = random.randint(0, N - 1)
                    j = random.randint(i + 1, N - 1)
                    idx = random.randint(0, len(net))
                    candidate = net[:idx] + [(i, j)] + net[idx:]
                    if check_network(candidate):
                        net = candidate
                        improved = True
                        continue

        return net

    # Iterated Local Search
    while time.time() - start_time < timeout:
        # Local Search
        current_network = local_search(current_network)
        
        # Update best
        if len(current_network) < best_cost:
            best_cost = len(current_network)
            best_network = current_network[:]
            
        # Perturb
        # Apply K random moves to current_network
        # We don't check validity during perturbation
        num_perturbations = random.randint(2, 5)
        for _ in range(num_perturbations):
            move_type = random.choice(['remove', 'insert', 'move', 'swap'])
            if move_type == 'remove' and len(current_network) > 0:
                idx = random.randint(0, len(current_network) - 1)
                current_network = current_network[:idx] + current_network[idx+1:]
            elif move_type == 'insert':
                i = random.randint(0, N - 1)
                j = random.randint(i + 1, N - 1)
                idx = random.randint(0, len(current_network))
                current_network = current_network[:idx] + [(i, j)] + current_network[idx:]
            elif move_type == 'move' and len(current_network) > 1:
                idx = random.randint(0, len(current_network) - 1)
                comp = current_network[idx]
                rest = current_network[:idx] + current_network[idx+1:]
                new_idx = random.randint(0, len(rest))
                current_network = rest[:new_idx] + [comp] + rest[new_idx:]
            elif move_type == 'swap' and len(current_network) >= 2:
                idx1 = random.randint(0, len(current_network) - 1)
                idx2 = random.randint(0, len(current_network) - 1)
                if idx1 != idx2:
                    current_network[idx1], current_network[idx2] = current_network[idx2], current_network[idx1]
            
    return best_network
