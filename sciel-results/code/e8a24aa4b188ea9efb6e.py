"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 17

# EVOLVE-BLOCK-START
def _check_network(network, full_check=True, sample_size=2000):
    """
    Check if the network sorts all inputs.
    Uses 0-1 principle.
    """
    if full_check:
        indices = np.arange(2 ** N)
        inputs = ((indices[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int8)
        
        for i, j in network:
            min_val = np.minimum(inputs[:, i], inputs[:, j])
            max_val = np.maximum(inputs[:, i], inputs[:, j])
            inputs[:, i] = min_val
            inputs[:, j] = max_val
            
        diffs = np.diff(inputs, axis=1)
        if np.any(diffs < 0):
            return False
        return True
    else:
        # Use a fixed set of hard samples for faster rejection
        # Include all-zero, all-one, single ones, single zeros, random
        if not hasattr(_check_network, '_hard_samples'):
            samples_list = []
            # Single ones
            for i in range(N):
                s = np.zeros(N, dtype=np.int8)
                s[i] = 1
                samples_list.append(s)
            # Single zeros
            for i in range(N):
                s = np.ones(N, dtype=np.int8)
                s[i] = 0
                samples_list.append(s)
            # Half zeros, half ones (random positions)
            for _ in range(200):
                s = np.random.randint(0, 2, size=N).astype(np.int8)
                samples_list.append(s)
            # Random
            for _ in range(sample_size):
                s = np.random.randint(0, 2, size=N).astype(np.int8)
                samples_list.append(s)
            _check_network._hard_samples = np.array(samples_list)
            
        samples = _check_network._hard_samples.copy()
        
        for i, j in network:
            min_val = np.minimum(samples[:, i], samples[:, j])
            max_val = np.maximum(samples[:, i], samples[:, j])
            samples[:, i] = min_val
            samples[:, j] = max_val
            
        diffs = np.diff(samples, axis=1)
        if np.any(diffs < 0):
            return False
        return True

def _generate_oet_network(n):
    """Generate Odd-Even Transposition Network."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def construct():
    """
    Construct a sorting network for 17 wires using local search optimization.
    Starts with Odd-Even Transposition and tries to remove/move comparators.
    """
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer for overhead
    
    # Start with OET
    network = _generate_oet_network(N)
    current_size = len(network)
    
    # Verify initial network
    if not _check_network(network, full_check=True):
        return network

    best_network = network.copy()
    best_size = current_size
    
    attempts = 0
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        attempts += 1
        if not network:
            break
            
        # Randomly choose a move type
        # 80% remove, 20% move
        if len(network) > 1:
            move_type = random.choice(['remove', 'move', 'remove', 'remove', 'remove', 'move'])
        else:
            move_type = 'remove'
            
        original_network = network.copy()
        
        if move_type == 'remove':
            idx = random.randint(0, len(network) - 1)
            new_network = network[:idx] + network[idx+1:]
            new_size = len(new_network)
            
            # Quick check
            if _check_network(new_network, full_check=False):
                # Full check
                if _check_network(new_network, full_check=True):
                    network = new_network
                    current_size = new_size
                    if current_size < best_size:
                        best_size = current_size
                        best_network = network.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
                    network = original_network
        elif move_type == 'move':
            idx = random.randint(0, len(network) - 1)
            comp = network[idx]
            # Remove it
            new_network = network[:idx] + network[idx+1:]
            # Insert at random position
            new_idx = random.randint(0, len(new_network))
            new_network = new_network[:new_idx] + [comp] + new_network[new_idx:]
            
            # Quick check
            if _check_network(new_network, full_check=False):
                # Full check
                if _check_network(new_network, full_check=True):
                    network = new_network
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    network = original_network

        # If stuck for too long, reset to best and try a different strategy
        if no_improve_count > max_no_improve:
            network = best_network.copy()
            current_size = best_size
            no_improve_count = 0
            max_no_improve = int(max_no_improve * 1.5)
            
    return best_network
# EVOLVE-BLOCK-END
