import time
import random
import numpy as np

N = 16
ALL_INPUTS = np.arange(2**N, dtype=np.uint32)

def _verify_network_full(network):
    """
    Verify if the network sorts all 2^16 inputs using bit-parallel operations.
    Returns True if valid, False otherwise.
    """
    if not network:
        # Check if identity is sorted (it's not, unless N=1)
        # But for N=16, identity is never sorted.
        # Actually, if network is empty, it's only valid if all inputs are already sorted.
        # Only 0 and 2^16-1 are sorted in identity.
        # So empty network is invalid for N=16.
        return False
        
    states = ALL_INPUTS.copy()
    for i, j in network:
        bit_i = (states >> i) & 1
        bit_j = (states >> j) & 1
        # If bit_i > bit_j (i.e., i=1, j=0), swap
        # swap_mask is 1 where we need to swap
        swap_mask = bit_i & (~bit_j) & 1
        mask = (swap_mask == 1)
        if np.any(mask):
            states[mask] ^= ((1 << i) | (1 << j))
            
    # Check if all states are sorted
    # Sorted: 0...01...1
    # x = ~state & 0xFFFF should be 0...01...1 (contiguous 1s at LSB)
    # i.e., x & (x + 1) == 0
    x = (~states) & 0xFFFF
    sorted_mask = (x & (x + 1)) == 0
    return np.all(sorted_mask)

def _get_unsorted_mask(states):
    """
    Returns a boolean mask of unsorted states.
    """
    x = (~states) & 0xFFFF
    return (x & (x + 1)) != 0

def _count_unsorted(states):
    """
    Returns the number of unsorted states.
    """
    return np.sum(_get_unsorted_mask(states))

def _apply_network_to_states(states, network):
    """
    Apply the network to the given states array (in-place).
    """
    for i, j in network:
        bit_i = (states >> i) & 1
        bit_j = (states >> j) & 1
        swap_mask = bit_i & (~bit_j) & 1
        mask = (swap_mask == 1)
        if np.any(mask):
            states[mask] ^= ((1 << i) | (1 << j))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_network = None
    best_score = float('inf')
    
    # All possible comparators
    all_comparators = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Initial network: Odd-Even Transposition Sort (120 comparators)
    # This is a known valid network.
    initial_network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Verify initial network just in case
    if not _verify_network_full(initial_network):
        # Fallback to a simpler valid network if odd-even fails (shouldn't happen)
        # But let's assume it works.
        pass
        
    current_network = initial_network.copy()
    current_score = len(current_network)
    
    # Precompute states for the current network to avoid re-computing from scratch on every step
    # We will maintain `current_states` and update it incrementally.
    
    def compute_states(net):
        s = ALL_INPUTS.copy()
        _apply_network_to_states(s, net)
        return s
        
    current_states = compute_states(current_network)
    current_unsorted = _count_unsorted(current_states)
    
    # Simulated Annealing
    # Temperature T starts high, cools down.
    # Cost function: len(network) + penalty * unsorted_count
    # We want to minimize len(network).
    # If unsorted_count > 0, the network is invalid.
    # We treat invalid networks as having a very high cost, but we allow them to escape local minima.
    # Cost = len(network) + 1000 * unsorted_count
    
    def get_cost(net, unsorted_count):
        return len(net) + 1000 * unsorted_count
        
    current_cost = get_cost(current_network, current_unsorted)
    
    T = 50.0  # Initial temperature
    cooling_rate = 0.995
    
    # Multi-start: We will run SA for a fixed number of steps or until time runs out.
    # We might restart with a different initial network if stuck.
    
    steps = 0
    max_steps = 5000  # Bit-parallel verification is fast, so we can do many steps.
    
    while time.time() < deadline:
        if steps >= max_steps:
            # Restart with a new random network or a perturbed version of the best
            # For simplicity, let's just continue cooling if we have time, or break if T is very low
            if T < 0.1:
                # Try a multi-start: pick a random valid network?
                # Or just perturb the current best.
                # Let's perturb the current network by adding 5 random comparators.
                current_network = best_network.copy() if best_network else initial_network.copy()
                for _ in range(5):
                    c = random.choice(all_comparators)
                    current_network.append(c)
                current_states = compute_states(current_network)
                current_unsorted = _count_unsorted(current_states)
                current_cost = get_cost(current_network, current_unsorted)
                T = 50.0
                steps = 0
            else:
                T *= cooling_rate
                steps = 0
                continue
        
        steps += 1
        T *= cooling_rate
        
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove a comparator
            if len(current_network) > 0:
                idx = random.randint(0, len(current_network) - 1)
                removed_comp = current_network.pop(idx)
                new_cost = get_cost(current_network, current_unsorted) # Unsorted count doesn't change immediately in this naive view?
                # Wait, removing a comparator changes the states!
                # We need to recompute states or update them.
                # Recomputing states is O(C * 2^16). 
                # Updating states incrementally for a removal is hard because the network order changes.
                # So we must recompute states for the candidate network.
                
                # To make this efficient, we should only recompute if we are going to accept or if we need to evaluate cost.
                # Let's recompute states for the candidate.
                candidate_states = ALL_INPUTS.copy()
                _apply_network_to_states(candidate_states, current_network)
                candidate_unsorted = _count_unsorted(candidate_states)
                new_cost = get_cost(current_network, candidate_unsorted)
                
                delta_cost = new_cost - current_cost
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    # Accept
                    current_states = candidate_states
                    current_unsorted = candidate_unsorted
                    current_cost = new_cost
                    
                    if current_unsorted == 0 and len(current_network) < best_score:
                        best_score = len(current_network)
                        best_network = current_network.copy()
                else:
                    # Reject: restore the removed comparator
                    current_network.insert(idx, removed_comp)
                    # States remain current_states (which correspond to the old network)
                    
        elif move_type < 0.7:
            # Insert a comparator
            comp = random.choice(all_comparators)
            pos = random.randint(0, len(current_network))
            current_network.insert(pos, comp)
            
            candidate_states = ALL_INPUTS.copy()
            _apply_network_to_states(candidate_states, current_network)
            candidate_unsorted = _count_unsorted(candidate_states)
            new_cost = get_cost(current_network, candidate_unsorted)
            
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                # Accept
                current_states = candidate_states
                current_unsorted = candidate_unsorted
                current_cost = new_cost
                
                if current_unsorted == 0 and len(current_network) < best_score:
                    best_score = len(current_network)
                    best_network = current_network.copy()
            else:
                # Reject
                current_network.pop(pos)
                
        else:
            # Swap two comparators
            if len(current_network) >= 2:
                i, j = random.sample(range(len(current_network)), 2)
                current_network[i], current_network[j] = current_network[j], current_network[i]
                
                candidate_states = ALL_INPUTS.copy()
                _apply_network_to_states(candidate_states, current_network)
                candidate_unsorted = _count_unsorted(candidate_states)
                new_cost = get_cost(current_network, candidate_unsorted)
                
                delta_cost = new_cost - current_cost
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    # Accept
                    current_states = candidate_states
                    current_unsorted = candidate_unsorted
                    current_cost = new_cost
                    
                    if current_unsorted == 0 and len(current_network) < best_score:
                        best_score = len(current_network)
                        best_network = current_network.copy()
                else:
                    # Reject
                    current_network[i], current_network[j] = current_network[j], current_network[i]

    # Final check: ensure best_network is valid
    if best_network is None:
        best_network = initial_network
        
    if _verify_network_full(best_network):
        return best_network
    else:
        # If somehow invalid, return the known valid initial network
        return initial_network

# Ensure the function is defined exactly as required
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # All possible comparators
    all_comparators = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Initial network: Odd-Even Transposition Sort
    initial_network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    best_network = initial_network.copy()
    best_score = len(best_network)
    
    current_network = initial_network.copy()
    
    def compute_states(net):
        s = ALL_INPUTS.copy()
        _apply_network_to_states(s, net)
        return s
        
    current_states = compute_states(current_network)
    current_unsorted = _count_unsorted(current_states)
    
    def get_cost(net, unsorted_count):
        return len(net) + 1000 * unsorted_count
        
    current_cost = get_cost(current_network, current_unsorted)
    
    T = 50.0
    cooling_rate = 0.995
    steps = 0
    max_steps = 5000
    
    while time.time() < deadline:
        if steps >= max_steps:
            if T < 0.1:
                # Multi-start: Perturb best network
                if best_network:
                    current_network = best_network.copy()
                else:
                    current_network = initial_network.copy()
                
                # Add some random noise
                for _ in range(5):
                    c = random.choice(all_comparators)
                    current_network.append(c)
                
                current_states = compute_states(current_network)
                current_unsorted = _count_unsorted(current_states)
                current_cost = get_cost(current_network, current_unsorted)
                T = 50.0
                steps = 0
            else:
                T *= cooling_rate
                steps = 0
                continue
        
        steps += 1
        T *= cooling_rate
        
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove
            if len(current_network) > 0:
                idx = random.randint(0, len(current_network) - 1)
                removed_comp = current_network.pop(idx)
                
                candidate_states = compute_states(current_network)
                candidate_unsorted = _count_unsorted(candidate_states)
                new_cost = get_cost(current_network, candidate_unsorted)
                
                delta_cost = new_cost - current_cost
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    current_states = candidate_states
                    current_unsorted = candidate_unsorted
                    current_cost = new_cost
                    
                    if current_unsorted == 0 and len(current_network) < best_score:
                        best_score = len(current_network)
                        best_network = current_network.copy()
                else:
                    current_network.insert(idx, removed_comp)
                    
        elif move_type < 0.7:
            # Insert
            comp = random.choice(all_comparators)
            pos = random.randint(0, len(current_network))
            current_network.insert(pos, comp)
            
            candidate_states = compute_states(current_network)
            candidate_unsorted = _count_unsorted(candidate_states)
            new_cost = get_cost(current_network, candidate_unsorted)
            
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                current_states = candidate_states
                current_unsorted = candidate_unsorted
                current_cost = new_cost
                
                if current_unsorted == 0 and len(current_network) < best_score:
                    best_score = len(current_network)
                    best_network = current_network.copy()
            else:
                current_network.pop(pos)
                
        else:
            # Swap
            if len(current_network) >= 2:
                i, j = random.sample(range(len(current_network)), 2)
                current_network[i], current_network[j] = current_network[j], current_network[i]
                
                candidate_states = compute_states(current_network)
                candidate_unsorted = _count_unsorted(candidate_states)
                new_cost = get_cost(current_network, candidate_unsorted)
                
                delta_cost = new_cost - current_cost
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    current_states = candidate_states
                    current_unsorted = candidate_unsorted
                    current_cost = new_cost
                    
                    if current_unsorted == 0 and len(current_network) < best_score:
                        best_score = len(current_network)
                        best_network = current_network.copy()
                else:
                    current_network[i], current_network[j] = current_network[j], current_network[i]

    if _verify_network_full(best_network):
        return best_network
    else:
        return initial_network

# Global N
N = 16
ALL_INPUTS = np.arange(2**N, dtype=np.uint32)

def _apply_network_to_states(states, network):
    for i, j in network:
        bit_i = (states >> i) & 1
        bit_j = (states >> j) & 1
        swap_mask = bit_i & (~bit_j) & 1
        mask = (swap_mask == 1)
        if np.any(mask):
            states[mask] ^= ((1 << i) | (1 << j))

def _count_unsorted(states):
    x = (~states) & 0xFFFF
    return np.sum((x & (x + 1)) != 0)

def _verify_network_full(network):
    if not network:
        return False
    s = ALL_INPUTS.copy()
    _apply_network_to_states(s, network)
    return _count_unsorted(s) == 0

# Re-define construct to use the globals correctly
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    all_comparators = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    initial_network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    best_network = initial_network.copy()
    best_score = len(best_network)
    
    current_network = initial_network.copy()
    
    def compute_states(net):
        s = ALL_INPUTS.copy()
        _apply_network_to_states(s, net)
        return s
        
    current_states = compute_states(current_network)
    current_unsorted = _count_unsorted(current_states)
    
    def get_cost(net, unsorted_count):
        return len(net) + 1000 * unsorted_count
        
    current_cost = get_cost(current_network, current_unsorted)
    
    T = 50.0
    cooling_rate = 0.995
    steps = 0
    max_steps = 5000
    
    while time.time() < deadline:
        if steps >= max_steps:
            if T < 0.1:
                if best_network:
                    current_network = best_network.copy()
                else:
                    current_network = initial_network.copy()
                
                for _ in range(5):
                    c = random.choice(all_comparators)
                    current_network.append(c)
                
                current_states = compute_states(current_network)
                current_unsorted = _count_unsorted(current_states)
                current_cost = get_cost(current_network, current_unsorted)
                T = 50.0
                steps = 0
            else:
                T *= cooling_rate
                steps = 0
                continue
        
        steps += 1
        T *= cooling_rate
        
        move_type = random.random()
        
        if move_type < 0.4:
            if len(current_network) > 0:
                idx = random.randint(0, len(current_network) - 1)
                removed_comp = current_network.pop(idx)
                
                candidate_states = compute_states(current_network)
                candidate_unsorted = _count_unsorted(candidate_states)
                new_cost = get_cost(current_network, candidate_unsorted)
                
                delta_cost = new_cost - current_cost
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    current_states = candidate_states
                    current_unsorted = candidate_unsorted
                    current_cost = new_cost
                    
                    if current_unsorted == 0 and len(current_network) < best_score:
                        best_score = len(current_network)
                        best_network = current_network.copy()
                else:
                    current_network.insert(idx, removed_comp)
                    
        elif move_type < 0.7:
            comp = random.choice(all_comparators)
            pos = random.randint(0, len(current_network))
            current_network.insert(pos, comp)
            
            candidate_states = compute_states(current_network)
            candidate_unsorted = _count_unsorted(candidate_states)
            new_cost = get_cost(current_network, candidate_unsorted)
            
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                current_states = candidate_states
                current_unsorted = candidate_unsorted
                current_cost = new_cost
                
                if current_unsorted == 0 and len(current_network) < best_score:
                    best_score = len(current_network)
                    best_network = current_network.copy()
            else:
                current_network.pop(pos)
                
        else:
            if len(current_network) >= 2:
                i, j = random.sample(range(len(current_network)), 2)
                current_network[i], current_network[j] = current_network[j], current_network[i]
                
                candidate_states = compute_states(current_network)
                candidate_unsorted = _count_unsorted(candidate_states)
                new_cost = get_cost(current_network, candidate_unsorted)
                
                delta_cost = new_cost - current_cost
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    current_states = candidate_states
                    current_unsorted = candidate_unsorted
                    current_cost = new_cost
                    
                    if current_unsorted == 0 and len(current_network) < best_score:
                        best_score = len(current_network)
                        best_network = current_network.copy()
                else:
                    current_network[i], current_network[j] = current_network[j], current_network[i]

    if _verify_network_full(best_network):
        return best_network
    else:
        return initial_network

