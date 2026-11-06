import numpy as np
import time
import random

N = 18
NUM_INPUTS = 2 ** N

def _generate_inputs():
    """Generate all 2^N boolean input vectors as a uint8 array."""
    x = np.arange(NUM_INPUTS, dtype=np.uint64)[:, None]
    bits = (x >> np.arange(N, dtype=np.uint64)) & 1
    return bits.astype(np.uint8)

def _check_sorted(state):
    """Check if all rows of state are sorted (non-decreasing)."""
    if N <= 1:
        return True
    return np.all(state[:, :-1] <= state[:, 1:])

def _apply_comparators(state, network, start_idx, end_idx):
    """Apply comparators from start_idx to end_idx-1 to state in-place. Returns state."""
    for k in range(start_idx, end_idx):
        i, j = network[k]
        vi = state[:, i]
        vj = state[:, j]
        min_val = vi & vj
        max_val = vi | vj
        state[:, i] = min_val
        state[:, j] = max_val
    return state

def _fast_verify_network(network, inputs, prefix_states):
    """
    Verify network using cached prefix states.
    prefix_states[k] = state after applying first k comparators of the CURRENT network.
    But this is tricky because the network changes.
    
    Alternative: just simulate from scratch but with early exit if possible.
    Actually, the simplest speedup: vectorize the simulation loop better.
    """
    if not network:
        return _check_sorted(inputs)
    
    state = inputs.copy()
    for i, j in network:
        vi = state[:, i]
        vj = state[:, j]
        min_val = vi & vj
        max_val = vi | vj
        state[:, i] = min_val
        state[:, j] = max_val
    
    return _check_sorted(state)

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _build_prefix_states(network, inputs):
    """
    Build prefix states: prefix_states[k] is the state after applying first k comparators.
    prefix_states[0] = inputs
    prefix_states[k] = state after applying network[0..k-1]
    """
    prefix_states = [inputs.copy()]
    state = inputs.copy()
    for k, (i, j) in enumerate(network):
        vi = state[:, i]
        vj = state[:, j]
        min_val = vi & vj
        max_val = vi | vj
        state[:, i] = min_val
        state[:, j] = max_val
        prefix_states.append(state.copy())
    return prefix_states

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    inputs = _generate_inputs()
    
    network = _odd_even_transposition()
    best_network = network.copy()
    
    # Build prefix states for the current network
    prefix_states = _build_prefix_states(network, inputs)
    
    consecutive_failures = 0
    max_failures = 50
    
    while time.time() < deadline:
        if len(network) == 0:
            break
        
        # Try to remove a random comparator
        idx = random.randint(0, len(network) - 1)
        comp = network.pop(idx)
        
        # To verify network without comparator at idx:
        # We need to simulate: apply network[0:idx], then network[idx+1:]
        # We can use prefix_states[idx] as the starting state, then apply the rest.
        
        if idx < len(prefix_states):
            # Use cached prefix state
            state = prefix_states[idx].copy()
            # Apply comparators from idx to end of the NEW network (which is old network without idx)
            # The new network is network[0:idx] + network[idx:] (old indexing)
            # But we already popped idx, so network is now the new network.
            # We need to apply network[idx:] to state (which is state after first idx comparators)
            for k in range(idx, len(network)):
                i, j = network[k]
                vi = state[:, i]
                vj = state[:, j]
                min_val = vi & vj
                max_val = vi | vj
                state[:, i] = min_val
                state[:, j] = max_val
            
            is_valid = _check_sorted(state)
        else:
            # Fallback: full simulation
            state = inputs.copy()
            for i, j in network:
                vi = state[:, i]
                vj = state[:, j]
                min_val = vi & vj
                max_val = vi | vj
                state[:, i] = min_val
                state[:, j] = max_val
            is_valid = _check_sorted(state)
        
        if is_valid:
            # Success: rebuild prefix states
            prefix_states = _build_prefix_states(network, inputs)
            best_network = network.copy()
            consecutive_failures = 0
        else:
            # Failure: restore
            network.insert(idx, comp)
            consecutive_failures += 1
        
        # Check time
        if time.time() > deadline:
            break
    
    return best_network
