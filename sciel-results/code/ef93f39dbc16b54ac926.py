"""Size-efficient sorting network on 17 wires using Simulated Annealing with Bitmask Verification."""
import numpy as np
import random
import math

N = 17
NUM_INPUTS = 1 << N  # 131,072

# Precompute the bitmask for each input wire.
# wire_bitmask[i] is an integer of NUM_INPUTS bits where bit k is 1 iff wire i is 1 in input k.
def _build_wire_bitmasks():
    masks = [0] * N
    for k in range(NUM_INPUTS):
        for i in range(N):
            if (k >> i) & 1:
                masks[i] |= (1 << k)
    return masks

_WIRE_MASKS = _build_wire_bitmasks()

def _apply_comparator(state, i, j):
    """Apply compare-exchange on wires i and j (i < j) in-place on state."""
    new_i = state[i] & state[j]
    new_j = state[i] | state[j]
    state[i] = new_i
    state[j] = new_j

def _get_violations(state):
    """Count the total number of unsorted pairs across all inputs."""
    violations = 0
    for i in range(N - 1):
        # If wire i is 1 and wire i+1 is 0, it's a violation for that input.
        # state[i] & ~state[i+1] gives the bitmask of inputs where this violation occurs.
        violations += (state[i] & ~state[i+1]).bit_count()
    return violations

def _evaluate(network):
    """Evaluate a network: return (num_comparators, violations)."""
    state = list(_WIRE_MASKS)
    for i, j in network:
        _apply_comparator(state, i, j)
    return len(network), _get_violations(state)

def construct():
    """Construct a sorting network for N=17 wires using Simulated Annealing."""
    
    # 1. Generate a seed network (Odd-Even Transposition)
    def _odd_even_transposition(n):
        comps = []
        for phase in range(n):
            if phase % 2 == 0:
                for i in range(0, n - 1, 2):
                    comps.append((i, i + 1))
            else:
                for i in range(1, n - 1, 2):
                    comps.append((i, i + 1))
        return comps
    
    current_network = _odd_even_transposition(N)
    current_len, current_violations = _evaluate(current_network)
    current_cost = current_len + 10 * current_violations
    
    best_network = list(current_network)
    best_cost = current_cost
    
    # 2. Simulated Annealing
    # Parameters
    T_start = 1.0
    T_end = 0.01
    steps = 50000  # Adjust based on time limit
    
    # Pre-generate random pairs? No, just use random.
    
    for step in range(steps):
        # Cooling schedule
        T = T_start * (T_end / T_start) ** (step / steps)
        
        # Create a copy of the current network to modify
        new_network = list(current_network)
        
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.4 and len(new_network) > 0:
            # Delete a random comparator
            idx = random.randint(0, len(new_network) - 1)
            new_network.pop(idx)
            
        elif move_type < 0.7:
            # Insert a random comparator at a random position
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(new_network))
            new_network.insert(pos, (i, j))
            
        elif move_type < 0.9 and len(new_network) > 0:
            # Modify a random comparator
            idx = random.randint(0, len(new_network) - 1)
            i, j = new_network[idx]
            if random.random() < 0.5:
                # Change i
                new_i = random.randint(0, j - 1)
                new_network[idx] = (new_i, j)
            else:
                # Change j
                new_j = random.randint(i + 1, N - 1)
                new_network[idx] = (i, new_j)
                
        else:
            # Move a random comparator to a new position
            if len(new_network) > 1:
                idx = random.randint(0, len(new_network) - 1)
                comp = new_network.pop(idx)
                new_idx = random.randint(0, len(new_network))
                new_network.insert(new_idx, comp)

        # Evaluate new network
        new_len, new_violations = _evaluate(new_network)
        new_cost = new_len + 10 * new_violations
        
        # Acceptance criterion
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_network = new_network
            current_len = new_len
            current_violations = new_violations
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_network = list(current_network)
                
    # 3. Final Pruning (Greedy) on the best network found
    # This helps remove any remaining redundancies that annealing might have left.
    def _validate_network(comps):
        st = list(_WIRE_MASKS)
        for i, j in comps:
            _apply_comparator(st, i, j)
        return _get_violations(st) == 0
    
    if _validate_network(best_network):
        changed = True
        while changed:
            changed = False
            for idx in range(len(best_network)):
                reduced = best_network[:idx] + best_network[idx+1:]
                if _validate_network(reduced):
                    best_network = reduced
                    changed = True
                    break
    
    return best_network
