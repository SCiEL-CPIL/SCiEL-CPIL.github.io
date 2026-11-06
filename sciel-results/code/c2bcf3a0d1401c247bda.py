"""Size-efficient sorting networks on 17 wires via NumPy vectorized search."""
import time
import random
import numpy as np

N = 17
NUM_INPUTS = 1 << N
DEADLINE = 55.0  # Safety margin before 60s limit
START_TIME = time.time()

# Precompute initial states for all 2^N inputs using NumPy.
# Shape: (NUM_INPUTS, N), dtype: uint8
# input_states[input_idx, wire_idx] = 1 if wire has 1, else 0
def _init_states():
    # Generate all input vectors
    # We can use np.arange and bit operations, or just iterate.
    # np.arange is fast.
    inputs = np.arange(NUM_INPUTS, dtype=np.uint32)
    # Expand to 2D: shape (NUM_INPUTS, N)
    # We want bit k of inputs to be in column k.
    # (inputs[:, None] >> np.arange(N)) & 1
    states = (inputs[:, None] >> np.arange(N)) & 1
    return states.astype(np.uint8)

INITIAL_STATES = _init_states()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def verify_network(comps):
    """
    Verify network using NumPy vectorized simulation.
    comps: list of (i, j) tuples.
    Returns True if valid.
    """
    if not comps:
        # Check if initial states are sorted (only true for N=1 or specific cases, usually false for N>1)
        # For N=17, initial state is not sorted.
        return False
        
    # Copy initial states
    # We need a mutable copy. .copy() is fast enough.
    state = INITIAL_STATES.copy()
    
    for i, j in comps:
        wi = state[:, i]
        wj = state[:, j]
        
        # Comparator (i, j): min to i, max to j.
        # Vectorized min/max
        new_wi = np.minimum(wi, wj)
        new_wj = np.maximum(wi, wj)
        
        state[:, i] = new_wi
        state[:, j] = new_wj
        
    # Check if sorted: for all rows, state[:, k] <= state[:, k+1]
    # This is equivalent to checking if any state[:, k] > state[:, k+1]
    # diff = state[:, 1:] - state[:, :-1]
    # if np.any(diff < 0): return False
    # Faster: check if state[:, k] & ~state[:, k+1] has any 1s?
    # Actually, since values are 0 or 1:
    # sorted if state[:, k] <= state[:, k+1] for all k.
    # This is true if there is no index where state[:, k] == 1 and state[:, k+1] == 0.
    # i.e., ~np.any(state[:, :-1] & (1 - state[:, 1:]))
    
    # Cast to signed integer to avoid wrap-around in subtraction
    # uint8 subtraction wraps around (0-1=255), which breaks the < 0 check
    state_signed = state.astype(np.int8)
    diffs = state_signed[:, 1:] - state_signed[:, :-1]
    if np.any(diffs < 0):
        return False
        
    return True

def make_seed():
    """Odd-even transposition network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def greedy_remove(comps):
    """
    Try to remove each comparator if the network remains valid.
    """
    n = len(comps)
    improved = True
    while improved:
        improved = False
        # Iterate backwards to handle list modifications safely if we were modifying in place, 
        # but we are creating new lists.
        for idx in range(n - 1, -1, -1):
            if _time_left() < 2.0:
                return comps
            candidate = comps[:idx] + comps[idx+1:]
            if verify_network(candidate):
                comps = candidate
                n = len(comps)
                improved = True
                break # Restart scan
    return comps

def construct():
    """Construct a sorting network for 17 wires, minimizing comparator count."""
    global START_TIME
    START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Start with odd-even transposition
    comps = make_seed()
    # print(f"Seed network: {len(comps)} comparators", flush=True)
    
    # First pass: aggressive greedy removal
    comps = greedy_remove(comps)
    # print(f"After greedy removal: {len(comps)} comparators", flush=True)
    
    best = list(comps)
    best_len = len(best)
    
    # Local Search / Genetic Algorithm
    # We will use a simple genetic algorithm or hill climbing.
    # Given the fast verifier, we can do many iterations.
    
    population = [list(best)]
    # Add some random variations to population?
    # Let's stick to hill climbing with restarts or a simple GA.
    
    # Strategy:
    # 1. Hill climb from best.
    # 2. If stuck, mutate and restart.
    
    current = list(best)
    current_len = best_len
    
    while _time_left() > 1.0:
        # Generate a candidate via mutation
        candidate = list(current)
        n = len(candidate)
        
        # Mutation types:
        # 1. Remove a comparator (if valid, it's a win)
        # 2. Add a comparator (usually bad for size, but might enable removals? Unlikely in simple search)
        # 3. Swap two comparators
        # 4. Move a comparator
        
        # Since we want to minimize size, "Remove" is the most powerful move.
        # "Swap" and "Move" change the structure but keep size. They can help escape local minima 
        # where no single removal is valid, but a sequence of swap+remove is.
        
        op = rng.choice(['remove', 'swap', 'move', 'remove', 'swap', 'move']) # Weight towards removal
        
        if op == 'remove':
            if n == 0:
                continue
            idx = rng.randrange(n)
            candidate.pop(idx)
        elif op == 'swap' and n >= 2:
            i1 = rng.randrange(n)
            i2 = rng.randrange(n)
            if i1 != i2:
                candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
        elif op == 'move' and n >= 1:
            i1 = rng.randrange(n)
            i2 = rng.randrange(n)
            if i1 != i2:
                elem = candidate.pop(i1)
                candidate.insert(i2, elem)
        
        # Evaluate candidate
        # Optimization: If candidate is larger than current, it's unlikely to be useful unless it leads to a much smaller network later.
        # But in pure hill climbing, we only accept if it's valid and smaller (for remove) or valid and same size (for swap/move) to explore.
        # However, exploring same-size valid networks is expensive.
        # Let's prioritize finding valid smaller networks.
        
        # If we removed, size is smaller. Check validity.
        # If we swapped/moved, size is same. Check validity.
        
        if verify_network(candidate):
            if len(candidate) < current_len:
                current = candidate
                current_len = len(candidate)
                if current_len < best_len:
                    best = list(current)
                    best_len = current_len
                    # print(f"New best: {best_len}", flush=True)
                
                # Greedy remove after improvement
                current = greedy_remove(current)
                current_len = len(current)
                if current_len < best_len:
                    best = list(current)
                    best_len = current_len
                    # print(f"Greedy improved to: {best_len}", flush=True)
                    
            else:
                # Same size or larger (shouldn't be larger if we only do remove/swap/move)
                # If same size and valid, we might update current to explore this new local region
                # but only if it's different from current
                if len(candidate) == current_len and candidate != current:
                    # Accept this move to explore, but don't update best
                    current = candidate
                    # Optionally run greedy remove to see if we can shrink
                    # This is risky time-wise, but verifier is fast
                    # Let's just do a quick greedy remove check
                    temp = greedy_remove(current)
                    if len(temp) < current_len:
                        current = temp
                        current_len = len(current)
                        if current_len < best_len:
                            best = list(current)
                            best_len = current_len
                            # print(f"Greedy from move improved to: {best_len}", flush=True)
        else:
            # Invalid. Do nothing.
            pass
            
    # Final greedy pass
    best = greedy_remove(best)
    # print(f"Final: {len(best)} comparators", flush=True)
    
    return best
