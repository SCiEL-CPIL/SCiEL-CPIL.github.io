import time
import random

N = 17

# Precompute initial state bitmasks
# state[i] is a bitmask where bit k is 1 if input k has a 1 at wire i
# Input k is an integer 0 to 2^N - 1
INITIAL_STATE = [0] * N
for i in range(N):
    mask = 0
    bit_val = 1 << i
    for k in range(1 << N):
        if k & bit_val:
            mask |= (1 << k)
    INITIAL_STATE[i] = mask

def _is_valid(network):
    """
    Check if network sorts all 0/1 inputs using zero-one principle.
    Uses bitmask simulation for speed.
    """
    if not network:
        return N <= 1
    
    # Copy initial state
    state = list(INITIAL_STATE)
    
    # Apply comparators
    for i, j in network:
        # min goes to i, max goes to j
        # new_i = state[i] & state[j]
        # new_j = state[i] | state[j]
        si = state[i]
        sj = state[j]
        state[i] = si & sj
        state[j] = si | sj
        
    # Check if state is sorted: S[0] subseteq S[1] subseteq ... subseteq S[N-1]
    # This is equivalent to checking S[i] & ~S[i+1] == 0 for all i
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
            
    return True

def _construct_odd_even_transposition():
    """Standard odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    """Construct a sorting network with as few comparators as possible."""
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Start with Odd-Even Transposition
    network = _construct_odd_even_transposition()
    best_network = list(network)
    best_len = len(network)
    
    # Local Search / Simulated Annealing
    # Strategy: Greedy deletion, with perturbation if stuck
    
    current = list(network)
    current_len = len(current)
    
    while time.time() < deadline:
        # Phase 1: Greedy Deletion
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each comparator in random order
            indices = list(range(len(current)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline:
                    break
                
                # Try removing this comparator
                test_network = current[:idx] + current[idx+1:]
                
                if _is_valid(test_network):
                    current = test_network
                    current_len -= 1
                    improved = True
                    break # Restart the loop with the new list
            
            # If we removed one, the 'improved' loop continues
            # If we didn't remove any, we are stuck in this configuration
        
        # Update best
        if current_len < best_len:
            best_network = list(current)
            best_len = current_len
            
        # Check time
        if time.time() > deadline - 5.0:
            break
            
        # Phase 2: Perturbation to escape local minima
        # Add some random comparators to allow different deletion paths
        # We add a few, then try to delete them and others
        
        # How many to add? Maybe 2-5
        num_adds = random.randint(2, 5)
        for _ in range(num_adds):
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            current.append((i, j))
            
        # Optionally shuffle the network?
        # Shuffling changes the order, which might allow different deletions
        if random.random() < 0.5:
            random.shuffle(current)
            
        # Now, go back to Phase 1 (Greedy Deletion)
        # The while loop will handle it

    return best_network
