import time
import random
import math

N = 17
TOTAL_INPUTS = 1 << N

def init_masks():
    masks = [0] * N
    for i in range(N):
        bit = 1 << i
        # Pattern: 0..0 1..1 0..0 1..1 ...
        # Block size 2^i, half 0, half 1
        # Actually, we can just loop k
        # Or use a trick:
        # m[i] is a bitmask of length 2^N
        # Bit k is 1 if k % (2^(i+1)) >= 2^i
        pass
    # Simpler:
    for k in range(TOTAL_INPUTS):
        for i in range(N):
            if (k >> i) & 1:
                masks[i] |= (1 << k)
    return masks

INITIAL_MASKS = init_masks()

def simulate(comparators):
    masks = INITIAL_MASKS[:]
    for i, j in comparators:
        mi = masks[i]
        mj = masks[j]
        # min goes to i, max goes to j
        # min is mi & mj, max is mi | mj
        masks[i] = mi & mj
        masks[j] = mi | mj
    return masks

def is_correct(masks):
    for i in range(1, N):
        if masks[i] & ~masks[i-1]:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Initial network: Odd-even transposition
    # N=17, N phases
    net = [(i, i + 1) for phase in range(N)
           for i in range(phase % 2, N - 1, 2)]
    
    # Check initial
    masks = simulate(net)
    if not is_correct(masks):
        # Should not happen for odd-even
        return net

    best_net = net[:]
    best_len = len(net)
    current_net = net[:]
    current_len = len(net)
    
    T = 10.0  # Temperature
    T_min = 0.1
    cooling = 0.99
    
    while time.time() < deadline:
        # Pick a move
        move_type = random.random()
        if move_type < 0.5:
            # Remove
            if current_len == 0:
                continue
            idx = random.randrange(current_len)
            new_net = current_net[:idx] + current_net[idx+1:]
        elif move_type < 0.8:
            # Add
            i = random.randrange(N)
            j = random.randrange(N)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            new_net = current_net + [(i, j)]
        elif move_type < 0.9:
            # Swap
            if current_len < 2:
                continue
            i1, i2 = random.sample(range(current_len), 2)
            new_net = current_net[:]
            new_net[i1], new_net[i2] = new_net[i2], new_net[i1]
        else:
            # Move
            if current_len < 2:
                continue
            i1 = random.randrange(current_len)
            i2 = random.randrange(current_len)
            if i1 == i2:
                continue
            new_net = current_net[:]
            comp = new_net.pop(i1)
            new_net.insert(i2, comp)
        
        # Evaluate
        new_len = len(new_net)
        masks = simulate(new_net)
        if not is_correct(masks):
            # Reject incorrect
            continue
            
        # Acceptance
        delta = new_len - current_len
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_net = new_net
            current_len = new_len
            if current_len < best_len:
                best_net = current_net[:]
                best_len = current_len
                # Reset temperature? Or keep cooling
                T = max(T_min, T * 1.1) # Reheat slightly on improvement
            else:
                T = max(T_min, T * cooling)
        else:
            T = max(T_min, T * cooling)
            
    return best_net
