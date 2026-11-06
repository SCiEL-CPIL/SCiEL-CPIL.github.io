import time
import numpy as np
import random

N = 18
TIME_LIMIT = 58

def _odd_even_transposition():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    num_blocks = 2**N // 64
    states = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    for k in range(2**N):
        block_idx = k >> 6
        bit_idx = k & 63
        mask = 1 << bit_idx
        for w in range(N):
            if (k >> w) & 1:
                states[w][block_idx] |= mask

    def _fast_verify(comparators):
        current = [s.copy() for s in states]
        for i, j in comparators:
            a = current[i]
            b = current[j]
            current[i] = a & b
            current[j] = a | b
        for w in range(N - 1):
            if np.any(current[w] & ~current[w + 1]):
                return False
        return True

    def _normalize(net):
        changed = True
        while changed:
            changed = False
            for i in range(len(net) - 1):
                a1, b1 = net[i]
                a2, b2 = net[i+1]
                if a1 != a2 and a1 != b2 and b1 != a2 and b1 != b2:
                    net[i], net[i+1] = net[i+1], net[i]
                    changed = True
        return net

    def _greedy_reduce(net):
        changed = True
        while changed:
            changed = False
            idx = 0
            while idx < len(net):
                if time.time() > deadline:
                    return net
                trial = net[:idx] + net[idx+1:]
                if _fast_verify(trial):
                    net = trial
                    changed = True
                else:
                    idx += 1
        return net

    current_network = _odd_even_transposition()
    current_network = _normalize(current_network)
    current_network = _greedy_reduce(current_network)
    
    best_network = list(current_network)
    best_len = len(current_network)
    
    T = 5.0
    cooling = 0.999
    min_T = 0.01
    steps_since_improve = 0
    reduce_interval = 50
    
    P_DELETE = 0.4
    P_INSERT = 0.2
    P_SWAP = 0.2
    P_MUTATE = 0.2
    
    while time.time() < deadline - 2:
        if T < min_T:
            T = min_T
            
        move_type = random.random()
        new_network = list(current_network)
        
        if move_type < P_DELETE:
            if len(new_network) == 0: continue
            idx = random.randint(0, len(new_network) - 1)
            new_network.pop(idx)
        elif move_type < P_DELETE + P_INSERT:
            idx = random.randint(0, len(new_network))
            a, b = random.sample(range(N), 2)
            if a > b: a, b = b, a
            new_network.insert(idx, (a, b))
        elif move_type < P_DELETE + P_INSERT + P_SWAP:
            if len(new_network) < 2: continue
            i, j = random.sample(range(len(new_network)), 2)
            new_network[i], new_network[j] = new_network[j], new_network[i]
        else:
            if len(new_network) == 0: continue
            idx = random.randint(0, len(new_network) - 1)
            a, b = random.sample(range(N), 2)
            if a > b: a, b = b, a
            new_network[idx] = (a, b)
            
        new_network = _normalize(new_network)
        
        if not _fast_verify(new_network):
            continue
            
        new_len = len(new_network)
        old_len = len(current_network)
        delta = new_len - old_len
        
        if delta < 0:
            # Improvement
            current_network = new_network
            current_network = _greedy_reduce(current_network)
            if len(current_network) < best_len:
                best_network = list(current_network)
                best_len = len(current_network)
                steps_since_improve = 0
            else:
                steps_since_improve += 1
        else:
            # Same or worse
            if random.random() < np.exp(-delta / T):
                current_network = new_network
                steps_since_improve += 1
            else:
                steps_since_improve += 1
                
        T *= cooling
        
        # Periodic reduction to keep network tight
        if steps_since_improve > reduce_interval:
            current_network = _greedy_reduce(current_network)
            if len(current_network) < best_len:
                best_network = list(current_network)
                best_len = len(current_network)
            steps_since_improve = 0

    return best_network
