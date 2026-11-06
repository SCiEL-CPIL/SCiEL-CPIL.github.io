import time
import random
import numpy as np
from itertools import combinations

N = 17
MAX_TIME = 55.0

# Precompute valid sorted states (0^k 1^(N-k))
# A state is valid if it is of the form 0...01...1
# Values: 0, 2^N-2^(N-1), ..., 2^N-1
_VALID_MASK = np.zeros(1 << N, dtype=bool)
for k in range(N + 1):
    if k == 0:
        val = 0
    else:
        val = (1 << N) - (1 << (N - k))
    _VALID_MASK[val] = True

# Precompute comparator effects as bitmasks for fast lookup?
# Actually, we will use a set of "unsorted" states to guide the search.

def seed_network():
    """Odd-even transposition network (136 comparators)."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def verify_network(net):
    """
    Fast verification using numpy.
    """
    if not net:
        return N <= 1
    
    states = np.arange(1 << N, dtype=np.uint32)
    for i, j in net:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        mask = (bi == 1) & (bj == 0)
        if np.any(mask):
            states[mask] ^= (1 << i) | (1 << j)
    
    return bool(np.all(_VALID_MASK[states]))

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Initial best is the seed
    best_net = seed_network()
    best_count = len(best_net)
    
    # We will use a local search that is more aggressive about removing comparators.
    # Strategy: 
    # 1. Start with seed.
    # 2. Repeatedly try to remove a random comparator. If valid, keep it.
    # 3. If stuck, try to swap two comparators to "shake up" the structure.
    # 4. If stuck, try to replace a comparator with a "useful" one (one that sorts more states).
    
    current = list(best_net)
    
    # Helper to check if a comparator is "useful" for a given set of unsorted states
    # This is expensive to compute for all states every time, so we'll rely on the global verify.
    
    def minimize(net, time_limit):
        current = list(net)
        progress = True
        # Use a randomized order to avoid getting stuck in specific local minima
        while progress and time.time() < time_limit:
            progress = False
            indices = list(range(len(current)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > time_limit:
                    break
                # Try removing comparator at idx
                new_net = current[:idx] + current[idx+1:]
                if verify_network(new_net):
                    current = new_net
                    progress = True
                    break # Restart scan with new list
        return current

    # Phase 1: Aggressive Minimization
    current = minimize(current, deadline)
    if len(current) < best_count:
        best_net = current
        best_count = len(current)
        print(f"Phase 1: Reduced to {best_count}", flush=True)
    
    # Phase 2: Simulated Annealing / Hill Climbing with Restart
    # We will keep a pool of good networks and mutate them.
    
    # To speed up verification, we can pre-compute the effect of a comparator on a state?
    # No, numpy is fast enough for 131k states.
    
    # Let's try a different mutation: "Depth-based" insertion.
    # Instead of random position, insert at a specific depth.
    
    while time.time() < deadline:
        # Pick a random move
        move_type = random.random()
        
        if len(current) < 2:
            break
            
        if move_type < 0.5:
            # Try to remove a random comparator (greedy improvement)
            idx = random.randint(0, len(current) - 1)
            new_net = current[:idx] + current[idx+1:]
            if verify_network(new_net):
                current = new_net
                if len(current) < best_count:
                    best_net = current
                    best_count = len(current)
                    print(f"Phase 2 (remove): Improved to {best_count}", flush=True)
                # Try to minimize further immediately
                if time.time() < deadline - 1:
                    current = minimize(current, deadline)
                    if len(current) < best_count:
                        best_net = current
                        best_count = len(current)
        elif move_type < 0.8:
            # Swap two comparators
            i, j = random.sample(range(len(current)), 2)
            new_net = list(current)
            new_net[i], new_net[j] = new_net[j], new_net[i]
            if verify_network(new_net):
                current = new_net
                # Minimize
                if time.time() < deadline - 1:
                    current = minimize(current, deadline)
                    if len(current) < best_count:
                        best_net = current
                        best_count = len(current)
        else:
            # Replace a random comparator with a random new one
            idx = random.randint(0, len(current) - 1)
            new_i = random.randint(0, N - 1)
            new_j = random.randint(0, N - 1)
            if new_i == new_j:
                continue
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            
            new_net = list(current)
            new_net[idx] = (new_i, new_j)
            
            if verify_network(new_net):
                current = new_net
                if time.time() < deadline - 1:
                    current = minimize(current, deadline)
                    if len(current) < best_count:
                        best_net = current
                        best_count = len(current)
                        
    # Final safety check
    if not verify_network(best_net):
        best_net = seed_network()
        
    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Result size: {len(net)}")
    print(f"Valid: {verify_network(net)}")
