"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 9
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    def is_valid_extension(v, path, used):
        if v in used:
            return False
        for i in range(len(path) - 1):
            if (v ^ path[i]).bit_count() == 1:
                return False
        return True
    
    def greedy_extend(path, used, bit_order):
        while True:
            if time.time() > deadline:
                return path
            cur = path[-1]
            nxt = None
            for b in bit_order:
                v = cur ^ (1 << b)
                if is_valid_extension(v, path, used):
                    nxt = v
                    break
            if nxt is None:
                return path
            path.append(nxt)
            used.add(nxt)
    
    # Try multiple random bit orderings and keep the best
    best_path = []
    
    # First try the original greedy (bit order 0,1,2,...,8)
    path = [0]
    used = {0}
    bit_order = list(range(N))
    path = greedy_extend(path, used, bit_order)
    best_path = path
    
    # Try random bit orderings
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random_bit_order = list(range(N))
        random.shuffle(random_bit_order)
        
        path = [0]
        used = {0}
        path = greedy_extend(path, used, random_bit_order)
        
        if len(path) > len(best_path):
            best_path = path
    
    return best_path
# EVOLVE-BLOCK-END
