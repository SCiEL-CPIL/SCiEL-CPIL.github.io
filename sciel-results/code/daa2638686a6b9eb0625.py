import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety

    N = 256

    # Perfect squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1

    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)

    n = len(valid_nodes)
    all_nodes_mask = (1 << n) - 1

    # Build conflict masks
    conflicts = [0] * n
    for i_idx in range(n):
        v = valid_nodes[i_idx]
        mask = 0
        for j_idx in range(i_idx + 1, n):
            w = valid_nodes[j_idx]
            if v + w in squares:
                mask |= (1 << j_idx)
                conflicts[j_idx] |= (1 << i_idx)
        conflicts[i_idx] = mask

    def popcount(x):
        return x.bit_count()

    def bits_to_list(mask):
        res = []
        i = 0
        while mask:
            if mask & 1:
                res.append(i)
            mask >>= 1
            i += 1
        return res

    def greedy_mis(order):
        selected = 0
        for idx in order:
            if not (conflicts[idx] & selected):
                selected |= (1 << idx)
        return selected

    def local_search(mask):
        """
        Fast local search: Add any available node.
        """
        improved = True
        while improved:
            improved = False
            # Calculate conflicts with current set
            conf_with_set = 0
            m = mask
            i = 0
            while m:
                if m & 1:
                    conf_with_set |= conflicts[i]
                m >>= 1
                i += 1
            
            available = all_nodes_mask & ~mask & ~conf_with_set
            if available:
                # Pick the first available node (lowest index)
                # This is deterministic and fast
                idx = (available & -available).bit_length() - 1
                mask |= (1 << idx)
                improved = True
        return mask

    def simulated_annealing(initial_mask, time_limit=5.0):
        """
        Simulated Annealing for MIS.
        """
        current = initial_mask
        best = initial_mask
        best_size = popcount(best)
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        start_sa = time.time()
        
        while (time.time() - start_sa) < time_limit:
            if T < min_T:
                T = 1.0 # Restart temperature if too low? Or break?
                # Let's just break or restart T
                # Restarting T allows continued exploration
                pass
            
            # Pick random node not in current
            not_in_set = all_nodes_mask & ~current
            if not_in_set == 0:
                break
            
            # Pick random bit from not_in_set
            # To do this efficiently without generating a list:
            # Pick a random number of shifts? No.
            # Just pick random index and check.
            u = random.randint(0, n - 1)
            if current & (1 << u):
                continue
            
            C = conflicts[u] & current
            C_size = popcount(C)
            new_mask = (current & ~C) | (1 << u)
            
            delta = 1 - C_size
            
            if delta > 0:
                current = new_mask
                if popcount(current) > best_size:
                    best = current
                    best_size = popcount(best)
            else:
                # Accept with probability exp(delta / T)
                # If delta is very negative, exp is 0.
                # If T is small, exp is 0.
                if random.random() < math.exp(delta / T):
                    current = new_mask
                    if popcount(current) > best_size:
                        best = current
                        best_size = popcount(best)
            
            T *= cooling

        return best

    best_mask = 0
    best_size = 0

    random.seed(42)

    # 1. Deterministic Greedy (Low Degree First)
    degrees = [popcount(conflicts[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x])
    mask = greedy_mis(order)
    mask = local_search(mask)
    if popcount(mask) > best_size:
        best_mask = mask
        best_size = popcount(mask)

    # 2. Deterministic Greedy (High Degree First)
    order = sorted(range(n), key=lambda x: -degrees[x])
    mask = greedy_mis(order)
    mask = local_search(mask)
    if popcount(mask) > best_size:
        best_mask = mask
        best_size = popcount(mask)

    # 3. Random Restarts with Greedy + Local Search
    # Use 20 seconds for this phase
    restart_deadline = start_time + 20
    
    while (time.time() - start_time) < restart_deadline:
        order = list(range(n))
        random.shuffle(order)
        mask = greedy_mis(order)
        mask = local_search(mask)
        if popcount(mask) > best_size:
            best_mask = mask
            best_size = popcount(mask)

    # 4. Simulated Annealing on the best found so far
    # Use remaining time minus safety margin
    remaining_time = deadline - time.time() - 2
    if remaining_time > 5:
        # Run SA for a chunk of time, then restart with best found
        # We can run SA multiple times with different seeds/initializations
        # But SA from a good start is usually best.
        
        # Let's run SA for 30 seconds
        sa_time = min(30, remaining_time)
        best_mask = simulated_annealing(best_mask, time_limit=sa_time)
        best_size = popcount(best_mask)
        
        # If we have more time, run SA again from the new best?
        # Or run SA from a different random start?
        # Let's run SA from the best found, with a fresh T.
        if (time.time() - start_time) < deadline - 5:
            remaining_time = deadline - time.time() - 2
            if remaining_time > 5:
                 sa_time = min(remaining_time, 20)
                 # Use a different random seed for diversity?
                 # The state is deterministic given the seed.
                 # Let's just run it again.
                 best_mask = simulated_annealing(best_mask, time_limit=sa_time)
                 best_size = popcount(best_mask)

    # 5. Final Local Search on Best
    if (time.time() - start_time) < deadline - 2:
        best_mask = local_search(best_mask)
        if popcount(best_mask) > best_size:
            best_size = popcount(best_mask)

    # Convert back to values
    result_indices = bits_to_list(best_mask)
    result = [valid_nodes[i] for i in result_indices]
    result.sort()

    if not result:
        result = [1]

    return result
