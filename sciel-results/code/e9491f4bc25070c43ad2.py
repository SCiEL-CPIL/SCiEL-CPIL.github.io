import time
import random
import sys

N = 12
SIZE = 1 << N
DEADLINE = time.time() + 55  # 55s safety margin for 60s limit

def compute_reset_length(trans_a, trans_b):
    """
    Computes the exact shortest reset word length for a deterministic binary automaton.
    Returns -1 if not synchronizing.
    Uses BFS on the subset automaton represented by bitmasks.
    """
    # Precompute transitions for all subsets
    # This is O(2^N * N) which is 12 * 4096 = 49152 ops, very fast.
    trans_a_sub = [0] * SIZE
    trans_b_sub = [0] * SIZE
    
    for subset in range(1, SIZE):
        sa = 0
        sb = 0
        temp = subset
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            sa |= (1 << trans_a[i])
            sb |= (1 << trans_b[i])
            temp ^= lsb
        trans_a_sub[subset] = sa
        trans_b_sub[subset] = sb

    start = (1 << N) - 1
    if start == 1:
        return 0
        
    visited = [False] * SIZE
    # Use a list as a queue, but process level by level
    # We can optimize by using a deque, but list slicing or index tracking is fine for 4096 items
    queue = [start]
    visited[start] = True
    dist = 0
    current_level_end = 1
    
    while queue:
        # Check if any state in the current level is a singleton
        # Actually, we check when we pop or when we generate.
        # Standard BFS: check distance when we reach a singleton.
        
        # Optimization: Check singletons at the start of the level?
        # No, we check when we encounter them.
        
        # Process current level
        next_queue = []
        
        for subset in queue:
            # Check if singleton
            if subset & (subset - 1) == 0:
                return dist
            
            next_a = trans_a_sub[subset]
            if not visited[next_a]:
                visited[next_a] = True
                next_queue.append(next_a)
            
            next_b = trans_b_sub[subset]
            if not visited[next_b]:
                visited[next_b] = True
                next_queue.append(next_b)
        
        queue = next_queue
        dist += 1
        
    return -1

def construct():
    start_time = time.time()
    
    # 1. Initialize with the known best "Pinwheel" structure
    # The Pinwheel P_n usually has:
    # a: (0 1 ... n-1)
    # b: (0 0 1 ... n-2) -> b[i] = i-1 for i>0, b[0]=0? 
    # Actually, the standard high-threshold pinwheel for N=12 is often:
    # a[i] = (i+1) % N
    # b[i] = (i-1) % N for i != 0, b[0] = 0? No, that's just a cycle with a break.
    # Let's use the specific construction that yields 112.
    # A common optimal for small N is the "Pinwheel" where:
    # a = (0 1 2 ... 11)
    # b = (0 0 1 2 ... 10) is NOT it.
    # The "Pinwheel" automaton P_n:
    #   x: (0 1 ... n-1)
    #   y: (0 0 1 ... n-2) is not synchronizing?
    #   Actually, let's use the structure from the prompt's current best:
    #   a: i -> (i+1) % N
    #   b: i -> (i-1) % N for i != k, b[k] = t.
    #   The specific k,t that gives 112 needs to be found or we start with a random valid one.
    
    # Let's start with a known good candidate. 
    # For N=12, the max is 112. 
    # One such automaton is:
    # a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
    # b = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ? 
    # Let's just generate a few random valid ones and pick the best as seed.
    
    best_a = None
    best_b = None
    best_len = -1
    
    # Generate initial population of 50 random "Pinwheel-like" candidates
    # a is a random cycle
    # b is a random function with low rank (to ensure synchronizability and high threshold)
    
    for _ in range(50):
        # Random cycle for a
        perm = list(range(N))
        random.shuffle(perm)
        a = [0] * N
        # Construct cycle from perm
        # perm[0] -> perm[1] ... perm[-1] -> perm[0]
        for i in range(N):
            a[perm[i]] = perm[(i+1)%N]
        
        # Random b, biased towards "almost cycle" or "almost constant"
        # To get high thresholds, b should not be too "mixing".
        # Let's try b as a random permutation or random function.
        # Random function is 12^12 possibilities, too many to sample blindly for high score.
        # Better: Start with a cycle for b, and break it.
        perm_b = list(range(N))
        random.shuffle(perm_b)
        b = [0] * N
        for i in range(N):
            b[perm_b[i]] = perm_b[(i+1)%N]
        
        # Mutate b slightly
        for _ in range(random.randint(0, 3)):
            idx = random.randint(0, N-1)
            b[idx] = random.randint(0, N-1)
            
        l = compute_reset_length(a, b)
        if l > best_len:
            best_len = l
            best_a = a[:]
            best_b = b[:]
            
    if best_a is None:
        # Fallback
        best_a = [(i+1)%N for i in range(N)]
        best_b = [0]*N
        for i in range(1, N):
            best_b[i] = i-1
        best_len = compute_reset_length(best_a, best_b)
        
    # 2. Local Search / Hill Climbing with Random Restarts
    # We will perform a series of "walks" starting from the best known state.
    # At each step, we mutate one transition in a or b, and evaluate.
    # If it improves, we keep it. If it stalls, we restart from the best global.
    
    current_a = best_a[:]
    current_b = best_b[:]
    current_len = best_len
    
    stagnation_count = 0
    max_stagnation = 200
    
    while time.time() < DEADLINE:
        # Mutate
        # Choose which letter to mutate (50/50)
        if random.random() < 0.5:
            idx = random.randint(0, N-1)
            old_val = current_a[idx]
            new_val = random.randint(0, N-1)
            if new_val == old_val:
                continue
            current_a[idx] = new_val
            new_a = current_a
            new_b = current_b
        else:
            idx = random.randint(0, N-1)
            old_val = current_b[idx]
            new_val = random.randint(0, N-1)
            if new_val == old_val:
                continue
            current_b[idx] = new_val
            new_a = current_a
            new_b = current_b
            
        new_len = compute_reset_length(new_a, new_b)
        
        if new_len > current_len:
            # Accept
            current_len = new_len
            if new_len > best_len:
                best_len = new_len
                best_a = new_a[:]
                best_b = new_b[:]
                stagnation_count = 0
            else:
                stagnation_count += 1
        else:
            # Reject, revert mutation
            if random.random() < 0.5:
                current_a[idx] = old_val
            else:
                current_b[idx] = old_val
            stagnation_count += 1
            
        # If we've stagnated for a while, restart from best global
        if stagnation_count > max_stagnation:
            current_a = best_a[:]
            current_b = best_b[:]
            current_len = best_len
            stagnation_count = 0
            
            # Optional: Add a "jump" by mutating the best solution significantly before restarting
            # But for now, simple restart is robust.
            
            # If we have time, try a "random restart" from scratch occasionally
            if random.random() < 0.1:
                # Generate a new random candidate
                perm = list(range(N))
                random.shuffle(perm)
                a = [0] * N
                for i in range(N):
                    a[perm[i]] = perm[(i+1)%N]
                
                perm_b = list(range(N))
                random.shuffle(perm_b)
                b = [0] * N
                for i in range(N):
                    b[perm_b[i]] = perm_b[(i+1)%N]
                
                for _ in range(random.randint(1, 5)):
                    idx = random.randint(0, N-1)
                    b[idx] = random.randint(0, N-1)
                    
                l = compute_reset_length(a, b)
                if l > best_len:
                    best_len = l
                    best_a = a[:]
                    best_b = b[:]
                    current_a = a[:]
                    current_b = b[:]
                    current_len = l
                else:
                    # Keep the best global as the restart point
                    current_a = best_a[:]
                    current_b = best_b[:]
                    current_len = best_len

    return [best_a, best_b]

if __name__ == "__main__":
    # For local testing
    a, b = construct()
    print(f"Best length: {compute_reset_length(a, b)}")
    print(f"A: {a}")
    print(f"B: {b}")
