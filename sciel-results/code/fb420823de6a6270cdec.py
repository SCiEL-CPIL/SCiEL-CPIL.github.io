import time
import random

N = 12
SIZE = 1 << N

def compute_reset_length(trans_a, trans_b):
    """
    Computes the exact shortest reset word length for a deterministic binary automaton.
    Returns -1 if not synchronizing.
    Uses BFS on the subset automaton represented by bitmasks.
    Optimized with precomputed subset transitions.
    """
    # Precompute transitions for all subsets
    # We only need to compute these once per evaluation if we cache them, 
    # but since trans_a and trans_b change, we recompute.
    # To speed up, we can compute on the fly in BFS, but precomputation 
    # is O(2^N * N) which is ~50k ops, negligible.
    
    trans_a_sub = [0] * SIZE
    trans_b_sub = [0] * SIZE
    
    # Optimization: Use list unpacking or simple loops
    # Since N=12, SIZE=4096.
    # We can precompute the "image" of each state for a and b.
    # img_a[i] = 1 << trans_a[i]
    # img_b[i] = 1 << trans_b[i]
    
    img_a = [1 << x for x in trans_a]
    img_b = [1 << x for x in trans_b]
    
    for subset in range(1, SIZE):
        # Instead of iterating bits of subset every time, 
        # we can use a DP approach or just iterate bits.
        # Iterating bits of a 12-bit integer is very fast.
        sa = 0
        sb = 0
        temp = subset
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            sa |= img_a[i]
            sb |= img_b[i]
            temp ^= lsb
        trans_a_sub[subset] = sa
        trans_b_sub[subset] = sb

    start = (1 << N) - 1
    if start == 1:
        return 0
        
    visited = [False] * SIZE
    queue = [start]
    visited[start] = True
    dist = 0
    
    while queue:
        new_queue = []
        for subset in queue:
            # Check if singleton
            if subset & (subset - 1) == 0:
                return dist
            
            next_a = trans_a_sub[subset]
            if not visited[next_a]:
                visited[next_a] = True
                new_queue.append(next_a)
            
            next_b = trans_b_sub[subset]
            if not visited[next_b]:
                visited[next_b] = True
                new_queue.append(next_b)
        
        queue = new_queue
        dist += 1
        
    return -1

def construct():
    deadline = time.time() + 55  # Safety margin for 90s limit, leaving 35s for evaluator overhead
    
    # Initial state: A known high-threshold automaton (Pinwheel)
    # a: cycle (0->1->...->11->0)
    # b: (0->0, 1->0, 2->1, ..., 11->10)
    # This is a standard Pinwheel variant.
    init_a = [(i + 1) % N for i in range(N)]
    init_b = [0] * N
    for i in range(1, N):
        init_b[i] = i - 1
        
    current_a = init_a[:]
    current_b = init_b[:]
    current_len = compute_reset_length(current_a, current_b)
    
    best_a = current_a[:]
    best_b = current_b[:]
    best_len = current_len
    
    # Simulated Annealing / Hill Climbing
    # We will mutate one transition at a time.
    # If the new automaton is synchronizing and has a longer reset word, we accept.
    # If it's shorter, we accept with probability exp(-delta / T) to escape local maxima.
    
    T = 10.0  # Initial temperature
    cooling = 0.995
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Choose a letter and a state to mutate
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        
        old_val = current_a[state] if letter == 0 else current_b[state]
        
        # Choose a new destination
        # Avoid picking the same value to ensure a change
        new_val = random.randint(0, N - 1)
        while new_val == old_val:
            new_val = random.randint(0, N - 1)
            
        # Apply mutation
        if letter == 0:
            current_a[state] = new_val
        else:
            current_b[state] = new_val
            
        new_len = compute_reset_length(current_a, current_b)
        
        # If not synchronizing, revert immediately (or treat as -inf)
        if new_len == -1:
            if letter == 0:
                current_a[state] = old_val
            else:
                current_b[state] = old_val
            continue
            
        # Acceptance criterion
        if new_len > current_len:
            # Better, always accept
            current_len = new_len
            if new_len > best_len:
                best_len = new_len
                best_a = current_a[:]
                best_b = current_b[:]
        else:
            # Worse or equal
            # Accept with probability based on temperature
            # delta is negative
            delta = new_len - current_len
            # If delta is 0, accept with 0.5 chance to explore plateau
            if delta == 0:
                if random.random() < 0.5:
                    current_len = new_len
            else:
                # delta < 0
                prob = 2.718281828 ** (delta / T)
                if random.random() < prob:
                    current_len = new_len
                else:
                    # Revert
                    if letter == 0:
                        current_a[state] = old_val
                    else:
                        current_b[state] = old_val
        
        # Cool down
        T *= cooling
        
        # Restart mechanism: if we haven't improved best for a long time, 
        # we might restart from best or randomize temperature.
        # For now, simple cooling is usually sufficient for N=12.
        
    return [best_a, best_b]

# Run a quick self-test to ensure it works and is fast
if __name__ == "__main__":
    start = time.time()
    a, b = construct()
    end = time.time()
    l = compute_reset_length(a, b)
    print(f"Time: {end-start:.2f}s, Length: {l}")
    print(f"A: {a}")
    print(f"B: {b}")
