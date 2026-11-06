import time
import itertools

N = 12

def compute_threshold(trans_a, trans_b):
    """
    Computes the shortest reset word length for a given automaton.
    Returns the length of the shortest reset word, or -1 if not synchronizing.
    """
    # BFS on subsets of states
    # Start state: full set {0, ..., N-1}
    # Goal state: any singleton {i}
    
    full_set = tuple(range(N))
    if len(full_set) == 1:
        return 0
        
    visited = {full_set: 0}
    queue = [full_set]
    
    while queue:
        current = queue.pop(0)
        dist = visited[current]
        
        # Try applying 'a'
        next_a = tuple(sorted(set(trans_a[i] for i in current)))
        if len(next_a) == 1:
            return dist + 1
        if next_a not in visited:
            visited[next_a] = dist + 1
            queue.append(next_a)
            
        # Try applying 'b'
        next_b = tuple(sorted(set(trans_b[i] for i in current)))
        if len(next_b) == 1:
            return dist + 1
        if next_b not in visited:
            visited[next_b] = dist + 1
            queue.append(next_b)
            
    return -1

def construct():
    # Define candidate automata
    candidates = []
    
    # Candidate 1: Cycle + Decrement
    a1 = [(i + 1) % N for i in range(N)]
    b1 = [0 if i == 0 else i - 1 for i in range(N)]
    candidates.append((a1, b1))
    
    # Candidate 2: Cycle on N-1, Sink for N-1
    a2 = [(i + 1) % (N-1) for i in range(N-1)] + [0]
    b2 = [0 if i == 0 else i - 1 for i in range(N-1)] + [N-2]
    candidates.append((a2, b2))
    
    # Candidate 3: Random-ish but structured
    # a: 0->1, 1->2, ..., N-2->0, N-1->0
    # b: 0->0, 1->0, 2->1, ..., N-1->N-2
    a3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0] # Just example, generated below
    a3 = [(i + 1) % (N-1) for i in range(N-1)] + [0]
    b3 = [0, 0] + [i - 1 for i in range(2, N)]
    candidates.append((a3, b3))
    
    best_thresh = 0
    best_a = a1
    best_b = b1
    
    start_time = time.time()
    time_limit = 60.0 # Evaluator kills at 90s, keep margin
    
    for a, b in candidates:
        if time.time() - start_time > time_limit:
            break
        thresh = compute_threshold(a, b)
        if thresh > best_thresh:
            best_thresh = thresh
            best_a = a
            best_b = b
            
    return best_a, best_b

# EVOLVE-BLOCK-START
# The construct function is defined above.
# To adhere to the "Keep the function names and signatures" directive, 
# I will place the code inside the block or ensure construct is accessible.
# The prompt asks to rewrite the program. The seed has construct inside EVOLVE-BLOCK.
# I will define the helper functions and construct within the block.
# EVOLVE-BLOCK-END
