import random
import time
import collections

N = 12

def calc_threshold(trans):
    """
    Calculate the exact shortest reset word length using BFS on subsets.
    trans: list of two lists, trans[0] and trans[1], each of length N.
    Returns -1 if not synchronizing, else the length.
    """
    # State is a frozenset of states
    start = frozenset(range(N))
    if len(start) == 1:
        return 0
        
    # BFS
    visited = {start: 0}
    queue = collections.deque([start])
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        # Apply letter 0
        next_subset = frozenset(trans[0][s] for s in current)
        if len(next_subset) == 1:
            return dist + 1
        if next_subset not in visited:
            visited[next_subset] = dist + 1
            queue.append(next_subset)
            
        # Apply letter 1
        next_subset = frozenset(trans[1][s] for s in current)
        if len(next_subset) == 1:
            return dist + 1
        if next_subset not in visited:
            visited[next_subset] = dist + 1
            queue.append(next_subset)
            
    return -1

def is_synchronizing(trans):
    return calc_threshold(trans) != -1

def construct():
    # Explicit deadline guard: The evaluator kills at 60s (or 90s, but task says 60s).
    # We must return before 60s. Let's target 55s to be safe.
    deadline = time.time() + 55.0
    
    best_trans = None
    best_score = -1
    
    # Generate initial random automaton
    # Instead of a slow random loop, just pick a few random starts or a known good structure.
    # A "slide" with a collapse is a good start.
    # Let's try a few random starts quickly.
    
    for _ in range(10):
        if time.time() > deadline - 5.0: # Leave 5s for optimization
            break
        trans = [[random.randint(0, N-1) for _ in range(N)] for _ in range(2)]
        score = calc_threshold(trans)
        if score > best_score:
            best_score = score
            best_trans = [row[:] for row in trans]
            
    if best_trans is None:
        # Fallback to a simple synchronizing automaton
        # a: i -> (i+1) % N, b: i -> 0
        return [[(i + 1) % N for i in range(N)], [0] * N]

    # Hill Climbing
    current_trans = [row[:] for row in best_trans]
    current_score = best_score
    
    # Optimization Loop
    # We need to be faster. calc_threshold is the bottleneck.
    # For N=12, BFS is fast, but Python overhead adds up.
    # We will run as many iterations as possible before the deadline.
    
    while time.time() < deadline:
        # Random mutation
        letter = random.randint(0, 1)
        state = random.randint(0, N-1)
        old_val = current_trans[letter][state]
        new_val = random.randint(0, N-1)
        
        if new_val == old_val:
            continue
            
        current_trans[letter][state] = new_val
        
        new_score = calc_threshold(current_trans)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_trans = [row[:] for row in current_trans]
        else:
            # Revert
            current_trans[letter][state] = old_val
            
    return best_trans
