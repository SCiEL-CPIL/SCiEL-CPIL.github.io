"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def _compute_reset_length(a, b):
    """Compute shortest reset word length via BFS on subsets. Returns -1 if not synchronizing."""
    start = tuple(range(N))
    if len(set(a)) == 1 or len(set(b)) == 1:
        return 0
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        subset, dist = queue.popleft()
        if len(subset) == 1:
            return dist
        # apply a
        next_a = tuple(sorted(set(a[i] for i in subset)))
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, dist + 1))
        # apply b
        next_b = tuple(sorted(set(b[i] for i in subset)))
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, dist + 1))
    return -1

def _is_synchronizing(a, b):
    return _compute_reset_length(a, b) >= 0

def _generate_cycle_with_random_map():
    """Generate an automaton where one letter is a cycle and the other is random."""
    a = [(i + 1) % N for i in range(N)]
    b = [random.randint(0, N - 1) for _ in range(N)]
    return a, b

def _generate_random_automaton():
    """Generate a fully random automaton."""
    a = [random.randint(0, N - 1) for _ in range(N)]
    b = [random.randint(0, N - 1) for _ in range(N)]
    return a, b

def _generate_parking_automaton():
    """Generate a parking automaton: one letter is identity, other is a random permutation with one fixed point."""
    a = list(range(N))
    b = list(range(N))
    # Create a random permutation
    perm = list(range(N))
    random.shuffle(perm)
    # Ensure at least one state maps to 0 to help synchronization
    # Actually, let's make b a random function with good mixing properties
    b = [random.randint(0, N - 1) for _ in range(N)]
    return a, b

def _generate_star_automaton():
    """Star automaton: one letter maps everything to 0, other is a cycle."""
    a = [0] * N
    b = [(i + 1) % N for i in range(N)]
    return a, b

def _generate_binary_tree_automaton():
    """Generate an automaton based on binary tree structure."""
    # States 0-11, create a tree-like structure
    a = [0] * N
    b = [0] * N
    # Create a path with branches
    for i in range(N):
        a[i] = (i + 1) % N
        b[i] = i // 2 if i > 0 else 0
    return a, b

def _generate_known_good_automaton():
    """Return a known synchronizing automaton with decent reset length."""
    # This is a specific automaton designed to have a longer reset sequence
    # Based on known constructions for small n
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]  # cycle
    b = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # almost identity with 0->0, 1->0
    return a, b

def _generate_random_with_constraints():
    """Generate random automaton with constraints to encourage synchronization."""
    a = [random.randint(0, N - 1) for _ in range(N)]
    b = [random.randint(0, N - 1) for _ in range(N)]
    # Ensure both letters have at least one state mapping to each other's images
    # to increase chance of synchronization
    return a, b

def _hill_climb(start_a, start_b, max_iterations=5000, timeout=30):
    """Hill climbing to improve reset length."""
    best_a = start_a[:]
    best_b = start_b[:]
    best_len = _compute_reset_length(best_a, best_b)
    
    if best_len < 0:
        return None
    
    start_time = time.time()
    current_a = best_a[:]
    current_b = best_b[:]
    current_len = best_len
    
    for _ in range(max_iterations):
        if time.time() - start_time > timeout:
            break
        
        # Randomly modify one state of one letter
        letter = random.choice([0, 1])
        state = random.randint(0, N - 1)
        old_val = current_a[state] if letter == 0 else current_b[state]
        new_val = random.randint(0, N - 1)
        
        if letter == 0:
            current_a[state] = new_val
        else:
            current_b[state] = new_val
        
        new_len = _compute_reset_length(current_a, current_b)
        
        if new_len >= 0 and new_len > current_len:
            current_len = new_len
            if current_len > best_len:
                best_a = current_a[:]
                best_b = current_b[:]
                best_len = current_len
        else:
            # Revert
            if letter == 0:
                current_a[state] = old_val
            else:
                current_b[state] = old_val
    
    return best_a, best_b, best_len

def construct():
    """Construct a synchronizing automaton maximizing reset word length."""
    deadline = time.time() + 85  # Leave 5s safety margin
    
    best_result = None
    best_score = -1
    
    # Strategy 1: Try known good automaton
    a, b = _generate_known_good_automaton()
    length = _compute_reset_length(a, b)
    if length > best_score:
        best_score = length
        best_result = (a[:], b[:])
    
    # Strategy 2: Try various structured generators
    generators = [
        _generate_cycle_with_random_map,
        _generate_star_automaton,
        _generate_binary_tree_automaton,
        _generate_parking_automaton,
    ]
    
    for gen in generators:
        if time.time() > deadline:
            break
        a, b = gen()
        length = _compute_reset_length(a, b)
        if length > best_score:
            best_score = length
            best_result = (a[:], b[:])
    
    # Strategy 3: Random search with hill climbing
    if time.time() < deadline - 20:
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            # Generate random starting point
            a = [random.randint(0, N - 1) for _ in range(N)]
            b = [random.randint(0, N - 1) for _ in range(N)]
            
            length = _compute_reset_length(a, b)
            if length < 0:
                continue
            
            if length > best_score:
                best_score = length
                best_result = (a[:], b[:])
            
            # Hill climb from this point
            result = _hill_climb(a, b, max_iterations=2000, timeout=5)
            if result is not None:
                ha, hb, hlen = result
                if hlen > best_score:
                    best_score = hlen
                    best_result = (ha, hb)
    
    # Strategy 4: More extensive random search if time remains
    if time.time() < deadline - 5:
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            
            a = [random.randint(0, N - 1) for _ in range(N)]
            b = [random.randint(0, N - 1) for _ in range(N)]
            
            length = _compute_reset_length(a, b)
            if length > best_score:
                best_score = length
                best_result = (a[:], b[:])
    
    if best_result is None:
        # Fallback: simple synchronizing automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    return best_result
# EVOLVE-BLOCK-END
