"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import random
import time
import math
from collections import deque

FULL_MASK = (1 << N) - 1
DEADLINE_SECONDS = 82.0

def _build_subset_trans(trans):
    """Precompute subset transitions for one letter using dynamic programming."""
    size = 1 << N
    next_subset = [0] * size
    for mask in range(1, size):
        lsb = mask & (-mask)
        i = lsb.bit_length() - 1
        rest = mask ^ lsb
        next_subset[mask] = next_subset[rest] | (1 << trans[i])
    return next_subset

def compute_threshold(trans_a, trans_b):
    """
    Computes the length of the shortest reset word for the automaton defined by
    trans_a and trans_b. Returns 0 if not synchronizing.
    Uses BFS on the subset automaton with precomputed transitions.
    """
    # Quick check: if either letter is constant, threshold is 1
    if len(set(trans_a)) == 1 or len(set(trans_b)) == 1:
        return 1
    
    # Precompute subset transitions
    trans_a_sub = _build_subset_trans(trans_a)
    trans_b_sub = _build_subset_trans(trans_b)
    
    # BFS from full set to any singleton
    visited = bytearray(1 << N)  # 0 = unvisited, 1 = visited
    visited[FULL_MASK] = 1
    
    queue = deque()
    queue.append((FULL_MASK, 0))
    
    while queue:
        mask, dist = queue.popleft()
        
        # Check if singleton
        if mask & (mask - 1) == 0:
            return dist
        
        # Try both letters
        next_a = trans_a_sub[mask]
        if not visited[next_a]:
            visited[next_a] = 1
            queue.append((next_a, dist + 1))
        
        next_b = trans_b_sub[mask]
        if not visited[next_b]:
            visited[next_b] = 1
            queue.append((next_b, dist + 1))
    
    return 0  # Not synchronizing

def _structural_starts():
    """Generate structurally motivated starting points."""
    starts = []
    
    # 1. Černý/Pinsker type: cycle + reset
    a = list(range(N))
    b = [0] * N
    starts.append((a[:], b[:]))
    
    # 2. Cycle with one perturbation
    a = list(range(N))
    b = [0] * N
    b[N-1] = 1
    starts.append((a[:], b[:]))
    
    # 3. Two-cycle structure
    a = [(i + 1) % N for i in range(N)]
    b = [i // 2 for i in range(N)]
    starts.append((a[:], b[:]))
    
    # 4. Random permutation + biased reset
    a = list(range(N))
    random.shuffle(a)
    b = [0 if random.random() < 0.7 else random.randint(1, N-1) for _ in range(N)]
    starts.append((a[:], b[:]))
    
    # 5. Another random structure
    a = list(range(N))
    random.shuffle(a)
    b = [random.randint(0, min(2, N-1)) for _ in range(N)]
    starts.append((a[:], b[:]))
    
    return starts

def _random_start():
    """Generate a random synchronizing automaton."""
    for _ in range(100):
        a = [random.randint(0, N-1) for _ in range(N)]
        b = [random.randint(0, N-1) for _ in range(N)]
        if compute_threshold(a, b) > 0:
            return a, b
    # Fallback
    return list(range(N)), [0] * N

def _mutate(a, b, num_mutations=None):
    """Apply random mutations to the automaton."""
    a_new = a[:]
    b_new = b[:]
    
    if num_mutations is None:
        num_mutations = random.randint(1, 3)
    
    for _ in range(num_mutations):
        if random.random() < 0.5:
            idx = random.randint(0, N - 1)
            a_new[idx] = random.randint(0, N - 1)
        else:
            idx = random.randint(0, N - 1)
            b_new[idx] = random.randint(0, N - 1)
    
    return a_new, b_new

def _anneal(a, b, current_len, deadline):
    """Simulated annealing to improve the automaton."""
    best_a, best_b, best_len = a[:], b[:], current_len
    temp = 5.0
    cooling = 0.995
    
    while time.time() < deadline and temp > 0.1:
        a_new, b_new = _mutate(a, b)
        new_len = compute_threshold(a_new, b_new)
        
        if new_len == 0:
            temp *= cooling
            continue
        
        if new_len > current_len:
            a, b, current_len = a_new, b_new, new_len
            if new_len > best_len:
                best_a, best_b, best_len = a_new[:], b_new[:], new_len
        else:
            # Accept worse solutions with probability based on temperature
            delta = current_len - new_len
            if random.random() < math.exp(-delta / temp):
                a, b, current_len = a_new, b_new, new_len
        
        temp *= cooling
    
    return best_a, best_b, best_len

def construct():
    """Construct a synchronizing automaton with 12 states maximizing reset threshold."""
    deadline = time.time() + DEADLINE_SECONDS
    
    # Try structural starts
    best_a, best_b, best_len = None, None, 0
    
    for a, b in _structural_starts():
        if time.time() > deadline - 10:
            break
        length = compute_threshold(a, b)
        if length > best_len:
            best_len = length
            best_a, best_b = a[:], b[:]
    
    # If no good structural start found, use random
    if best_len == 0:
        best_a, best_b = _random_start()
        best_len = compute_threshold(best_a, best_b)
    
    # Run simulated annealing from best start
    if best_len > 0:
        best_a, best_b, best_len = _anneal(best_a, best_b, best_len, deadline)
    
    # Additional random restarts if time permits
    restarts = 0
    while time.time() < deadline - 5 and restarts < 20:
        restarts += 1
        a, b = _random_start()
        length = compute_threshold(a, b)
        
        if length > best_len:
            best_a, best_b, best_len = a[:], b[:], length
            # Continue annealing from this better start
            best_a, best_b, best_len = _anneal(best_a, best_b, best_len, deadline)
        elif length > 0 and time.time() < deadline - 2:
            # Try annealing from this start too
            a2, b2, len2 = _anneal(a, b, length, deadline)
            if len2 > best_len:
                best_a, best_b, best_len = a2, b2, len2
    
    return best_a, best_b
# EVOLVE-BLOCK-END
