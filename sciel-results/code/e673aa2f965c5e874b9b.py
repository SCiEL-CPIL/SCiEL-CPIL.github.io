"""
Constructs a 12-state binary deterministic automaton maximizing shortest reset word length.
Uses bitmask BFS for fast threshold computation and random restart local search.
"""
import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

# EVOLVE-BLOCK-START
def _threshold_fast(a, b, timeout=0.005):
    """Compute reset threshold using bitmask BFS. Returns -1 if not synchronizing within timeout."""
    start = time.time()
    # BFS on subsets represented as bitmasks
    # Singletons are masks with exactly one bit set
    singletons = [1 << i for i in range(N)]
    visited = set()
    queue = deque()
    
    # Start from full set
    visited.add(FULL_MASK)
    queue.append((FULL_MASK, 0))
    
    while queue:
        if time.time() - start > timeout:
            return -1
        mask, depth = queue.popleft()
        
        # Check if any singleton is reachable (we're already at a singleton)
        if mask in singletons:
            return depth
        
        # Try letter a
        new_mask_a = 0
        for i in range(N):
            if mask & (1 << i):
                new_mask_a |= 1 << a[i]
        if new_mask_a not in visited:
            visited.add(new_mask_a)
            queue.append((new_mask_a, depth + 1))
        
        # Try letter b
        new_mask_b = 0
        for i in range(N):
            if mask & (1 << i):
                new_mask_b |= 1 << b[i]
        if new_mask_b not in visited:
            visited.add(new_mask_b)
            queue.append((new_mask_b, depth + 1))
    
    return -1  # Not synchronizing


def _is_synchronizing(a, b):
    """Check if automaton synchronizes (faster than full threshold)."""
    visited = set()
    queue = deque()
    visited.add(FULL_MASK)
    queue.append(FULL_MASK)
    
    singletons = set(1 << i for i in range(N))
    
    while queue:
        mask = queue.popleft()
        if mask in singletons:
            return True
        
        new_mask_a = 0
        for i in range(N):
            if mask & (1 << i):
                new_mask_a |= 1 << a[i]
        if new_mask_a not in visited:
            visited.add(new_mask_a)
            queue.append(new_mask_a)
        
        new_mask_b = 0
        for i in range(N):
            if mask & (1 << i):
                new_mask_b |= 1 << b[i]
        if new_mask_b not in visited:
            visited.add(new_mask_b)
            queue.append(new_mask_b)
    
    return False


def _random_automaton():
    """Generate a random automaton, biased toward synchronizing ones."""
    # Strategy: make 'a' a permutation (often a cycle), 'b' random with some structure
    a = list(range(N))
    random.shuffle(a)
    
    # Make 'b' have some structure: mostly a shifted version with some random elements
    shift = random.randint(1, N - 1)
    b = [(i + shift) % N for i in range(N)]
    # Perturb a few positions
    for _ in range(random.randint(0, 3)):
        pos = random.randint(0, N - 1)
        b[pos] = random.randint(0, N - 1)
    
    return a, b


def _mutate(a, b):
    """Create a mutated version of the automaton."""
    a_new = a[:]
    b_new = b[:]
    
    # Mutate a few positions in 'a'
    for _ in range(random.randint(1, 2)):
        pos = random.randint(0, N - 1)
        a_new[pos] = random.randint(0, N - 1)
    
    # Mutate a few positions in 'b'
    for _ in range(random.randint(1, 2)):
        pos = random.randint(0, N - 1)
        b_new[pos] = random.randint(0, N - 1)
    
    return a_new, b_new


def construct():
    """Construct a 12-state binary automaton maximizing reset threshold."""
    deadline = time.time() + 80.0  # Safety margin before 90s kill
    
    best_a = None
    best_b = None
    best_score = -1
    
    # Start with a known good construction: cycle + modified shift
    # This is the "Černý-like" construction
    a_cycle = [(i + 1) % N for i in range(N)]
    
    # Try the standard Černý construction variant
    # b maps 0->0, i->(i+1)%N for i>0
    b_standard = [0] * N
    for i in range(1, N):
        b_standard[i] = (i + 1) % N
    
    score = _threshold_fast(a_cycle, b_standard)
    if score > best_score:
        best_score = score
        best_a = a_cycle[:]
        best_b = b_standard[:]
    
    # Try various shift values for b
    for shift in range(1, N):
        b_try = [0] * N
        for i in range(1, N):
            b_try[i] = (i + shift) % N
        s = _threshold_fast(a_cycle, b_try)
        if s > best_score:
            best_score = s
            best_a = a_cycle[:]
            best_b = b_try[:]
    
    # Also try: a is cycle, b maps i -> (i - 1) % N for i > 0, b[0] = 0
    b_back = [0] * N
    for i in range(1, N):
        b_back[i] = (i - 1) % N
    s = _threshold_fast(a_cycle, b_back)
    if s > best_score:
        best_score = s
        best_a = a_cycle[:]
        best_b = b_back[:]
    
    # Random restart local search
    random.seed(42)
    
    for restart in range(500):
        if time.time() > deadline - 5.0:
            break
        
        # Generate random starting point
        a, b = _random_automaton()
        
        # Check if synchronizing
        if not _is_synchronizing(a, b):
            continue
        
        score = _threshold_fast(a, b)
        if score < 0:
            continue
        
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
        
        # Local search: try mutations
        for step in range(100):
            if time.time() > deadline - 5.0:
                break
            
            a_mut, b_mut = _mutate(a, b)
            
            if not _is_synchronizing(a_mut, b_mut):
                continue
            
            score_mut = _threshold_fast(a_mut, b_mut)
            if score_mut < 0:
                continue
            
            if score_mut > score:
                a, b = a_mut, b_mut
                score = score_mut
                if score > best_score:
                    best_score = score
                    best_a = a[:]
                    best_b = b[:]
    
    # If we didn't find anything better, use the best known construction
    if best_a is None:
        best_a = a_cycle
        best_b = b_standard
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
