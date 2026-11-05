"""
Constructs a 12-state binary deterministic automaton maximizing shortest reset word length.
Uses bitmask BFS for fast threshold computation and simulated annealing with restarts.
"""
import time
import random
import math
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

# EVOLVE-BLOCK-START

def _apply_letter(mask, mapping):
    """Apply a letter mapping to a bitmask subset."""
    result = 0
    # Unroll for speed
    for i in range(N):
        if mask & (1 << i):
            result |= 1 << mapping[i]
    return result


def _threshold_fast(a, b, timeout=0.002):
    """Compute reset threshold using bitmask BFS. Returns -1 if not synchronizing within timeout."""
    start = time.time()
    visited = bytearray(1 << N)  # 4096 bytes, fast
    visited[FULL_MASK] = 1
    queue = deque()
    queue.append((FULL_MASK, 0))
    
    singletons_set = set(1 << i for i in range(N))
    
    while queue:
        if time.time() - start > timeout:
            return -1
        mask, depth = queue.popleft()
        
        if mask in singletons_set:
            return depth
        
        new_mask_a = _apply_letter(mask, a)
        if not visited[new_mask_a]:
            visited[new_mask_a] = 1
            queue.append((new_mask_a, depth + 1))
        
        new_mask_b = _apply_letter(mask, b)
        if not visited[new_mask_b]:
            visited[new_mask_b] = 1
            queue.append((new_mask_b, depth + 1))
    
    return -1  # Not synchronizing


def _is_synchronizing(a, b):
    """Check if automaton synchronizes."""
    visited = bytearray(1 << N)
    visited[FULL_MASK] = 1
    queue = deque()
    queue.append(FULL_MASK)
    
    singletons_set = set(1 << i for i in range(N))
    
    while queue:
        mask = queue.popleft()
        if mask in singletons_set:
            return True
        
        new_mask_a = _apply_letter(mask, a)
        if not visited[new_mask_a]:
            visited[new_mask_a] = 1
            queue.append(new_mask_a)
        
        new_mask_b = _apply_letter(mask, b)
        if not visited[new_mask_b]:
            visited[new_mask_b] = 1
            queue.append(new_mask_b)
    
    return False


def _random_automaton():
    """Generate a random automaton biased toward synchronizing ones."""
    # Make 'a' a permutation (cycle structure helps synchronization)
    a = list(range(N))
    random.shuffle(a)
    
    # Make 'b' have structure: mostly a shift with perturbations
    shift = random.randint(1, N - 1)
    b = [(i + shift) % N for i in range(N)]
    for _ in range(random.randint(0, 3)):
        pos = random.randint(0, N - 1)
        b[pos] = random.randint(0, N - 1)
    
    return a, b


def _mutate(a, b, rate=0.3):
    """Create a mutated version of the automaton."""
    a_new = a[:]
    b_new = b[:]
    
    for i in range(N):
        if random.random() < rate:
            a_new[i] = random.randint(0, N - 1)
    for i in range(N):
        if random.random() < rate:
            b_new[i] = random.randint(0, N - 1)
    
    return a_new, b_new


def construct():
    """Construct a 12-state binary automaton maximizing reset threshold."""
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    best_a = None
    best_b = None
    best_score = -1
    
    # Start with the Černý construction: a is a cycle, b maps 0->0 and i->(i+1)%N for i>0
    a_cycle = [(i + 1) % N for i in range(N)]
    b_cerny = [0] * N
    for i in range(1, N):
        b_cerny[i] = (i + 1) % N
    
    score = _threshold_fast(a_cycle, b_cerny)
    if score > best_score:
        best_score = score
        best_a = a_cycle[:]
        best_b = b_cerny[:]
    
    # Try variations of the Černý construction
    for shift in range(1, N):
        b_try = [0] * N
        for i in range(1, N):
            b_try[i] = (i + shift) % N
        s = _threshold_fast(a_cycle, b_try)
        if s > best_score:
            best_score = s
            best_a = a_cycle[:]
            best_b = b_try[:]
    
    # Try reverse cycle
    a_rev = [(i - 1) % N for i in range(N)]
    for shift in range(1, N):
        b_try = [0] * N
        for i in range(1, N):
            b_try[i] = (i + shift) % N
        s = _threshold_fast(a_rev, b_try)
        if s > best_score:
            best_score = s
            best_a = a_rev[:]
            best_b = b_try[:]
    
    # Simulated annealing with random restarts
    random.seed(42)
    
    # Temperature schedule
    t_start = 1.0
    t_end = 0.01
    cooling = 0.995
    
    for restart in range(200):
        if time.time() > deadline - 3.0:
            break
        
        # Initialize: either from best or random
        if restart == 0 or random.random() < 0.3:
            a, b = _random_automaton()
        else:
            a, b = best_a[:], best_b[:]
        
        # Check initial state
        if not _is_synchronizing(a, b):
            continue
        
        score = _threshold_fast(a, b)
        if score < 0:
            continue
        
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
        
        # Annealing
        t = t_start
        steps_per_temp = 50
        
        for step in range(5000):
            if time.time() > deadline - 3.0:
                break
            
            if step % steps_per_temp == 0:
                t *= cooling
                if t < t_end:
                    t = t_end
            
            # Mutate
            a_mut, b_mut = _mutate(a, b, rate=0.2)
            
            # Check if synchronizing
            if not _is_synchronizing(a_mut, b_mut):
                continue
            
            score_mut = _threshold_fast(a_mut, b_mut)
            if score_mut < 0:
                continue
            
            # Acceptance criterion
            if score_mut >= score:
                accept = True
            else:
                delta = score - score_mut
                accept = random.random() < math.exp(-delta / max(t, 1e-10))
            
            if accept:
                a, b = a_mut, b_mut
                score = score_mut
                if score > best_score:
                    best_score = score
                    best_a = a[:]
                    best_b = b[:]
    
    if best_a is None:
        best_a = a_cycle
        best_b = b_cerny
    
    return [best_a, best_b]

# EVOLVE-BLOCK-END
