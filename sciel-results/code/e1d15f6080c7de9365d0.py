"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

N = 12

# EVOLVE-BLOCK-START
def _apply_subset(subset, trans, letter):
    """Apply a letter to a subset of states. subset is a frozenset."""
    new_set = set()
    for s in subset:
        new_set.add(trans[letter][s])
    return frozenset(new_set)

def _compute_threshold(trans):
    """
    Compute the exact shortest reset word length using BFS on the subset automaton.
    Returns -1 if not synchronizing.
    """
    full_set = frozenset(range(N))
    
    visited = {full_set: 0}
    queue = deque([full_set])
    
    while queue:
        subset = queue.popleft()
        dist = visited[subset]
        
        if len(subset) == 1:
            return dist
        
        for letter in range(2):
            new_subset = _apply_subset(subset, trans, letter)
            if new_subset not in visited:
                visited[new_subset] = dist + 1
                queue.append(new_subset)
    
    return -1

def _generate_pinwheel():
    """Generate a pinwheel-like base structure."""
    # State 0 is the sink-ish state, others form a cycle
    a = [0] * N
    b = [0] * N
    
    # b is a cycle: 0->1->2->...->11->0
    for i in range(N):
        b[i] = (i + 1) % N
    
    # a maps most states to 0, but one state maps to something else to create a "pinwheel"
    # Classic pinwheel: a[i] = 0 for all i, but a[0] = 1 (or similar)
    # Actually, the standard pinwheel automaton for Černý:
    # a: i -> i+1 mod n, b: i -> 0 for i>0, b[0]=1
    # Let's try a different known high-threshold structure.
    
    # Let's use a "pinwheel" where:
    # Letter 'a': i -> (i+1) % N
    # Letter 'b': i -> 0 for i != 0, b[0] = 1
    # This is the standard pinwheel.
    
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    b[0] = 1
    
    return [a, b]

def _mutate(trans):
    """Apply random mutations to the transition table."""
    a = trans[0][:]
    b = trans[1][:]
    
    num_mutations = random.randint(1, 5)
    for _ in range(num_mutations):
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        new_dest = random.randint(0, N - 1)
        
        if letter == 0:
            a[state] = new_dest
        else:
            b[state] = new_dest
    
    return [a, b]

def construct():
    """
    Construct a synchronizing automaton with 12 states and two input letters,
    maximizing the length of its shortest reset word.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    best_trans = None
    best_score = -1
    
    # Start with a known good structure
    base = _generate_pinwheel()
    base_score = _compute_threshold(base)
    
    if base_score > best_score:
        best_score = base_score
        best_trans = [row[:] for row in base]
    
    # Random search with mutations
    current = [row[:] for row in base]
    
    while time.time() < deadline:
        # Generate a candidate
        if random.random() < 0.3:
            # Mutate the current best
            candidate = _mutate(best_trans)
        else:
            # Mutate the current local search point
            candidate = _mutate(current)
        
        # Evaluate
        score = _compute_threshold(candidate)
        
        if score > best_score:
            best_score = score
            best_trans = [row[:] for row in candidate]
        
        # Update current point (accept if not worse, or with some probability)
        if score >= _compute_threshold(current):
            current = [row[:] for row in candidate]
        elif random.random() < 0.1:
            # Occasional random restart
            current = _mutate(base)
    
    return best_trans
# EVOLVE-BLOCK-END
