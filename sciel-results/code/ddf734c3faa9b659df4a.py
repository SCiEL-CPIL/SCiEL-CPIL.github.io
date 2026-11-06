"""Černý problem: 12-state binary automata with long reset thresholds."""
import time
import random
from collections import deque

N = 12

# EVOLVE-BLOCK-START

def _bfs_reset_length(automaton):
    """Compute the exact shortest reset word length via BFS on subset automaton.
    Returns (length, word) or (None, None) if not synchronizing."""
    n = len(automaton[0])
    # Represent subsets as frozensets of state indices
    start = frozenset(range(n))
    target = None  # any singleton
    
    # BFS
    visited = {start}
    queue = deque()
    queue.append((start, ""))
    
    while queue:
        subset, word = queue.popleft()
        if len(subset) == 1:
            return len(word), word
        
        for letter in range(2):
            # Apply letter to all states in subset
            new_states = set(automaton[letter][s] for s in subset)
            new_subset = frozenset(new_states)
            if new_subset not in visited:
                visited.add(new_subset)
                queue.append((new_subset, word + str(letter)))
    
    return None, None


def _construct_pinwheel(k):
    """Construct a pinwheel-like automaton with parameter k.
    Letter 0: cycle (i -> i+1 mod N)
    Letter 1: maps states to create long synchronization path
    
    The pinwheel structure: letter 1 maps most states forward by k,
    but one state maps backward, creating a long reset path.
    """
    letter0 = [(i + 1) % N for i in range(N)]
    
    # Letter 1: for each state i, map to (i + k) % N, but with a twist
    # to ensure synchronization and long reset word
    # Use a "broken cycle" where one transition goes to 0
    letter1 = []
    for i in range(N):
        if i == 0:
            letter1.append(0)  # fixed point at 0
        else:
            letter1.append((i + k) % N)
    
    return [letter0, letter1]


def _construct_cycle_clamp(k, clamp_state):
    """Cycle with a clamping transition.
    Letter 0: cycle
    Letter 1: maps everything to (i+k)%N except clamp_state maps to 0
    """
    letter0 = [(i + 1) % N for i in range(N)]
    letter1 = []
    for i in range(N):
        if i == clamp_state:
            letter1.append(0)
        else:
            letter1.append((i + k) % N)
    return [letter0, letter1]


def _construct_double_cycle(k1, k2):
    """Two different cycle steps for the two letters."""
    letter0 = [(i + k1) % N for i in range(N)]
    letter1 = [(i + k2) % N for i in range(N)]
    return [letter0, letter1]


def _construct_pinwheel_v2(k, offset):
    """Pinwheel variant with offset in the clamp position."""
    letter0 = [(i + 1) % N for i in range(N)]
    letter1 = []
    for i in range(N):
        if i == offset:
            letter1.append(0)
        else:
            letter1.append((i + k) % N)
    return [letter0, letter1]


def _construct_star_cycle(k):
    """Star + cycle hybrid: letter 1 maps most to 0, one state cycles."""
    letter0 = [(i + 1) % N for i in range(N)]
    letter1 = [0] * N
    letter1[0] = k  # one state doesn't go to 0
    return [letter0, letter1]


def _construct_asymmetric(k, m):
    """Asymmetric transitions: letter 1 maps i -> (i+k)%N for i<m, else i -> (i+m)%N"""
    letter0 = [(i + 1) % N for i in range(N)]
    letter1 = []
    for i in range(N):
        if i < m:
            letter1.append((i + k) % N)
        else:
            letter1.append((i + m) % N)
    return [letter0, letter1]


def _construct_reflective(k):
    """Reflective structure: letter 1 mirrors around a point."""
    letter0 = [(i + 1) % N for i in range(N)]
    letter1 = []
    center = N // 2
    for i in range(N):
        # Reflect i around center, then shift by k
        reflected = (2 * center - i) % N
        letter1.append((reflected + k) % N)
    return [letter0, letter1]


def _construct_hierarchical(level_k):
    """Hierarchical: group states into levels, letter 1 collapses levels."""
    letter0 = [(i + 1) % N for i in range(N)]
    letter1 = []
    # Divide into 3 groups of 4
    for i in range(N):
        group = i // 4
        if group == 0:
            letter1.append(i)  # stay in group 0
        elif group == 1:
            letter1.append((i - 4) % N)  # collapse to group 0
        else:
            letter1.append((i - 8) % N)  # collapse to group 0
    return [letter0, letter1]


def _construct_spiral(k):
    """Spiral pattern: letter 1 creates a spiral inward."""
    letter0 = [(i + 1) % N for i in range(N)]
    letter1 = []
    # Create a pattern that moves states inward in a spiral
    for i in range(N):
        if i == 0:
            letter1.append(0)
        elif i % 2 == 0:
            letter1.append((i + k) % N)
        else:
            letter1.append((i - 1) % N)
    return [letter0, letter1]


def _construct_shift_break(k, break_pos):
    """Shift by k with a single break at break_pos mapping to 0."""
    letter0 = [(i + 1) % N for i in range(N)]
    letter1 = []
    for i in range(N):
        if i == break_pos:
            letter1.append(0)
        else:
            letter1.append((i + k) % N)
    return [letter0, letter1]


def _construct_two_breaks(k, b1, b2):
    """Shift by k with two breaks mapping to 0."""
    letter0 = [(i + 1) % N for i in range(N)]
    letter1 = []
    for i in range(N):
        if i == b1 or i == b2:
            letter1.append(0)
        else:
            letter1.append((i + k) % N)
    return [letter0, letter1]


def _construct_modular(k, mod_base):
    """Modular structure: letter 1 applies different shifts based on i mod mod_base."""
    letter0 = [(i + 1) % N for i in range(N)]
    letter1 = []
    for i in range(N):
        r = i % mod_base
        letter1.append((i + k * (r + 1)) % N)
    return [letter0, letter1]


def _construct_gcd(k1, k2):
    """Structure based on gcd properties."""
    letter0 = [(i + k1) % N for i in range(N)]
    letter1 = [(i + k2) % N for i in range(N)]
    return [letter0, letter1]


def _construct_random_perturbed(base_automaton, n_perturb):
    """Perturb a base automaton at random positions."""
    letter0 = list(base_automaton[0])
    letter1 = list(base_automaton[1])
    for _ in range(n_perturb):
        pos = random.randint(0, N - 1)
        letter1[pos] = random.randint(0, N - 1)
    return [letter0, letter1]


def construct():
    """Construct a 12-state binary synchronizing automaton maximizing reset word length."""
    deadline = time.time() + 80  # 80 second budget, leaving 10s margin
    
    best_length = 0
    best_automaton = None
    
    # Generate candidate automata from various structural families
    candidates = []
    
    # Pinwheel variants
    for k in range(1, N):
        candidates.append(_construct_pinwheel(k))
    
    # Cycle + clamp variants
    for k in range(1, N):
        for cs in range(N):
            candidates.append(_construct_cycle_clamp(k, cs))
    
    # Double cycle
    for k1 in range(1, N):
        for k2 in range(1, N):
            if k1 != k2:
                candidates.append(_construct_double_cycle(k1, k2))
    
    # Pinwheel v2 with offset
    for k in range(1, N):
        for offset in range(N):
            candidates.append(_construct_pinwheel_v2(k, offset))
    
    # Star cycle
    for k in range(1, N):
        candidates.append(_construct_star_cycle(k))
    
    # Asymmetric
    for k in range(1, N):
        for m in range(1, N):
            candidates.append(_construct_asymmetric(k, m))
    
    # Reflective
    for k in range(1, N):
        candidates.append(_construct_reflective(k))
    
    # Hierarchical
    candidates.append(_construct_hierarchical(1))
    
    # Spiral
    for k in range(1, N):
        candidates.append(_construct_spiral(k))
    
    # Shift with single break
    for k in range(1, N):
        for bp in range(N):
            candidates.append(_construct_shift_break(k, bp))
    
    # Shift with two breaks
    for k in range(1, N):
        for b1 in range(N):
            for b2 in range(N):
                if b1 < b2:
                    candidates.append(_construct_two_breaks(k, b1, b2))
    
    # Modular
    for k in range(1, N):
        for mb in range(2, N + 1):
            candidates.append(_construct_modular(k, mb))
    
    # GCD-based
    for k1 in range(1, N):
        for k2 in range(1, N):
            candidates.append(_construct_gcd(k1, k2))
    
    # Evaluate candidates
    for auto in candidates:
        if time.time() > deadline:
            break
        
        length, word = _bfs_reset_length(auto)
        if length is not None and length > best_length:
            best_length = length
            best_automaton = [list(auto[0]), list(auto[1])]
    
    # If we have time, do random local search around the best
    if best_automaton is not None and time.time() < deadline - 5:
        current_best = best_length
        current_auto = [list(best_automaton[0]), list(best_automaton[1])]
        
        # Local search: perturb letter 1 at random positions
        for iteration in range(2000):
            if time.time() > deadline - 2:
                break
            
            trial = [list(current_auto[0]), list(current_auto[1])]
            pos = random.randint(0, N - 1)
            trial[1][pos] = random.randint(0, N - 1)
            
            length, word = _bfs_reset_length(trial)
            if length is not None and length > current_best:
                current_best = length
                current_auto = trial
                best_length = length
                best_automaton = [list(current_auto[0]), list(current_auto[1])]
        
        # Also try perturbing letter 0
        for iteration in range(1000):
            if time.time() > deadline - 2:
                break
            
            trial = [list(current_auto[0]), list(current_auto[1])]
            pos = random.randint(0, N - 1)
            trial[0][pos] = random.randint(0, N - 1)
            
            length, word = _bfs_reset_length(trial)
            if length is not None and length > current_best:
                current_best = length
                current_auto = trial
                best_length = length
                best_automaton = [list(current_auto[0]), list(current_auto[1])]
    
    # Fallback: return the simple seed if nothing better found
    if best_automaton is None:
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    return best_automaton

# EVOLVE-BLOCK-END
