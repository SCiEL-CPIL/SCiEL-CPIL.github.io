"""
Construct a 12-state binary deterministic automaton maximizing the shortest reset word length.
Uses structured initialization (ping-pong/pinwheel families) plus targeted local search.
"""
import time
import random
from collections import deque

N = 12
DEADLINE = 85.0  # safety margin before 90s kill

def _compute_threshold(trans):
    """
    Compute exact shortest reset word length via BFS on subset automaton.
    trans: list of two lists, trans[letter][state] = dest state.
    Returns (threshold_length, is_synchronizing).
    """
    full = (1 << N) - 1  # bitmask for all states
    # Precompute subset transitions for speed
    # For each letter and each subset (bitmask), compute image subset
    # But 2^12 * 2 = 8192 entries, each is a bitmask. Precompute per automaton.
    
    # Actually, for BFS we need to apply letters to subsets.
    # Let's use bitmask representation.
    # subset is an int bitmask.
    # Apply letter l: new_subset = OR over s in subset of (1 << trans[l][s])
    
    # Precompute for this automaton: for each letter, a list of length 2^N? No, too big.
    # Instead, precompute for each letter and each state the destination bit.
    dest_bits = []
    for l in range(2):
        db = [1 << trans[l][s] for s in range(N)]
        dest_bits.append(db)
    
    # BFS from full set
    visited = {full: 0}
    queue = deque([full])
    
    while queue:
        subset = queue.popleft()
        dist = visited[subset]
        # Check if singleton
        if subset & (subset - 1) == 0:  # power of 2
            return (dist, True)
        for l in range(2):
            db = dest_bits[l]
            new_subset = 0
            # Apply letter l to subset
            # Iterate over set bits
            temp = subset
            while temp:
                bit = temp & (-temp)
                s = bit.bit_length() - 1
                new_subset |= db[s]
                temp ^= bit
            if new_subset not in visited:
                visited[new_subset] = dist + 1
                queue.append(new_subset)
    
    return (-1, False)

def _ping_pong_init():
    """
    Ping-pong automaton: states 0..N-1.
    Letter 0: i -> i+1 for i < N-1, N-1 -> N-1 (or some sink)
    Letter 1: i -> i-1 for i > 0, 0 -> 0
    This has threshold N-1.
    Let's make it slightly different to be a valid candidate.
    """
    a = [0] * N
    b = [0] * N
    for i in range(N):
        a[i] = (i + 1) % N  # cycle
        b[i] = max(0, i - 1)  # shift left, 0 stays
    return [a, b]

def _pinwheel_init():
    """
    Pinwheel-like: a is a cycle, b maps most states to 0 but one state to something else.
    """
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    b[N-1] = N-2  # slight asymmetry
    return [a, b]

def _random_init():
    a = [random.randint(0, N-1) for _ in range(N)]
    b = [random.randint(0, N-1) for _ in range(N)]
    return [a, b]

def _mutate(trans, rng):
    """
    Targeted mutation: change 1-3 random state mappings in random letters.
    """
    new_trans = [trans[0][:], trans[1][:]]
    num_changes = rng.randint(1, 3)
    for _ in range(num_changes):
        letter = rng.randint(0, 1)
        state = rng.randint(0, N-1)
        new_trans[letter][state] = rng.randint(0, N-1)
    return new_trans

def _cycle_defect_mutation(trans, rng):
    """
    Mutation that introduces a cycle defect: pick a letter, break its cycle structure.
    """
    new_trans = [trans[0][:], trans[1][:]]
    letter = rng.randint(0, 1)
    # Pick a random state and redirect it
    state = rng.randint(0, N-1)
    new_trans[letter][state] = rng.randint(0, N-1)
    # Possibly also adjust another state
    if rng.random() < 0.3:
        state2 = rng.randint(0, N-1)
        new_trans[letter][state2] = rng.randint(0, N-1)
    return new_trans

def _reset_adjust_mutation(trans, rng):
    """
    Adjust the 'reset' letter (usually letter 1) to create stronger convergence.
    """
    new_trans = [trans[0][:], trans[1][:]]
    letter = 1  # assume letter 1 is the reset letter
    # Map a few more states to 0
    for _ in range(rng.randint(1, 2)):
        state = rng.randint(0, N-1)
        if new_trans[letter][state] != 0:
            new_trans[letter][state] = 0
    return new_trans

def _asymmetry_mutation(trans, rng):
    """
    Create asymmetry between the two letters.
    """
    new_trans = [trans[0][:], trans[1][:]]
    # Swap some mappings between letters
    num_swaps = rng.randint(1, 2)
    for _ in range(num_swaps):
        state = rng.randint(0, N-1)
        new_trans[0][state], new_trans[1][state] = new_trans[1][state], new_trans[0][state]
    return new_trans

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_trans = None
    best_score = -1
    
    # Initial candidates
    candidates = []
    candidates.append(_ping_pong_init())
    candidates.append(_pinwheel_init())
    for _ in range(5):
        candidates.append(_random_init())
    
    # Evaluate initial candidates
    for trans in candidates:
        if time.time() - start_time > DEADLINE:
            break
        score, sync = _compute_threshold(trans)
        if sync and score > best_score:
            best_score = score
            best_trans = [trans[0][:], trans[1][:]]
    
    # Local search
    mutation_fns = [_mutate, _cycle_defect_mutation, _reset_adjust_mutation, _asymmetry_mutation]
    
    while time.time() - start_time < DEADLINE - 5:
        # Pick a mutation function
        mut_fn = rng.choice(mutation_fns)
        if best_trans is None:
            new_trans = _random_init()
        else:
            new_trans = mut_fn(best_trans, rng)
        
        score, sync = _compute_threshold(new_trans)
        if sync and score > best_score:
            best_score = score
            best_trans = [new_trans[0][:], new_trans[1][:]]
            # Exploit: do more mutations around this good solution
            for _ in range(3):
                if time.time() - start_time > DEADLINE - 5:
                    break
                mut_fn2 = rng.choice(mutation_fns)
                new_trans2 = mut_fn2(best_trans, rng)
                score2, sync2 = _compute_threshold(new_trans2)
                if sync2 and score2 > best_score:
                    best_score = score2
                    best_trans = [new_trans2[0][:], new_trans2[1][:]]
    
    if best_trans is None:
        # Fallback: return a simple synchronizing automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    return best_trans

# EVOLVE-BLOCK-START
# (The construct function is defined above; this block is kept for compatibility)
# EVOLVE-BLOCK-END
