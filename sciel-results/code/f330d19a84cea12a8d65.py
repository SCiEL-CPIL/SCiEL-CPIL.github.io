import time
import random
import itertools
import numpy as np

# Precompute all words and their neighbors for the specific task (q=3, n=7, radius=1)
Q = 3
N_LEN = 7
RADIUS = 1
TOTAL_WORDS = Q ** N_LEN

# Map word tuple to index
word_to_idx = {}
idx_to_word = {}
index = 0
for w in itertools.product(range(Q), repeat=N_LEN):
    word_to_idx[w] = index
    idx_to_word[index] = w
    index += 1

# Precompute neighbors (ball of radius 1) as numpy arrays for speed
neighbors_list = []
for i in range(TOTAL_WORDS):
    w = idx_to_word[i]
    ball = [i]
    for pos in range(N_LEN):
        for val in range(Q):
            if val == w[pos]:
                continue
            new_w = list(w)
            new_w[pos] = val
            ball.append(word_to_idx[tuple(new_w)])
    neighbors_list.append(ball)

# Convert to numpy int32 arrays for fast indexing
neighbors = [np.array(b, dtype=np.int32) for b in neighbors_list]

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (150s kill)
    
    # Fallback to the known product code if time is tight or for initialization
    # Product code: tetracode (9 words, len 4) x 3^(n-4)
    # This gives 243 words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    base_sol = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    base_size = len(base_sol)
    
    best_size = base_size
    best_solution = base_sol
    
    seed = 0
    while time.time() < deadline:
        sol, size = run_greedy_search(seed)
        if size < best_size:
            best_size = size
            best_solution = sol
            # If we hit the theoretical lower bound (156), we can stop early
            if best_size <= 156:
                break
        seed += 1
        
    return best_solution

def run_greedy_search(seed_val):
    rng = random.Random(seed_val)
    cov = np.zeros(TOTAL_WORDS, dtype=np.int32)
    sel = []
    uncovered = TOTAL_WORDS
    
    # Greedy Construction
    while uncovered > 0:
        best_gain = -1
        best_candidates = []
        
        # Iterate all possible codewords
        for c in range(TOTAL_WORDS):
            # Vectorized gain calculation
            gain = np.sum(cov[neighbors[c]] == 0)
            if gain > best_gain:
                best_gain = gain
                best_candidates = [c]
            elif gain == best_gain and gain > 0:
                best_candidates.append(c)
        
        if best_gain == 0:
            # Should not happen if uncovered > 0
            break
            
        # Randomly pick from best candidates to diversify
        c = rng.choice(best_candidates)
        sel.append(c)
        
        # Update coverage
        ball = neighbors[c]
        new_covered = cov[ball] == 0
        uncovered -= np.sum(new_covered)
        cov[ball] += 1
            
    # Deletion Phase
    sel_set = set(sel)
    changed = True
    sel_list = list(sel_set)
    
    while changed:
        changed = False
        rng.shuffle(sel_list)
        for c in sel_list:
            if c not in sel_set:
                continue
            # Check if c is essential
            # If any word in its ball has coverage 1, it's essential
            ball = neighbors[c]
            is_essential = np.any(cov[ball] == 1)
            
            if not is_essential:
                # Remove c
                sel_set.remove(c)
                cov[ball] -= 1
                changed = True
    
    # Convert back to list of words
    solution = [list(idx_to_word[c]) for c in sel_set]
    return solution, len(solution)

# Helper to ensure the function signature matches and handles edge cases if needed
_original_construct = construct

def construct(q, n, radius):
    if q == 3 and n == 7 and radius == 1:
        return _original_construct(q, n, radius)
    else:
        # Fallback for other parameters: simple greedy or product code logic if applicable
        start_time = time.time()
        deadline = start_time + 115.0
        
        if n < 4:
            return [list(w) for w in itertools.product(range(q), repeat=n)]
        
        # Generic product code attempt (only valid for specific q, but safe fallback is full space or simple greedy)
        # For safety in unknown cases, return full space (valid but large) or a simple greedy if time permits.
        # Given the task is specific, this path is unlikely.
        # Simple greedy for generic q, n, radius
        total = q ** n
        # Precompute neighbors for generic case if needed, but for now, just return full space to be safe
        # as constructing generic neighbors might be complex in this snippet context.
        return [list(w) for w in itertools.product(range(q), repeat=n)]
