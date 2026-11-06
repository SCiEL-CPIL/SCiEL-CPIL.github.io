<<<<<<< SEARCH
import time
import random

def score_set(S):
    S = set(S)
    diffs = set()
    # Iterate over all pairs to find potential differences
    # a, b in S, a < b. d = b - a.
    # Check if a + 2d is in S.
    # Note: a + 2d = a + 2(b-a) = 2b - a.
    # So we check if 2b - a is in S.
    for a in S:
        for b in S:
            if a == b: continue
            # We only care about d > 0.
            # If a < b, d = b - a > 0.
            # The progression is a, a+d, a+2d.
            # So we need a, b, 2b-a in S.
            # This covers the case where a is the start.
            # What if b is the start? Then we'd have b, b+d, b+2d.
            # This is covered when we pick the pair (b, b+d) and check b+2d.
            # So iterating all ordered pairs (a,b) with a!=b and checking if the third term exists is sufficient.
            # Actually, just checking if 2*b - a is in S for all a!=b in S is enough to find ALL 3-APs.
            # If a, a+d, a+2d are in S, then for pair (a, a+d), 2(a+d)-a = a+2d is in S.
            # And d = (a+d) - a.
            c = 2 * b - a
            if c in S:
                d = abs(b - a)
                if d > 0:
                    diffs.add(d)
    return len(diffs)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0
    best_set = set()
    
    # Initial random sets or structured starts
    # Start with a few diverse initializations
    initial_sets = []
    
    # 1. Consecutive integers
    initial_sets.append(list(range(1, 33)))
    
    # 2. Random sets
    for _ in range(5):
        s = random.sample(range(1, 513), 32)
        initial_sets.append(s)
    
    # 3. Structured: Powers of 2? No, too sparse.
    # Maybe arithmetic progressions with different steps?
    # e.g., 1, 1+k, 1+2k ... but we need 32 elements.
    
    for init in initial_sets:
        current_set = set(init)
        current_score = score_set(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = current_set
            
        # Local Search
        no_improvement_count = 0
        max_no_improvement = 2000
        
        while time.time() < deadline:
            if no_improvement_count > max_no_improvement:
                break
                
            # Pick a random element to remove
            x = random.choice(list(current_set))
            
            # Pick a random element to add from 1..512 not in current_set
            # To make it faster, we can just pick a random y in 1..512 and if it's in set, retry or skip.
            # Better: maintain a list of available numbers? 
            # 512 is small. 
            y = random.randint(1, 512)
            if y in current_set:
                continue
                
            new_set = current_set.copy()
            new_set.remove(x)
            new_set.add(y)
            
            new_score = score_set(new_set)
            
            if new_score > current_score:
                current_set = new_set
                current_score = new_score
                no_improvement_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set
            else:
                no_improvement_count += 1
                
            # Occasional restart if stuck
            if no_improvement_count > max_no_improvement * 2:
                break

    return list(best_set)
=======
import time
import random
import numpy as np

def _score_bitset(bits):
    """Fast score using numpy: count distinct d>0 with a 3-AP in the set.
    bits: np.array of shape (512,) with 1/0 indicating membership (index = value-1).
    Returns integer score.
    """
    # For each a in 1..512 (index 0..511), and each d in 1..255:
    #   check bits[a-1] & bits[a+d-1] & bits[a+2d-1]
    # We want distinct d values that work for at least one a.
    # 
    # Vectorized approach:
    # For each d, compute whether there exists a such that all three are in set.
    # a ranges from 1 to 512-2d, so a+2d <= 512.
    # 
    # Let's build a 2D check: for each d in 1..255, and a in 1..(512-2d+1),
    #   check if bits[a-1] & bits[a+d-1] & bits[a+2d-1]
    # This is awkward to fully vectorize in one shot. Instead, loop over d (255 iterations)
    # and use numpy for the a-dimension.
    
    found = np.zeros(256, dtype=bool)  # found[d] = True if some a works for this d
    
    for d in range(1, 256):
        # a ranges from 0 to 512-2d-1 (0-indexed), so a+2d <= 511
        max_a = 512 - 2 * d  # number of valid a positions (0-indexed a from 0 to max_a-1)
        if max_a <= 0:
            break
        # indices: a, a+d, a+2d for a in [0, max_a)
        idx_a = np.arange(max_a)
        idx_mid = idx_a + d
        idx_end = idx_a + 2 * d
        # Check if all three are set
        all_set = (bits[idx_a] & bits[idx_mid] & bits[idx_end])
        if np.any(all_set):
            found[d] = True
    
    return int(np.sum(found))

def _make_bits(S):
    """Convert a set of integers (1..512) to a numpy bitset array."""
    bits = np.zeros(512, dtype=np.uint8)
    for x in S:
        if 1 <= x <= 512:
            bits[x - 1] = 1
    return bits

def _greedy_init():
    """Greedy construction: add elements one at a time to maximize score."""
    S = set()
    bits = np.zeros(512, dtype=np.uint8)
    
    # Start with a random seed element
    start = random.randint(1, 512)
    S.add(start)
    bits[start - 1] = 1
    
    # Greedily add elements
    for _ in range(31):
        best_x = None
        best_score = -1
        # Sample a subset of candidates to speed up
        candidates = random.sample(range(1, 513), min(100, 512 - len(S)))
        for x in candidates:
            if x in S:
                continue
            bits[x - 1] = 1
            sc = _score_bitset(bits)
            bits[x - 1] = 0
            if sc > best_score:
                best_score = sc
                best_x = x
        if best_x is not None:
            S.add(best_x)
            bits[best_x - 1] = 1
    
    return S, _score_bitset(bits)

def _simulated_annealing(S, start_time, deadline):
    """Simulated annealing on a 32-element set."""
    bits = _make_bits(S)
    current_score = _score_bitset(bits)
    best_score = current_score
    best_S = set(S)
    
    T = 5.0  # initial temperature
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() < deadline:
        if T < T_min:
            break
        
        # Pick a random element to remove
        x = random.choice(list(S))
        # Pick a random element to add
        y = random.randint(1, 512)
        if y == x or y in S:
            continue
        
        # Apply move
        bits[x - 1] = 0
        bits[y - 1] = 1
        S.discard(x)
        S.add(y)
        
        new_score = _score_bitset(bits)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_S = set(S)
        else:
            # Revert
            bits[y - 1] = 0
            bits[x - 1] = 1
            S.discard(y)
            S.add(x)
        
        T *= cooling
    
    return best_S, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_score = 0
    best_S = set()
    
    # Multiple restarts with different strategies
    n_restarts = 8
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Greedy initialization
        S, sc = _greedy_init()
        
        if sc > best_score:
            best_score = sc
            best_S = set(S)
        
        # Simulated annealing refinement
        S2, sc2 = _simulated_annealing(S, start_time, deadline)
        
        if sc2 > best_score:
            best_score = sc2
            best_S = set(S2)
    
    return list(best_S)
>>>>>>> REPLACE
