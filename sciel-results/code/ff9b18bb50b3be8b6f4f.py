"""Additive bases with few representations — structural redesign.

Strategy:
  1. Build an initial valid set A via a greedy "cover-first" construction that
     ensures every target in [128,256] is representable as a sum of two
     elements of A.
  2. Refine with Simulated Annealing on the bitset representation of A,
     using a fast NumPy-based representation-count evaluation.
  3. Deadline guard: return the best valid set found before 85 s.

The objective is to minimize the maximum representation multiplicity over
[128, 256], where a representation of t is an unordered pair (a, b) with
a <= b, a, b in A, and a + b = t.
"""

import time
import random
import numpy as np

# Fixed parameters
N = 128
TARGET_LO = 128
TARGET_HI = 256
TARGETS = list(range(TARGET_LO, TARGET_HI + 1))
TARGET_COUNT = len(TARGETS)  # 129

# Precompute for speed: for each element a in [0,128], which targets does (a,a) cover?
# Actually we need pair sums. Let's precompute the mapping from element index to
# its contribution to each target when paired with each other element.
# But that's 129*129*129 which is fine to compute on the fly with numpy.

def _rep_counts_from_bitset(bitset):
    """
    Given a boolean numpy array of length 129 (index = element value),
    compute for each target in [128,256] the number of unordered pairs
    (a,b) with a<=b, a in A, b in A, a+b=target.
    Returns a numpy array of length 129 (indexed by target - 128).
    """
    # bitset[i] is True iff i is in A
    # For target t, count pairs (a,b), a<=b, a+b=t, a,b in A
    # a ranges from max(0, t-128) to min(128, t//2)
    # For each such a, b = t - a, need b >= a (i.e., a <= t//2) and b in [0,128]
    # and bitset[a] and bitset[b]
    
    counts = np.zeros(TARGET_COUNT, dtype=np.int32)
    for i, t in enumerate(TARGETS):
        lo = max(0, t - 128)
        hi = t // 2
        if lo > hi:
            continue
        # a from lo to hi, b = t - a
        a_arr = np.arange(lo, hi + 1)
        b_arr = t - a_arr
        # b must be in [0,128]
        valid = (b_arr >= 0) & (b_arr <= 128)
        a_valid = a_arr[valid]
        b_valid = b_arr[valid]
        # Check membership
        mask = bitset[a_valid] & bitset[b_valid]
        counts[i] = np.sum(mask)
    return counts


def _max_rep(bitset):
    """Return the maximum representation count over [128,256]."""
    return int(np.max(_rep_counts_from_bitset(bitset)))


def _is_valid(bitset):
    """Check that every target in [128,256] has at least one representation."""
    counts = _rep_counts_from_bitset(bitset)
    return np.all(counts > 0)


def _greedy_construct():
    """
    Greedy construction: start with empty set, add elements to cover all targets.
    At each step, pick the element that covers the most currently-uncovered targets.
    """
    bitset = np.zeros(129, dtype=bool)
    
    # Track which targets are covered
    covered = np.zeros(TARGET_COUNT, dtype=bool)
    
    while not np.all(covered):
        best_elem = -1
        best_gain = -1
        for a in range(129):
            if bitset[a]:
                continue
            # If we add a, which new targets become covered?
            # A target t is newly covered if there exists b in A (or b=a) such that
            # a+b=t and t was not previously covered.
            # Actually, adding a creates new pairs (a, b) for all b in A, and (a,a).
            # A target t is covered if there's any pair summing to t.
            # Let's compute: for each target t, is it currently covered? If not,
            # can adding a make it covered? That happens if there exists b in A ∪ {a}
            # such that a + b = t. Since a is new, the new pairs are (a, b) for b in A
            # and (a, a). So t is newly coverable if t - a is in A or t - a == a (i.e., t=2a).
            
            gain = 0
            for i, t in enumerate(TARGETS):
                if covered[i]:
                    continue
                b = t - a
                if b == a:
                    # pair (a,a)
                    gain += 1
                elif 0 <= b <= 128 and bitset[b]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_elem = a
        if best_elem == -1:
            break
        bitset[best_elem] = True
        # Update covered
        for i, t in enumerate(TARGETS):
            if covered[i]:
                continue
            # Check if t is now covered
            lo = max(0, t - 128)
            hi = t // 2
            found = False
            for a in range(lo, hi + 1):
                b = t - a
                if 0 <= b <= 128 and bitset[a] and bitset[b]:
                    found = True
                    break
            if found:
                covered[i] = True
    
    return bitset


def _sa_refine(bitset, time_limit=80.0):
    """
    Simulated Annealing refinement.
    Moves: add an element, remove an element, or swap (add one, remove another).
    Objective: minimize max rep, with validity as a hard constraint (penalty if invalid).
    """
    def objective(b):
        counts = _rep_counts_from_bitset(b)
        peak = int(np.max(counts))
        # Penalty for uncovered targets
        uncovered = int(np.sum(counts == 0))
        return peak + 100 * uncovered
    
    current = bitset.copy()
    current_obj = objective(current)
    best = current.copy()
    best_obj = current_obj
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Temperature schedule
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Generate a move
        move_type = rng.choice(['add', 'remove', 'swap'])
        new_bitset = current.copy()
        
        if move_type == 'add':
            a = rng.randint(0, 128)
            if not new_bitset[a]:
                new_bitset[a] = True
            else:
                continue
        elif move_type == 'remove':
            a = rng.randint(0, 128)
            if new_bitset[a]:
                new_bitset[a] = False
            else:
                continue
        elif move_type == 'swap':
            a = rng.randint(0, 128)
            b = rng.randint(0, 128)
            if new_bitset[a] and not new_bitset[b]:
                new_bitset[a] = False
                new_bitset[b] = True
            elif not new_bitset[a] and new_bitset[b]:
                new_bitset[a] = True
                new_bitset[b] = False
            else:
                continue
        
        new_obj = objective(new_bitset)
        delta = new_obj - current_obj
        
        if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current = new_bitset
            current_obj = new_obj
            if current_obj < best_obj:
                best = current.copy()
                best_obj = current_obj
        
        T = max(T * cooling, T_min)
    
    return best, best_obj


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Phase 1: Greedy construction
    bitset = _greedy_construct()
    
    # Ensure validity
    if not _is_valid(bitset):
        # Fallback: use the seed program's set
        bitset = np.zeros(129, dtype=bool)
        bitset[64:129] = True
    
    best_bitset = bitset.copy()
    best_obj = objective(best_bitset)
    
    # Phase 2: Simulated Annealing refinement
    sa_time = min(75.0, deadline - time.time())
    if sa_time > 5.0:
        refined, refined_obj = _sa_refine(bitset, time_limit=sa_time)
        if refined_obj < best_obj:
            best_bitset = refined
            best_obj = refined_obj
    
    # Phase 3: Quick local search polish if time remains
    remaining = deadline - time.time()
    if remaining > 2.0:
        # Simple hill climbing: try single flips
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for a in range(129):
                if time.time() >= deadline:
                    break
                trial = best_bitset.copy()
                trial[a] = not trial[a]
                trial_obj = objective(trial)
                if trial_obj < best_obj:
                    best_bitset = trial
                    best_obj = trial_obj
                    improved = True
    
    # Convert to list
    return [i for i in range(129) if best_bitset[i]]


# Helper for objective used in SA
def objective(b):
    counts = _rep_counts_from_bitset(b)
    peak = int(np.max(counts))
    uncovered = int(np.sum(counts == 0))
    return peak + 100 * uncovered
