"""Additive bases with few representations – structural Max-SAT style design."""
import time
import random
import numpy as np

# Constants
N = 128
TARGET_LO = N          # 128
TARGET_HI = 2 * N      # 256
TARGETS = np.arange(TARGET_LO, TARGET_HI + 1)  # 128..256 inclusive
NUM_TARGETS = TARGET_HI - TARGET_LO + 1        # 129

# Precompute for each target t, the list of unordered pairs (a, b) with a+b=t, a<=b, 0<=a,b<=128
# We'll store as two arrays: pair_a[t_idx], pair_b[t_idx]
def _build_pairs():
    pair_a = []
    pair_b = []
    for t in range(TARGET_LO, TARGET_HI + 1):
        # a ranges from max(0, t-128) to min(128, t//2)
        lo = max(0, t - 128)
        hi = min(128, t // 2)
        for a in range(lo, hi + 1):
            b = t - a
            pair_a.append(a)
            pair_b.append(b)
    return np.array(pair_a, dtype=np.int32), np.array(pair_b, dtype=np.int32)

_pair_a, _pair_b = _build_pairs()

# For each target index, start and end in the pair arrays
_target_start = []
_target_end = []
pos = 0
for t in range(TARGET_LO, TARGET_HI + 1):
    lo = max(0, t - 128)
    hi = min(128, t // 2)
    _target_start.append(pos)
    pos += (hi - lo + 1)
    _target_end.append(pos)
_target_start = np.array(_target_start, dtype=np.int32)
_target_end = np.array(_target_end, dtype=np.int32)

def _compute_mults(A):
    """A is a boolean numpy array of length 129. Returns mult[t] for t in 128..256."""
    mults = np.zeros(NUM_TARGETS, dtype=np.int32)
    for ti in range(NUM_TARGETS):
        s = _target_start[ti]
        e = _target_end[ti]
        a_arr = _pair_a[s:e]
        b_arr = _pair_b[s:e]
        # count pairs where both a and b are in A
        mults[ti] = np.sum(A[a_arr] & A[b_arr])
    return mults

def _score(mults):
    """Score = max multiplicity. Lower is better."""
    return int(np.max(mults))

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    rng = np.random.RandomState(42)
    
    # --- Phase 1: Greedy construction to get a valid cover ---
    # Start with A = {64, 65, ..., 128} (the seed), which covers [128,256].
    # But let's try a smarter greedy: build A to cover all targets with minimal elements.
    
    # Actually, let's use a randomized local search from multiple starts.
    
    best_A = None
    best_score = float('inf')
    
    # Initial solution: the seed A = [64..128]
    A0 = np.zeros(N + 1, dtype=bool)
    A0[64:129] = True
    m0 = _compute_mults(A0)
    s0 = _score(m0)
    best_A = A0.copy()
    best_score = s0
    
    # --- Phase 2: Local search with simulated annealing ---
    # Moves: add/remove/swap elements. Track multiplicities incrementally.
    
    def _delta_remove(A, x, mults):
        """Compute change in mults if we remove x from A. Returns (delta_mults, new_mults_valid)."""
        # x contributes to target t if there exists y in A such that x+y=t and x<=y (unordered)
        # Actually, for unordered pairs (a,b) with a<=b, removing x affects:
        # - pairs where a=x, b=x (if x+x is in target range)
        # - pairs where a=x, b=x' for x' in A, x' >= x
        # - pairs where b=x, a=x' for x' in A, x' <= x
        # Since we count unordered pairs with a<=b, removing x decrements mults[t] for each
        # valid partner y in A where x+y=t.
        
        # For each target t, the pairs involving x are:
        # (x, y) where y = t - x, and we need x <= y (i.e., x <= t-x => t >= 2x) and y <= 128, y >= 0
        # Also (y, x) where y < x is already counted as (y,x) with y<x, so when we remove x,
        # we lose the pair (y,x) for y < x where y+x=t, i.e., y=t-x < x => t < 2x.
        
        # Let's just recompute affected targets.
        delta = np.zeros(NUM_TARGETS, dtype=np.int32)
        for t in range(TARGET_LO, TARGET_HI + 1):
            y = t - x
            if 0 <= y <= 128 and A[y]:
                # This pair (min(x,y), max(x,y)) is currently counted
                # After removing x, it's gone
                delta[t - TARGET_LO] -= 1
        return delta
    
    def _delta_add(A, x, mults):
        """Compute change in mults if we add x to A."""
        delta = np.zeros(NUM_TARGETS, dtype=np.int32)
        for t in range(TARGET_LO, TARGET_HI + 1):
            y = t - x
            if 0 <= y <= 128 and A[y]:
                delta[t - TARGET_LO] += 1
        return delta
    
    # For efficiency, precompute for each x the list of (target_idx, partner) 
    # such that x + partner = target.
    # Actually, let's precompute for each x in [0,128], the list of (ti, y) where y+x in [128,256]
    _x_targets = {}
    for x in range(N + 1):
        lst = []
        for t in range(TARGET_LO, TARGET_HI + 1):
            y = t - x
            if 0 <= y <= 128:
                lst.append((t - TARGET_LO, y))
        _x_targets[x] = lst
    
    def _apply_remove(A, x, mults):
        """Remove x from A, update mults in-place. Returns old A[x] state."""
        if not A[x]:
            return False
        A[x] = False
        for ti, y in _x_targets[x]:
            if A[y]:
                mults[ti] -= 1
        return True
    
    def _apply_add(A, x, mults):
        """Add x to A, update mults in-place. Returns whether it was added."""
        if A[x]:
            return False
        A[x] = True
        for ti, y in _x_targets[x]:
            if A[y]:
                mults[ti] += 1
        return True
    
    # Local search
    current_A = best_A.copy()
    current_mults = _compute_mults(current_A).copy()
    current_score = best_score
    
    T = 50.0  # initial temperature
    T_min = 0.1
    cooling = 0.999
    
    step = 0
    while time.time() < deadline and T > T_min:
        step += 1
        x = rng.randint(0, N)
        
        if current_A[x]:
            # Try removing x
            old_mults = current_mults.copy()
            _apply_remove(current_A, x, current_mults)
            new_score = _score(current_mults)
            # Check validity: all mults > 0
            valid = np.all(current_mults > 0)
            
            if valid and new_score <= current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_A = current_A.copy()
            else:
                # Accept with probability exp(-delta/T) if valid, else reject
                if valid:
                    delta = new_score - current_score
                    if delta < 0 or rng.random() < np.exp(-delta / T):
                        current_score = new_score
                    else:
                        # Reject
                        current_A[x] = True
                        current_mults = old_mults
                else:
                    # Invalid: reject
                    current_A[x] = True
                    current_mults = old_mults
        else:
            # Try adding x
            old_mults = current_mults.copy()
            _apply_add(current_A, x, current_mults)
            new_score = _score(current_mults)
            valid = np.all(current_mults > 0)
            
            if valid and new_score <= current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_A = current_A.copy()
            else:
                if valid:
                    delta = new_score - current_score
                    if delta < 0 or rng.random() < np.exp(-delta / T):
                        current_score = new_score
                    else:
                        current_A[x] = False
                        current_mults = old_mults
                else:
                    current_A[x] = False
                    current_mults = old_mults
        
        T *= cooling
    
    # --- Phase 3: Targeted local search on bottleneck sums ---
    # Focus on the targets with highest multiplicity, try to reduce them.
    current_A = best_A.copy()
    current_mults = _compute_mults(current_A).copy()
    current_score = best_score
    
    T2 = 20.0
    T2_min = 0.05
    cooling2 = 0.9995
    
    while time.time() < deadline and T2 > T2_min:
        # Find the target with max multiplicity
        max_ti = np.argmax(current_mults)
        max_mult = current_mults[max_ti]
        
        # Try to reduce this target by removing an element involved in its pairs
        t_val = TARGET_LO + max_ti
        # Get all pairs for this target
        s = _target_start[max_ti]
        e = _target_end[max_ti]
        a_arr = _pair_a[s:e]
        b_arr = _pair_b[s:e]
        
        # Find elements in A that are part of pairs for this target
        candidates = set()
        for i in range(len(a_arr)):
            a = a_arr[i]
            b = b_arr[i]
            if current_A[a]:
                candidates.add(a)
            if current_A[b]:
                candidates.add(b)
        
        if not candidates:
            T2 *= cooling2
            continue
        
        x = list(candidates)[rng.randint(0, len(candidates))]
        
        # Try removing x
        old_mults = current_mults.copy()
        _apply_remove(current_A, x, current_mults)
        new_score = _score(current_mults)
        valid = np.all(current_mults > 0)
        
        if valid and new_score <= current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = current_A.copy()
        else:
            if valid:
                delta = new_score - current_score
                if delta < 0 or rng.random() < np.exp(-delta / T2):
                    current_score = new_score
                else:
                    current_A[x] = True
                    current_mults = old_mults
            else:
                current_A[x] = True
                current_mults = old_mults
        
        # Also try swap: remove x, add y
        if rng.random() < 0.5:
            y = rng.randint(0, N)
            if y != x and not current_A[y]:
                old_mults2 = current_mults.copy()
                # Remove x (if still in A)
                if current_A[x]:
                    _apply_remove(current_A, x, current_mults)
                _apply_add(current_A, y, current_mults)
                new_score2 = _score(current_mults)
                valid2 = np.all(current_mults > 0)
                
                if valid2 and new_score2 <= current_score:
                    current_score = new_score2
                    if new_score2 < best_score:
                        best_score = new_score2
                        best_A = current_A.copy()
                else:
                    if valid2:
                        delta2 = new_score2 - current_score
                        if delta2 < 0 or rng.random() < np.exp(-delta2 / T2):
                            current_score = new_score2
                        else:
                            # Undo
                            current_A[y] = False
                            current_A[x] = True
                            current_mults = old_mults2
                    else:
                        current_A[y] = False
                        current_A[x] = True
                        current_mults = old_mults2
        
        T2 *= cooling2
    
    # Return the best solution found
    result = [i for i in range(N + 1) if best_A[i]]
    if not result:
        # Fallback to seed
        result = list(range(64, 129))
    return result
