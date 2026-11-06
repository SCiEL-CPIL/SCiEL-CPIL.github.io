"""
Size-efficient sorting network search for N=18 wires.

Strategy:
  1. Start from a known-good seed network (odd-even transposition).
  2. Iteratively attempt to remove comparators (deletion search), keeping the network
     only if it still sorts all 2^18 Boolean inputs.
  3. Additionally, try local moves: swap two comparators, or replace a comparator
     with a different one, accepting changes that reduce comparator count or maintain
     correctness with fewer comparators.
  4. Use a fast Boolean-input verification via numpy vectorization.
  5. Deadline guard: stop before 60s and return best found.

The key insight: many comparators in a naive network are redundant. By systematically
testing removals and local perturbations, we can prune the network significantly.
"""

import time
import random
import numpy as np

N = 18
NUM_BOOL_INPUTS = 1 << N  # 262144

# Precompute all Boolean inputs as an (N, 2^N) matrix: each column is one input.
# For speed, we represent the state as a 2D numpy array of shape (N, 2^N) with 0/1.
# Comparator (i, j): for each column, set row i = min(row i, row j), row j = max(...).

def _make_all_bool_inputs():
    """Return a numpy array of shape (N, 2^N) where column k is the binary representation of k."""
    # Use bit operations for speed
    indices = np.arange(1 << N, dtype=np.uint8)
    # Build rows: bit b of index k goes to row b
    rows = []
    for b in range(N):
        rows.append(((indices >> b) & 1).astype(np.uint8))
    return np.stack(rows, axis=0)  # shape (N, 2^N)

# Precompute once
_ALL_INPUTS = _make_all_bool_inputs()

def _apply_comparator(state, i, j):
    """Apply comparator (i, j) to state (N, 2^N) uint8 array in-place."""
    a = state[i]
    b = state[j]
    # min goes to i, max goes to j
    # For binary: min(a,b) = a & b, max(a,b) = a | b
    state[i] = a & b
    state[j] = a | b

def _verify_network(comparators):
    """
    Verify that the given list of comparators sorts all 2^N Boolean inputs.
    Returns True if sorted, False otherwise.
    """
    state = _ALL_INPUTS.copy()
    for (i, j) in comparators:
        _apply_comparator(state, i, j)
    # Check sorted: for every column, state[0] <= state[1] <= ... <= state[N-1]
    # i.e., for each consecutive pair, state[k] <= state[k+1] for all k
    # For binary, this means: if state[k+1] == 0 then state[k] must be 0.
    # Equivalently: state[k] & ~state[k+1] must be 0 for all k, columns.
    for k in range(N - 1):
        if np.any(state[k] & ~state[k + 1]):
            return False
    return True

def _count_comparators(comparators):
    return len(comparators)

def _seed_network():
    """Odd-even transposition network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s limit
    
    # Start with seed
    best = _seed_network()
    best_count = len(best)
    
    # Verify seed (should be correct)
    if not _verify_network(best):
        # Fallback: shouldn't happen, but just in case
        return best
    
    # Phase 1: Greedy deletion
    # Try removing each comparator; if still correct, keep it removed.
    # Repeat until no more deletions possible.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        cur = list(best)
        for idx in range(len(cur) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Try removing comparator at idx
            candidate = cur[:idx] + cur[idx+1:]
            if _verify_network(candidate):
                cur = candidate
                improved = True
                best = cur
                best_count = len(best)
        # If we broke out of inner loop due to deadline, break outer too
        if time.time() > deadline:
            break
    
    # Phase 2: Local search - try swapping or replacing comparators
    # Represent network as list of (i, j) pairs
    # Moves:
    #   a) Remove a comparator (already done in phase 1, but maybe combined moves help)
    #   b) Add a comparator (only if it allows further deletions later - skip for now)
    #   c) Replace comparator (i,j) with (i',j')
    #   d) Swap order of two comparators
    #   e) Duplicate a comparator and then delete another (net zero, but might enable further deletion)
    
    # Let's try a simpler local search: random perturbation + deletion
    rng = random.Random(42)
    
    # Try random deletions of pairs (sometimes two deletions together work when one doesn't)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        cur = list(best)
        n = len(cur)
        # Try all pairs of deletions
        for i1 in range(n):
            if time.time() > deadline:
                break
            for i2 in range(i1 + 1, n):
                candidate = cur[:i1] + cur[i1+1:i2] + cur[i2+1:]
                if _verify_network(candidate):
                    cur = candidate
                    improved = True
                    best = cur
                    best_count = len(count_comparators(best)) if False else len(best)
                    break
            if time.time() > deadline or improved:
                break
        if time.time() > deadline:
            break
    
    # Phase 3: Random restarts with mutation
    # Occasionally try: take best, remove k random comparators, re-add if needed
    # But since we can only add back what we removed, let's try:
    # - Take best network
    # - Remove 1-3 random comparators
    # - If still valid, accept
    # - If not, try adding back one of the removed ones in a different position
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        cur = list(best)
        n = len(cur)
        if n < 5:
            break
        
        # Remove 1-3 random comparators
        k = rng.randint(1, min(3, n - 1))
        removed_indices = sorted(rng.sample(range(n), k))
        candidate = []
        removed_comps = []
        for idx in range(n):
            if idx in removed_indices:
                removed_comps.append(cur[idx])
            else:
                candidate.append(cur[idx])
        
        if _verify_network(candidate):
            best = candidate
            best_count = len(best)
            continue
        
        # Try re-adding removed comparators in random positions
        # Only worth doing if we can end up with fewer total comparators
        # Since we removed k and would add back k, net is same. Skip.
        # Instead, try: remove k, add back k-1 (net -1)
        if k >= 2:
            for readd_count in range(1, k):  # add back 1 to k-1
                if time.time() > deadline:
                    break
                # Choose which to re-add
                readd_indices = rng.sample(range(k), readd_count)
                candidate2 = list(candidate)
                for ri in readd_indices:
                    pos = rng.randint(0, len(candidate2))
                    candidate2.insert(pos, removed_comps[ri])
                if len(candidate2) < len(best):
                    if _verify_network(candidate2):
                        best = candidate2
                        best_count = len(best)
    
    # Final: one more pass of single deletions
    improved = True
    while improved and time.time() < deadline:
        improved = False
        cur = list(best)
        for idx in range(len(cur) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = cur[:idx] + cur[idx+1:]
            if _verify_network(candidate):
                cur = candidate
                improved = True
                best = cur
                best_count = len(best)
        if time.time() > deadline:
            break
    
    return best
