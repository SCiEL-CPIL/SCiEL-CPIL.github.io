"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt
from functools import lru_cache
import numpy as np

N_MAX = 256
NUM_CANDIDATES = N_MAX - 1  # 255 candidates: 2..256

def _get_primes(n):
    """Return list of primes up to n."""
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def _can_add(S, x):
    """
    Check if x can be added to set S.
    S is a frozenset of ints for faster hashing.
    """
    s_list = list(S)
    n = len(s_list)
    for i in range(n):
        a = s_list[i]
        for j in range(i + 1, n):
            b = s_list[j]
            # Check x | a * b
            if (a * b) % x == 0:
                return False
            # Check a | x * b
            if (x * b) % a == 0:
                return False
            # Check b | x * a
            if (x * a) % b == 0:
                return False
    return True

def _is_valid_set(S):
    """Verify the full 3-wise constraint for set S."""
    s_list = list(S)
    n = len(s_list)
    for i in range(n):
        a = s_list[i]
        for j in range(i + 1, n):
            b = s_list[j]
            for k in range(j + 1, n):
                c = s_list[k]
                if (b * c) % a == 0:
                    return False
                if (a * c) % b == 0:
                    return False
                if (a * b) % c == 0:
                    return False
    return True

def _build_conflict_graph():
    """
    Build conflict graph where edge (i,j) exists if there exists
    a third candidate z (different from i,j) such that
    candidates[i] | candidates[j]*candidates[z] or candidates[j] | candidates[i]*candidates[z].
    
    Returns: (candidates, conflict_masks)
      candidates: list of ints 2..256
      conflict_masks: list of numpy uint64 bitmasks, one per candidate
    """
    candidates = list(range(2, N_MAX + 1))
    n = len(candidates)
    # For fast lookup: value -> index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    conflict_masks = [0] * n
    
    for i in range(n):
        a = candidates[i]
        for j in range(i + 1, n):
            b = candidates[j]
            has_conflict = False
            for z in range(n):
                if z == i or z == j:
                    continue
                c = candidates[z]
                if (b * c) % a == 0 or (a * c) % b == 0:
                    has_conflict = True
                    break
            if has_conflict:
                conflict_masks[i] |= (1 << j)
                conflict_masks[j] |= (1 << i)
    
    return candidates, conflict_masks

def _branch_and_bound(candidates, conflict_masks, deadline):
    """
    Maximum independent set via branch and bound with bitset operations.
    candidates: list of candidate values
    conflict_masks: list of int bitmasks for conflicts
    deadline: time limit
    Returns best set as list of candidate values.
    """
    n = len(candidates)
    
    # Convert to numpy uint64 for fast bit operations
    # Since n <= 255, we need 4 uint64 words (256 bits)
    num_words = (n + 63) // 64
    
    # conflict_masks_np[i] is a tuple of num_words uint64 values
    cm_np = [None] * n
    for i in range(n):
        words = []
        for w in range(num_words):
            mask = (conflict_masks[i] >> (w * 64)) & 0xFFFFFFFFFFFFFFFF
            words.append(np.uint64(mask))
        cm_np[i] = words
    
    best_set = []
    best_size = 0
    
    # Upper bound helper: greedy coloring of remaining candidates
    def _upper_bound(remaining_mask):
        """
        Greedy coloring bound for maximum independent set.
        remaining_mask: list of uint64 words representing which candidates are available.
        Returns upper bound on independent set size from remaining.
        """
        # Extract list of available indices
        avail = []
        for w in range(num_words):
            m = remaining_mask[w]
            if m:
                # Iterate set bits
                mm = int(m)
                while mm:
                    lb = mm & (-mm)
                    bit = lb.bit_length() - 1
                    avail.append(w * 64 + bit)
                    mm ^= lb
        
        if not avail:
            return 0
        
        # Greedy coloring: assign colors to vertices, bound = max color
        # We process in order and assign smallest possible color
        color_of = {}
        max_color = 0
        for v in avail:
            # Find which colors are used by neighbors of v that are already colored
            used_colors = set()
            v_conf = cm_np[v]
            for u in avail:
                if u >= v:
                    break
                if u not in color_of:
                    continue
                # Check if u conflicts with v
                conflict = False
                for w in range(num_words):
                    if v_conf[w] & (1 << u):
                        conflict = True
                        break
                if conflict:
                    used_colors.add(color_of[u])
            # Assign smallest available color
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            if c > max_color:
                max_color = c
        
        return max_color + 1
    
    def _popcount64(m):
        return bin(int(m)).count('1')
    
    def _remaining_count(mask):
        return sum(_popcount64(m) for m in mask)
    
    # Initial full mask
    full_mask = [np.uint64(0)] * num_words
    for w in range(num_words):
        if w < num_words - 1:
            full_mask[w] = np.uint64(0xFFFFFFFFFFFFFFFF)
        else:
            bits_in_last = n - (num_words - 1) * 64
            if bits_in_last < 64:
                full_mask[w] = np.uint64((1 << bits_in_last) - 1)
            else:
                full_mask[w] = np.uint64(0xFFFFFFFFFFFFFFFF)
    
    # Order candidates by degree (descending) for better pruning
    degrees = [_remaining_count(cm_np[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    # We'll work with a permutation: new index -> old index
    # Actually, let's just use the original indexing but pick vertices in degree order
    
    def _solve(remaining, current_set, current_size):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set[:]
        
        if not remaining:
            return
        
        rem_count = _remaining_count(remaining)
        if current_size + rem_count <= best_size:
            return
        
        # Pick vertex with highest degree in remaining subgraph
        # For efficiency, just pick first available
        pick = -1
        for w in range(num_words):
            m = int(remaining[w])
            if m:
                lb = m & (-m)
                pick = w * 64 + lb.bit_length() - 1
                break
        
        if pick < 0 or pick >= n:
            return
        
        # Branch 1: include pick
        new_remaining = [np.uint64(0)] * num_words
        pick_conf = cm_np[pick]
        for w in range(num_words):
            new_remaining[w] = remaining[w] & ~pick_conf[w]
        # Also remove pick itself
        word_idx = pick // 64
        bit_idx = pick % 64
        new_remaining[word_idx] &= ~(np.uint64(1) << bit_idx)
        
        _solve(new_remaining, current_set + [candidates[pick]], current_size + 1)
        
        if time.time() > deadline:
            return
        
        # Branch 2: exclude pick
        new_remaining2 = [np.uint64(r) for r in remaining]
        new_remaining2[word_idx] &= ~(np.uint64(1) << bit_idx)
        
        _solve(new_remaining2, current_set, current_size)
    
    # Simple initial solution: greedy
    greedy_set = []
    rem = [np.uint64(r) for r in full_mask]
    while _remaining_count(rem) > 0:
        if time.time() > deadline:
            break
        # Pick vertex with minimum degree in remaining
        best_v = -1
        best_deg = float('inf')
        for w in range(num_words):
            m = int(rem[w])
            if not m:
                continue
            mm = m
            while mm:
                lb = mm & (-mm)
                v = w * 64 + lb.bit_length() - 1
                if v >= n:
                    break
                # Count degree in remaining
                deg = 0
                vc = cm_np[v]
                for w2 in range(num_words):
                    deg += _popcount64(rem[w2] & vc[w2])
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                mm ^= lb
        if best_v < 0:
            break
        greedy_set.append(candidates[best_v])
        # Remove best_v and its neighbors
        word_idx = best_v // 64
        bit_idx = best_v % 64
        rem[word_idx] &= ~(np.uint64(1) << bit_idx)
        vc = cm_np[best_v]
        for w in range(num_words):
            rem[w] &= ~vc[w]
    
    if greedy_set:
        best_set = greedy_set[:]
        best_size = len(greedy_set)
    
    # Now run branch and bound
    _solve(full_mask, [], 0)
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    n = N_MAX
    all_candidates = list(range(2, n + 1))
    
    # Build conflict graph
    candidates, conflict_masks = _build_conflict_graph()
    
    # Get best solution from branch and bound
    best_S = set(_branch_and_bound(candidates, conflict_masks, deadline))
    
    # If B&B didn't find anything or timed out badly, use fallback
    if not best_S:
        # Fallback: upper half
        best_S = set(range(129, n + 1))
    
    # Try to improve with local search if time permits
    if time.time() < deadline - 5:
        # Try adding elements
        for x in all_candidates:
            if time.time() > deadline:
                break
            if x not in best_S:
                if _can_add(frozenset(best_S), x):
                    best_S.add(x)
    
    # Final verification
    if not _is_valid_set(best_S):
        # Fallback: use primes only
        best_S = set(_get_primes(n))
    
    return [int(x) for x in best_S]

# EVOLVE-BLOCK-END
