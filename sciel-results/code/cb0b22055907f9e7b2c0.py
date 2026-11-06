"""Find a large subset of [1,256] where no distinct pair a,b has (a+b) | (a*b)."""
import time
import random
from itertools import combinations

def _conflict(a, b):
    """Return True if (a+b) divides (a*b) for distinct a,b."""
    s = a + b
    if s == 0:
        return False
    return (a * b) % s == 0

def _build_conflict_bitmasks(n=256):
    """Precompute conflict masks as integers (bit i set means element i+1 conflicts with this)."""
    masks = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if _conflict(a, b):
                masks[a] |= (1 << b)
                masks[b] |= (1 << a)
    return masks

def _popcount(x):
    return bin(x).count('1')

def _greedy_coloring(candidate_mask, masks):
    """
    Greedy coloring of the subgraph induced by candidate_mask.
    Returns (color_classes, max_color) where color_classes is a list of masks.
    Each vertex gets the smallest color not used by its neighbors in candidate.
    """
    # Order candidates by degree (highest first) for tighter coloring
    candidates = []
    m = candidate_mask
    while m:
        lsb = m & (-m)
        idx = lsb.bit_length() - 1
        deg = _popcount(masks[idx] & candidate_mask)
        candidates.append((deg, idx))
        m ^= lsb
    candidates.sort(reverse=True)
    
    color_of = {}
    color_classes = []
    for deg, idx in candidates:
        # Find smallest color not used by neighbors
        used = set()
        for c_mask in color_classes:
            if (masks[idx] & c_mask) != 0:
                c = color_classes.index(c_mask)
                used.add(c)
        # Find smallest available color
        c = 0
        while c in used:
            c += 1
        color_of[idx] = c
        if c >= len(color_classes):
            color_classes.append(0)
        color_classes[c] |= (1 << idx)
    
    return color_classes, len(color_classes)

def _branch_and_bound(candidate_mask, masks, deadline, best_size_ref):
    """
    Tomita-style branch and bound for maximum independent set.
    candidate_mask: bitmask of vertices still available
    masks: list of conflict masks (bit i set means conflict with element i+1)
    Returns the best independent set found as a bitmask.
    """
    best_set_mask = [0]
    best_size = [0]
    
    def _bound(color_classes):
        # Upper bound: current_size + max color class size
        return _popcount(color_classes[-1]) if color_classes else 0
    
    def _greedy_independent_set(cand_mask):
        """Greedy: repeatedly pick vertex with fewest conflicts in cand."""
        result = 0
        m = cand_mask
        while m:
            # Pick vertex with minimum degree in remaining
            best_v = -1
            best_deg = float('inf')
            mm = m
            while mm:
                lsb = mm & (-mm)
                idx = lsb.bit_length() - 1
                deg = _popcount(masks[idx] & m)
                if deg < best_deg:
                    best_deg = deg
                    best_v = idx
                mm ^= lsb
            result |= (1 << best_v)
            m &= ~masks[best_v]
            m &= ~(1 << best_v)
        return result
    
    def _bbs(cand_mask, current_size, current_mask):
        if cand_mask == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_set_mask[0] = current_mask
            return
        
        # Prune if upper bound <= best
        color_classes, max_color = _greedy_coloring(cand_mask, masks)
        if current_size + max_color <= best_size[0]:
            return
        
        # Order candidates by color (process highest color first)
        # Build ordered list
        ordered = []
        for c in range(max_color - 1, -1, -1):
            if c < len(color_classes):
                cm = color_classes[c]
                mm = cm
                while mm:
                    lsb = mm & (-mm)
                    idx = lsb.bit_length() - 1
                    ordered.append(idx)
                    mm ^= lsb
        
        for idx in ordered:
            if time.time() >= deadline:
                return
            bit = 1 << idx
            if not (cand_mask & bit):
                continue
            # Include idx
            _bbs(cand_mask & ~masks[idx] & ~(bit), current_size + 1, current_mask | bit)
            # Exclude idx
            cand_mask &= ~bit
            # Update bound
            color_classes, max_color = _greedy_coloring(cand_mask, masks)
            if current_size + max_color <= best_size[0]:
                return
    
    _bbs(candidate_mask, 0, 0)
    return best_set_mask[0]

def _local_search_bitmask(initial_mask, masks, time_limit=2.0):
    """Local search using bitmask operations for speed."""
    current = initial_mask
    best = initial_mask
    best_size = _popcount(initial_mask)
    start = time.time()
    deadline = start + time_limit
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element
        to_try = []
        m = current
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            to_try.append(idx)
            m ^= lsb
        random.shuffle(to_try)
        
        for rem in to_try:
            if time.time() >= deadline:
                break
            rem_bit = 1 << rem
            trial = current & ~rem_bit
            # Try adding each non-current element
            avail = ((1 << 257) - 2) & ~trial  # elements 1..256 not in trial
            mm = avail
            while mm:
                lsb = mm & (-mm)
                idx = lsb.bit_length() - 1
                # Check if idx conflicts with any in trial
                if (masks[idx] & trial) == 0:
                    new_set = trial | lsb
                    if _popcount(new_set) > _popcount(current):
                        current = new_set
                        improved = True
                        if _popcount(current) > best_size:
                            best = current
                            best_size = best_size
                    break
                mm ^= lsb
    
    return best

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin before 90s kill
    
    # Precompute conflict masks
    masks = _build_conflict_bitmasks(256)
    
    # Full candidate mask: elements 1..256
    full_mask = ((1 << 257) - 2)  # bits 1 through 256 set
    
    # Initial: all odd numbers
    odd_mask = 0
    for i in range(1, 257, 2):
        odd_mask |= (1 << i)
    
    best_mask = odd_mask
    best_size = _popcount(odd_mask)
    
    # Try branch-and-bound on the full problem
    # This is expensive, so we time-box it
    if time.time() < deadline - 10:
        bnb_deadline = time.time() + 40
        bnb_result = _branch_and_bound(full_mask, masks, bnb_deadline, best_size)
        if _popcount(bnb_result) > best_size:
            best_mask = bnb_result
            best_size = _popcount(bnb_result)
    
    # Local search to potentially improve further
    if time.time() < deadline - 5:
        ls_deadline = min(time.time() + 10, deadline - 2)
        ls_result = _local_search_bitmask(best_mask, masks, time_limit=ls_deadline - time.time())
        if _popcount(ls_result) > best_size:
            best_mask = ls_result
            best_size = _popcount(ls_result)
    
    # Greedy completion: try to add any remaining elements
    if time.time() < deadline - 3:
        remaining = full_mask & ~best_mask
        mm = remaining
        while mm:
            lsb = mm & (-mm)
            idx = lsb.bit_length() - 1
            if (masks[idx] & best_mask) == 0:
                best_mask |= lsb
                best_size += 1
            mm ^= lsb
    
    # Convert to list
    result = []
    m = best_mask
    while m:
        lsb = m & (-m)
        idx = lsb.bit_length() - 1
        result.append(idx)
        m ^= lsb
    result.sort()
    
    # Final validation
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if _conflict(a, b):
                valid = False
                break
        if not valid:
            break
    
    if not valid or len(result) == 0:
        return list(range(1, 257, 2))
    
    return result

# EVOLVE-BLOCK-END
