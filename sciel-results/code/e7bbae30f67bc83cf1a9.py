"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random
import numpy as np

N = 256
MAX_SUM = 2 * N  # 512

# EVOLVE-BLOCK-START

def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set using bitset."""
    arr = np.array(selected)
    sums = arr[:, None] + arr[None, :]
    # Only check upper triangle including diagonal
    iu = np.triu_indices(len(arr))
    upper_sums = sums[iu]
    return len(np.unique(upper_sums)) == len(upper_sums)

def _is_maximal_np(selected):
    """Check if selected is inclusion-maximal in [1, N] using vectorized NumPy."""
    arr = np.array(selected, dtype=np.int32)
    n = len(arr)
    if n == 0:
        return False
    
    # Build sum bitset: sums[i] = 1 if i is a pair sum
    sums = arr[:, None] + arr[None, :]
    iu = np.triu_indices(n)
    upper_sums = sums[iu]
    sum_bitset = np.zeros(MAX_SUM + 1, dtype=np.int8)
    sum_bitset[upper_sums] = 1
    
    sel_set = set(selected)
    
    # Vectorized check: for each candidate x not in selected,
    # check if x can be added (i.e., no conflict with existing sums)
    candidates = np.array([x for x in range(1, N + 1) if x not in sel_set], dtype=np.int32)
    if len(candidates) == 0:
        return True
    
    # For each candidate x, it conflicts if:
    # 1) 2*x is in sum_bitset
    # 2) x + a is in sum_bitset for any a in selected
    # 3) x + x conflicts with x + a for a != x (but x not in selected so this is just 2x)
    
    # Check condition 1: 2*x in sums
    double_x = 2 * candidates
    cond1 = sum_bitset[double_x] > 0
    
    # Check condition 2: (x + a) in sums for some a
    # candidates shape: (m,), arr shape: (n,)
    # x_plus_a shape: (m, n)
    x_plus_a = candidates[:, None] + arr[None, :]
    # Check if any column has a sum in sum_bitset
    # sum_bitset[x_plus_a] shape: (m, n), we want any > 0 per row
    in_sums = sum_bitset[x_plus_a]
    cond2 = in_sums.any(axis=1)
    
    # Also need to check: among the new sums (x+a for a in selected), 
    # no two are equal to each other. But since x is fixed and a's are distinct,
    # x+a1 == x+a2 implies a1==a2, so this is automatic.
    # However, we also need x+a not equal to 2*x, which is covered by cond1 
    # (if 2x in sums, blocked; if 2x not in sums, then x+a != 2x for all a 
    #  since a != x and a in selected, so x+a != 2x means a != x which is true).
    # Wait: x + a = 2*x iff a = x, but x is not in selected, so x+a != 2x always.
    # So the only conflicts are: 2x in existing sums, or x+a in existing sums.
    
    can_add = ~(cond1 | cond2)
    return not can_add.any()

def _greedy_np(order):
    """Greedy construction using NumPy for conflict checking."""
    selected = []
    sum_bitset = np.zeros(MAX_SUM + 1, dtype=np.int8)
    
    for x in order:
        x = int(x)
        # Check 2*x
        if sum_bitset[2 * x]:
            continue
        # Check x + a for all a in selected
        if selected:
            arr_sel = np.array(selected, dtype=np.int32)
            new_sums = x + arr_sel
            if sum_bitset[new_sums].any():
                continue
        # Add x
        selected.append(x)
        sum_bitset[2 * x] = 1
        if selected:
            arr_sel = np.array(selected[:-1], dtype=np.int32)
            sum_bitset[x + arr_sel] = 1
    
    return selected

def _prune_np(S, deadline):
    """Try to remove elements while maintaining maximality, using fast check."""
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline:
                break
            new_S = S[:i] + S[i+1:]
            if not new_S:
                continue
            if _is_maximal_np(new_S):
                S = new_S
                improved = True
    return S

def _construct_block_greedy(rng, deadline):
    """
    Structural approach: partition [1,N] into blocks, and within each block,
    use a randomized greedy that prioritizes elements that block many others.
    This is a different search family than full random permutation.
    """
    # Divide into blocks of size B
    B = 16
    blocks = []
    for start in range(1, N + 1, B):
        end = min(start + B - 1, N)
        block = list(range(start, end + 1))
        rng.shuffle(block)
        blocks.append(block)
    
    # Randomize block order
    rng.shuffle(blocks)
    
    # Flatten to get an order
    order = []
    for block in blocks:
        order.extend(block)
    
    return _greedy_np(order)

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 55.0  # safety margin before 60s limit
    
    best_size = N + 1
    best_set = None
    
    # Phase 1: Ascending greedy baseline
    S0 = _greedy_np(range(1, N + 1))
    S0 = _prune_np(S0, deadline)
    if len(S0) < best_size:
        best_size = len(S0)
        best_set = list(S0)
    
    # Phase 2: Block-greedy with randomization (structural: block-based ordering)
    rng = random.Random(123)
    
    for trial in range(20000):
        if time.time() > deadline:
            break
        
        S = _construct_block_greedy(rng, deadline)
        S = _prune_np(S, deadline)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
        
        # Early termination if we find a very small set
        if best_size <= 12:
            break
    
    # Phase 3: Local search - try removing elements and checking maximality
    if best_set is not None and time.time() < deadline - 3:
        S = list(best_set)
        # Try all single removals
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline - 1:
                break
            new_S = S[:i] + S[i+1:]
            if new_S and _is_maximal_np(new_S):
                S = new_S
                # Continue trying more removals
                i = len(S) - 1
                while i >= 0:
                    if time.time() > deadline - 1:
                        break
                    new_S2 = S[:i] + S[i+1:]
                    if new_S2 and _is_maximal_np(new_S2):
                        S = new_S2
                    i -= 1
                best_set = list(S)
                best_size = len(S)
    
    # Final verification and repair
    if best_set is None:
        S0 = _greedy_np(range(1, N + 1))
        S0 = _prune_np(S0, deadline)
        best_set = S0
    
    # Ensure it's a valid Sidon set
    S_sorted = sorted(best_set)
    if not _is_valid_sidon(S_sorted):
        S0 = _greedy_np(range(1, N + 1))
        S0 = _prune_np(S0, deadline)
        return S0
    
    # Check maximality and repair if needed
    if not _is_maximal_np(S_sorted):
        changed = True
        while changed:
            changed = False
            S_set = set(S_sorted)
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                # Check if x can be added
                if not _is_maximal_np(S_sorted + [x]):
                    # x cannot be added, skip
                    continue
                # Actually, if x CAN be added, then the set is not maximal
                # We need to add x to make it maximal
                S_sorted.append(x)
                S_sorted.sort()
                S_set.add(x)
                changed = True
                break
        best_set = S_sorted
    
    return best_set

# EVOLVE-BLOCK-END
