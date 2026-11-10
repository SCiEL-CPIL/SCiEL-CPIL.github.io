import time
import random
import itertools

# EVOLVE-BLOCK-START

N = 12
M = 40
N_MASKS = 1 << N  # 4096

# Precompute comparability: for each mask, list of masks that are strict supersets
# and strict subsets. We'll use bit operations for speed.
# For a mask A, B is a strict superset of A iff (A & B) == A and A != B
# i.e., (A | B) == B and A != B

# Precompute for each mask, the set of masks it is a strict subset of (supersets)
# and the set of masks that are strict subsets of it (subsets)
# We'll store these as lists of indices for fast lookup.

# Actually, for efficiency, let's precompute a bitset or list for each mask.
# Since 4096 is small, we can precompute comparability matrix as a list of lists.

# Let's precompute: for each mask i, comp_sup[i] = list of masks j such that i is strict subset of j
# and comp_sub[i] = list of masks j such that j is strict subset of i

# But storing all pairs might be memory intensive. Let's think differently.
# For a given set of 40 masks, the number of comparable pairs is:
# sum over i<j of [i is strict subset of j or j is strict subset of i]

# We can compute this by, for each pair, checking if one is subset of other.
# With 40 masks, that's 780 pairs, and each check is O(1) with bit ops.
# So computing the score is fast.

# For local search / simulated annealing, we need to evaluate delta quickly.
# When we swap mask i with mask new_mask, the change in score depends on:
# - pairs involving i (removed)
# - pairs involving new_mask (added)
# - pairs between i and new_mask (if both in set, but they replace each other)

# Let's precompute for each mask, its comparability with all other masks.
# comp[i][j] = 1 if i is strict subset of j, 0 otherwise (for i != j)
# This is a 4096x4096 matrix, which is 16M entries. As a list of bytes, that's 16MB. OK.

# Actually, let's use a more memory-efficient approach.
# For each mask i, store a list of masks that are comparable with i (either direction).
# But for delta computation, we need to know specifically which direction.

# Let's precompute:
# sup_count[i] = number of masks j (j != i) such that i is strict subset of j
# sub_count[i] = number of masks j (j != i) such that j is strict subset of i
# And for delta, when adding new_mask to the set (replacing old_mask):
# delta = (comparability of new_mask with other 39 masks) - (comparability of old_mask with other 39 masks)
# where comparability means one is strict subset of the other.

# Let's precompute for each mask i, a bitset of all masks j such that i is strict subset of j or j is strict subset of i.
# Then comparability count of i with a set S is popcount(bitset[i] & bitset_of_S).

# With Python ints as bitsets, this is very fast.

# Precompute comp_bitset[i] = bitmask over 4096 masks, where bit j is set if i and j are comparable (one strict subset of other)

def _precompute():
    """Precompute comparability bitsets for all 4096 masks."""
    comp = [0] * N_MASKS
    for i in range(N_MASKS):
        bits = 0
        for j in range(N_MASKS):
            if i == j:
                continue
            # Check if i is strict subset of j or j is strict subset of i
            if (i | j) == j and i != j:
                # i is strict subset of j
                bits |= (1 << j)
            elif (i | j) == i and j != i:
                # j is strict subset of i
                bits |= (1 << j)
        comp[i] = bits
    return comp

# We'll do this lazily or at module load. But 4096^2 = 16M checks, each O(1).
# In Python, this might take a few seconds. Let's time it.

# Alternative: precompute more cleverly.
# For mask i, the masks j such that i is strict subset of j are those j where (i|j)==j and j!=i.
# These are all j that contain i as a subset, except i itself.
# The number of such j is 2^(N - popcount(i)) - 1.
# Similarly for j strict subset of i: 2^popcount(i) - 1.

# But we need the actual bitset, not just the count.

# Let's just do the O(N_MASKS^2) precomputation. 16M iterations in Python might be ~10-20 seconds.
# That's too slow. Let's optimize.

# Better approach: for each i, generate all supersets and subsets directly.
def _precompute_fast():
    """Precompute comparability bitsets using subset/superset enumeration."""
    comp = [0] * N_MASKS
    for i in range(N_MASKS):
        bits = 0
        # All strict supersets of i
        # A strict superset j of i satisfies: (i|j)==j and j!=i
        # Equivalently, j = i | s where s is a non-empty subset of (~i) & ((1<<N)-1)
        complement = ((1 << N) - 1) ^ i
        # Enumerate all non-empty subsets of complement
        s = complement
        while s:
            j = i | s
            bits |= (1 << j)
            s = (s - 1) & complement
        # All strict subsets of i
        # A strict subset j of i satisfies: (i|j)==i and j!=i
        # Equivalently, j is a proper subset of i
        s = i
        while s:
            j = s
            bits |= (1 << j)
            s = (s - 1) & i
        comp[i] = bits
    return comp

# This is still O(3^N) per mask in the worst case... no wait.
# For each i, the number of supersets is 2^(N-popcount(i)) and subsets is 2^popcount(i).
# Sum over all i of 2^(N-popcount(i)) + 2^popcount(i) = 3^N + 3^N = 2*3^N.
# 3^12 = 531441. So total operations ~1M. That's fast!

# Let's use this.

_comp_bitsets = None

def _get_comp_bitsets():
    global _comp_bitsets
    if _comp_bitsets is None:
        _comp_bitsets = _precompute_fast()
    return _comp_bitsets

def _score(family):
    """Compute number of comparable pairs in family (list of 40 masks)."""
    comp = _get_comp_bitsets()
    # Build bitset of family
    fam_bits = 0
    for m in family:
        fam_bits |= (1 << m)
    # For each mask in family, count comparable masks in family (excluding itself)
    total = 0
    for m in family:
        c = comp[m] & fam_bits
        total += bin(c).count('1')
    # Each pair is counted twice (once from each endpoint), so divide by 2
    return total // 2

def _greedy_init():
    """Greedy construction: start with empty, add mask that maximizes marginal gain."""
    comp = _get_comp_bitsets()
    family = []
    fam_bits = 0
    # Start with a good initial set. Let's try to start with all small sets.
    # Actually, let's just do pure greedy.
    # For efficiency, we'll maintain current score and update.
    current_score = 0
    for _ in range(M):
        best_mask = -1
        best_gain = -1
        for m in range(N_MASKS):
            if fam_bits & (1 << m):
                continue
            # Gain = number of masks in family comparable with m
            gain = bin(comp[m] & fam_bits).count('1')
            if gain > best_gain:
                best_gain = gain
                best_mask = m
        if best_mask < 0:
            break
        family.append(best_mask)
        fam_bits |= (1 << best_mask)
        current_score += best_gain
    return family, current_score

def _local_search(family, max_iters=5000, time_limit=10):
    """Local search: try swapping one mask with another to improve score."""
    comp = _get_comp_bitsets()
    fam_set = set(family)
    fam_bits = 0
    for m in family:
        fam_bits |= (1 << m)
    
    current_score = _score(family)
    best_family = family[:]
    best_score = current_score
    
    start_time = time.time()
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        # Try random swap
        i = random.randint(0, M - 1)
        old_mask = family[i]
        # Pick a new mask not in family
        while True:
            new_mask = random.randint(0, N_MASKS - 1)
            if new_mask not in fam_set:
                break
        
        # Compute delta
        # Remove old_mask, add new_mask
        # Pairs affected: those involving old_mask or new_mask
        # Delta = (comparability of new_mask with family\{old_mask}) - (comparability of old_mask with family\{old_mask})
        # + (1 if old_mask and new_mask are comparable else 0) [but they're not both in family after swap]
        # Actually:
        # Old contribution of old_mask: count of masks in family\{old_mask} comparable with old_mask
        # New contribution of new_mask: count of masks in family\{old_mask} comparable with new_mask
        # The pair (old_mask, new_mask) is not in family after swap, so no change there.
        
        fam_without_old_bits = fam_bits & ~(1 << old_mask)
        
        old_contrib = bin(comp[old_mask] & fam_without_old_bits).count('1')
        new_contrib = bin(comp[new_mask] & fam_without_old_bits).count('1')
        
        delta = new_contrib - old_contrib
        
        if delta > 0:
            # Accept
            family[i] = new_mask
            fam_set.remove(old_mask)
            fam_set.add(new_mask)
            fam_bits = (fam_bits & ~(1 << old_mask)) | (1 << new_mask)
            current_score += delta
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_family = family[:]
        else:
            no_improve += 1
            if no_improve > 500:
                break
    
    return best_family, best_score

def _simulated_annealing(family, time_limit=30):
    """Simulated annealing with bitset-based delta calculation."""
    comp = _get_comp_bitsets()
    fam_set = set(family)
    fam_bits = 0
    for m in family:
        fam_bits |= (1 << m)
    
    current_score = _score(family)
    best_family = family[:]
    best_score = current_score
    
    start_time = time.time()
    T = 10.0  # initial temperature
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        T *= cooling
        if T < 0.01:
            break
        
        i = random.randint(0, M - 1)
        old_mask = family[i]
        while True:
            new_mask = random.randint(0, N_MASKS - 1)
            if new_mask not in fam_set:
                break
        
        fam_without_old_bits = fam_bits & ~(1 << old_mask)
        
        old_contrib = bin(comp[old_mask] & fam_without_old_bits).count('1')
        new_contrib = bin(comp[new_mask] & fam_without_old_bits).count('1')
        
        delta = new_contrib - old_contrib
        
        if delta > 0 or random.random() < (2.718281828 ** (delta / T)):
            # Accept
            family[i] = new_mask
            fam_set.remove(old_mask)
            fam_set.add(new_mask)
            fam_bits = (fam_bits & ~(1 << old_mask)) | (1 << new_mask)
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_family = family[:]
    
    return best_family, best_score

def construct():
    start_time = time.time()
    deadline = 60.0  # seconds
    
    # Precompute comparability bitsets
    _get_comp_bitsets()
    
    # Greedy initialization
    family, score = _greedy_init()
    
    # Local search
    family, score = _local_search(family, max_iters=3000, time_limit=15)
    
    # Simulated annealing if time permits
    if time.time() - start_time < deadline - 10:
        family, score = _simulated_annealing(family, time_limit=min(30, deadline - (time.time() - start_time) - 5))
    
    # Convert masks to lists of vertices
    result = []
    for m in family:
        verts = []
        for b in range(N):
            if (m >> b) & 1:
                verts.append(b)
        result.append(verts)
    
    return result

# EVOLVE-BLOCK-END
