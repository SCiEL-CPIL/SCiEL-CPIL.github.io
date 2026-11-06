"""Structural redesign: explicit deadline-guarded heuristic search for size-efficient sorting networks."""
import time
import random
import itertools
import numpy as np

N = 17
ALL_INPUTS = 1 << N
MAX_COMPARATORS = 4096
SEED_COUNT = N * N  # 289 for N=17, matches odd-even transposition

# Precompute bit masks for each wire
WIRE_MASKS = [1 << i for i in range(N)]

# Precompute all possible comparator pairs
ALL_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
NUM_PAIRS = len(ALL_PAIRS)
PAIR_INDEX = {}
for idx, (i, j) in enumerate(ALL_PAIRS):
    PAIR_INDEX[(i, j)] = idx


def _apply_comparator_to_bitmasks(bitmasks, i, j):
    """Apply comparator (i,j) to all bitmasks. Returns new list."""
    a = bitmasks[i]
    b = bitmasks[j]
    # min goes to i, max goes to j
    # For bitmask: bit k is 1 if wire has value 1 in input k
    # After compare-exchange: wire i gets min, wire j gets max
    # min(a,b) per input: bit is 1 only if both a and b have 1 -> a & b
    # max(a,b) per input: bit is 1 if either a or b has 1 -> a | b
    new_a = a & b
    new_b = a | b
    if new_a == a and new_b == b:
        return bitmasks
    new_bitmasks = list(bitmasks)
    new_bitmasks[i] = new_a
    new_bitmasks[j] = new_b
    return new_bitmasks


def _is_sorted(bitmasks):
    """Check if all bitmasks represent sorted state (0s then 1s)."""
    for bm in bitmasks:
        if bm == 0:
            continue
        # A sorted bitmask for 17 wires should be of form 0b00...011...1
        # i.e., (1 << k) - 1 for some k
        k = bm.bit_length()
        if bm != (1 << k) - 1:
            return False
    return True


def _evaluate_network(comparators):
    """Evaluate a network over all 2^17 inputs. Returns (is_valid, depth)."""
    bitmasks = list(range(ALL_INPUTS))  # bitmask for each wire
    # Actually, we need to track the state of each wire across all inputs.
    # wire_state[w] = bitmask where bit k is 1 if wire w has value 1 in input k
    wire_state = [0] * N
    for k in range(ALL_INPUTS):
        for w in range(N):
            if (k >> w) & 1:
                wire_state[w] |= (1 << k)
    
    for (i, j) in comparators:
        a = wire_state[i]
        b = wire_state[j]
        new_a = a & b
        new_b = a | b
        if new_a != a or new_b != b:
            wire_state[i] = new_a
            wire_state[j] = new_b
    
    return _is_sorted(wire_state)


def _evaluate_network_fast(comparators):
    """Faster evaluation using numpy."""
    # Create input matrix: 131072 x 17
    # This might be too large, let's use a different approach
    # Use the bitmask approach but with numpy for speed
    n_inputs = 1 << N
    # wire_state[w] is an array of n_inputs booleans
    # We'll use numpy uint64 arrays for bitmasks (131072 bits = 16384 uint64s)
    
    n_uint64 = (n_inputs + 63) // 64
    
    # Initialize wire states
    # wire w has 1 in input k if bit w of k is set
    wire_states = np.zeros((N, n_uint64), dtype=np.uint64)
    
    for w in range(N):
        # For each uint64 block, determine which inputs have bit w set
        for block in range(n_uint64):
            start = block * 64
            end = min(start + 64, n_inputs)
            for k in range(start, end):
                if (k >> w) & 1:
                    wire_states[w, block] |= (1 << (k - start))
    
    for (i, j) in comparators:
        a = wire_states[i]
        b = wire_states[j]
        new_a = np.bitwise_and(a, b)
        new_b = np.bitwise_or(a, b)
        if not np.array_equal(new_a, a) or not np.array_equal(new_b, b):
            wire_states[i] = new_a
            wire_states[j] = new_b
    
    # Check if sorted
    for w in range(N):
        ws = wire_states[w]
        # Check if ws is a prefix of 1s (in terms of the full 131072-bit number)
        # A sorted state means: for each input k, the output is 0 or 1,
        # and the set of inputs where output is 1 is {k : popcount(k) >= threshold}
        # Actually, simpler: the bitmask should be of form (1<<m)-1
        # But we have it as uint64 array. Let's convert to check.
        
        # Build the full integer from the uint64 array
        # This is expensive. Alternative: check structure directly.
        # In a sorted network, wire w outputs 1 for exactly the inputs with
        # the (n_w) largest values, where n_w is determined by the permutation.
        # Actually, the zero-one principle says we just need to check all 2^17 inputs.
        # The sorted condition: the output sequence is non-decreasing.
        # For wire w, the output is 1 iff the input's rank puts it in the top group.
        
        # Simpler check: verify that the bitmask is "monotone" - if input k1 < k2 
        # in terms of the sorted order... this is complex.
        
        # Let's just verify: for the bitmask to represent a valid sorted output,
        # it must be that there exists some threshold such that exactly the inputs
        # with value >= threshold (in the appropriate ordering) have bit set.
        
        # Actually, the simplest correct check: reconstruct the integer and check.
        val = 0
        for block in range(n_uint64):
            val |= (int(ws[block]) << (block * 64))
        
        if val == 0:
            continue
        k = val.bit_length()
        if val != (1 << k) - 1:
            return False, 0
    
    # Compute depth (longest path)
    depth = 0
    wire_depth = [0] * N
    for (i, j) in comparators:
        d = max(wire_depth[i], wire_depth[j]) + 1
        wire_depth[i] = d
        wire_depth[j] = d
        depth = max(depth, d)
    
    return True, depth


def _depth_of_network(comparators):
    """Compute parallel depth of a network."""
    if not comparators:
        return 0
    wire_depth = [0] * N
    max_depth = 0
    for (i, j) in comparators:
        d = max(wire_depth[i], wire_depth[j]) + 1
        wire_depth[i] = d
        wire_depth[j] = d
        max_depth = max(max_depth, d)
    return max_depth


def _seed_network():
    """Odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _random_valid_network(n_comps, rng):
    """Generate a random network with n_comps comparators."""
    comps = []
    for _ in range(n_comps):
        i = rng.randint(0, N - 2)
        j = rng.randint(i + 1, N - 1)
        comps.append((i, j))
    return comps


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    best_network = _seed_network()
    best_count = len(best_network)
    
    # Quick validation of seed
    if not _evaluate_network_fast(best_network)[0]:
        # Fallback: return seed anyway
        return best_network
    
    # Try to improve via local search / hill climbing
    rng = random.Random(42)
    
    # Strategy: start from seed, try to remove/replace comparators
    # First, try removing comparators one at a time
    
    current = list(best_network)
    current_count = len(current)
    
    # Phase 1: Greedy removal
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Try removing comparator at idx
            candidate = current[:idx] + current[idx+1:]
            if len(candidate) < best_count:
                valid, _ = _evaluate_network_fast(candidate)
                if valid:
                    current = candidate
                    current_count = len(current)
                    best_count = current_count
                    best_network = list(current)
                    improved = True
                    break
    
    # Phase 2: Local search - swap/replace comparators
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Randomly pick a comparator to replace or add
        if len(current) > 60 and rng.random() < 0.7:
            # Replace a random comparator
            idx = rng.randint(0, len(current) - 1)
            new_i = rng.randint(0, N - 2)
            new_j = rng.randint(new_i + 1, N - 1)
            candidate = list(current)
            candidate[idx] = (new_i, new_j)
        else:
            # Add a comparator at random position
            new_i = rng.randint(0, N - 2)
            new_j = rng.randint(new_i + 1, N - 1)
            pos = rng.randint(0, len(current))
            candidate = list(current)
            candidate.insert(pos, (new_i, new_j))
        
        valid, _ = _evaluate_network_fast(candidate)
        if valid:
            if len(candidate) < current_count:
                current = candidate
                current_count = len(current)
                best_count = current_count
                best_network = list(current)
            elif len(candidate) == current_count and rng.random() < 0.3:
                # Accept equal-size moves occasionally to escape local minima
                current = candidate
    
    # Phase 3: Try some structured improvements
    # Try to build a smaller network from scratch using heuristic construction
    
    # Known good patterns: AKS-style, or specific constructions
    # For now, try a few hand-crafted small networks
    
    # Try: merge-sort based network
    def merge_sort_network(lo, hi):
        """Generate merge-sort network for wires lo..hi (exclusive)."""
        if hi - lo <= 1:
            return []
        mid = (lo + hi) // 2
        left = merge_sort_network(lo, mid)
        right = merge_sort_network(mid, hi)
        # Merge step: compare-exchange to merge two sorted halves
        # For merging two sorted sequences of lengths a and b,
        # we need a specific pattern
        merge_comps = []
        a_len = mid - lo
        b_len = hi - mid
        # Simple merge: for each position, compare appropriate wires
        # This is a simplified merge; a proper merge network is more complex
        for k in range(a_len + b_len - 1):
            for pos in range(max(0, k - b_len + 1), min(k + 1, a_len)):
                i = lo + pos
                j = mid + (k - pos)
                if 0 <= i < N and 0 <= j < N and i != j:
                    if i > j:
                        i, j = j, i
                    merge_comps.append((i, j))
        return left + right + merge_comps
    
    ms_network = merge_sort_network(0, N)
    if len(ms_network) < best_count:
        valid, _ = _evaluate_network_fast(ms_network)
        if valid:
            best_network = ms_network
            best_count = len(ms_network)
    
    return best_network
