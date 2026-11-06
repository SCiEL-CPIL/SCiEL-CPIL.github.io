"""Rewritten program: bit-reversal / shuffle-exchange based sorting network search."""
import time
import random
import itertools

N = 17
SEED_COMPARATORS = 136  # odd-even transposition comparator count
DEADLINE_SECONDS = 85.0  # safety margin under 90s kill

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_compare(values, i, j):
    """Apply a compare-exchange: min to i, max to j."""
    if values[i] > values[j]:
        values[i], values[j] = values[j], values[i]


def _simulate_network(comparators, input_vec):
    """Simulate the network on a single input vector (list of 0/1)."""
    v = list(input_vec)
    for i, j in comparators:
        if v[i] > v[j]:
            v[i], v[j] = v[j], v[i]
    return v


def _is_sorted_output(output):
    """Check if the output is sorted (non-decreasing)."""
    for k in range(len(output) - 1):
        if output[k] > output[k + 1]:
            return False
    return True


def _validate_full(comparators):
    """Validate over all 2^N boolean inputs (zero-one principle)."""
    n = len(comparators)
    for mask in range(1 << N):
        inp = [(mask >> k) & 1 for k in range(N)]
        out = _simulate_network(comparators, inp)
        if not _is_sorted_output(out):
            return False
    return True


def _validate_sample(comparators, num_samples=4096, rng=None):
    """Validate on a random sample of boolean inputs."""
    if rng is None:
        rng = random.Random(42)
    for _ in range(num_samples):
        inp = [rng.randint(0, 1) for _ in range(N)]
        out = _simulate_network(comparators, inp)
        if not _is_sorted_output(out):
            return False
    return True


def _compute_depth(comparators):
    """Compute the parallel depth of the network."""
    depth = [0] * N
    for i, j in comparators:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth) if depth else 0


# ---------------------------------------------------------------------------
# Known good starting points
# ---------------------------------------------------------------------------

def _odd_even_transposition():
    """Classic odd-even transposition sorting network."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _bit_reverse_perm(n, k):
    """Return the bit-reversal permutation index for n wires, position k."""
    bits = bin(k)[2:].zfill(n)
    rev = bits[::-1]
    return int(rev, 2)


def _shuffle_network_base():
    """
    Construct a baseline network using a sequence of perfect-shuffle-like
    permutations interleaved with adjacent comparators.
    
    For N wires, we use a series of 'rounds' where in each round we apply
    a fixed permutation (derived from bit-reversal / shuffle structure)
    followed by a layer of comparators on adjacent wires.
    
    This gives a structured starting point that is different from plain
    odd-even transposition and may have different redundancy patterns.
    """
    comps = []
    # Number of shuffle rounds: use N rounds (same as OE transposition)
    for round_idx in range(N):
        # Generate a permutation based on bit-reversal with a rotation
        perm = []
        for k in range(N):
            # Rotate the bit-reversal by round_idx positions
            shifted = (k + round_idx) % N
            perm.append(_bit_reverse_perm(5, shifted) % N)  # 5 bits for up to 32, mod N
        # Actually, for non-power-of-2, let's use a simpler shuffle:
        # Perfect shuffle: split into two halves, interleave
        # For N=17: half1 = 8, half2 = 9
        half = N // 2  # 8
        perm = []
        for i in range(half):
            perm.append(i)
            perm.append(i + half)
        if N % 2 == 1:
            perm.append(N - 1)
        # Rotate perm by round_idx to get variety
        perm = perm[round_idx % N:] + perm[:round_idx % N]
        
        # Apply comparators on adjacent wires according to the shuffled order
        for idx in range(N - 1):
            a = perm[idx]
            b = perm[idx + 1]
            if a < b:
                comps.append((a, b))
            else:
                comps.append((b, a))
    return comps


def _akl_network():
    """
    Attempt to build a network inspired by AKL (Akl, Krishnamurthy, Laganière)
    or similar structured approaches. For simplicity, we combine:
    1. A few rounds of bit-reversal shuffle + adjacent compares
    2. Then odd-even transposition to finish
    
    This hybrid may have fewer redundant comparators.
    """
    comps = []
    # Phase 1: 4 rounds of shuffle + adjacent compares
    for round_idx in range(4):
        half = N // 2
        perm = []
        for i in range(half):
            perm.append(i)
            perm.append(i + half)
        if N % 2 == 1:
            perm.append(N - 1)
        perm = perm[round_idx % N:] + perm[:round_idx % N]
        for idx in range(N - 1):
            a = perm[idx]
            b = perm[idx + 1]
            if a < b:
                comps.append((a, b))
            else:
                comps.append((b, a))
    # Phase 2: odd-even transposition for remaining rounds
    for phase in range(4, N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


# ---------------------------------------------------------------------------
# Local search: comparator removal
# ---------------------------------------------------------------------------

def _try_remove_comparators(comparators, time_limit, rng):
    """
    Greedy local search: try removing each comparator one at a time.
    Use sample validation for speed, then full validation at the end.
    """
    start_time = time.time()
    current = list(comparators)
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Shuffle the order of removal attempts for diversity
        indices = list(range(len(current)))
        rng.shuffle(indices)
        
        for idx in indices:
            if (time.time() - start_time) >= time_limit:
                break
            # Try removing comparator at idx
            candidate = current[:idx] + current[idx+1:]
            # Quick sample validation
            if _validate_sample(candidate, num_samples=2048, rng=rng):
                # Promising! Do full validation
                if _validate_full(candidate):
                    current = candidate
                    improved = True
                    break  # Restart scanning from beginning
    
    return current


# ---------------------------------------------------------------------------
# Local search: comparator replacement (swap one comparator for another)
# ---------------------------------------------------------------------------

def _try_replace_comparators(comparators, time_limit, rng):
    """
    Try replacing individual comparators with different ones.
    This explores the neighborhood more broadly than pure removal.
    """
    start_time = time.time()
    current = list(comparators)
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        indices = list(range(len(current)))
        rng.shuffle(indices)
        
        for idx in indices:
            if (time.time() - start_time) >= time_limit:
                break
            # Try replacing with a random different comparator
            for _ in range(20):  # Try a few replacements per position
                new_comp = rng.choice(all_pairs)
                if new_comp == current[idx]:
                    continue
                candidate = current[:idx] + [new_comp] + current[idx+1:]
                if _validate_sample(candidate, num_samples=1024, rng=rng):
                    if _validate_full(candidate):
                        current = candidate
                        improved = True
                        break
            if improved:
                break
    
    return current


# ---------------------------------------------------------------------------
# Hill-climbing with multiple restarts
# ---------------------------------------------------------------------------

def _local_search(comparators, time_budget, rng):
    """Run local search with a given time budget."""
    start_time = time.time()
    current = list(comparators)
    
    # Alternate between removal and replacement phases
    phase = 0
    while (time.time() - start_time) < time_budget:
        remaining = time_budget - (time.time() - start_time)
        if remaining < 2:
            break
        
        if phase % 2 == 0:
            new_current = _try_remove_comparators(current, remaining * 0.7, rng)
        else:
            new_current = _try_replace_comparators(current, remaining * 0.7, rng)
        
        if len(new_current) < len(current):
            current = new_current
        elif len(new_current) == len(current) and _compute_depth(new_current) < _compute_depth(current):
            current = new_current
        
        phase += 1
    
    return current


# ---------------------------------------------------------------------------
# Main construct function
# ---------------------------------------------------------------------------

def construct():
    start_time = time.time()
    rng = random.Random(12345)
    
    # Generate several candidate starting networks
    candidates = []
    
    # 1. Odd-even transposition (baseline)
    candidates.append(_odd_even_transposition())
    
    # 2. Bit-reversal shuffle network
    candidates.append(_shuffle_network_base())
    
    # 3. Hybrid AKL-inspired
    candidates.append(_akl_network())
    
    # 4. Reverse bit-reversal shuffle
    comps4 = []
    for round_idx in range(N):
        half = N // 2
        perm = []
        for i in range(half):
            perm.append(i)
            perm.append(i + half)
        if N % 2 == 1:
            perm.append(N - 1)
        # Reverse rotation
        perm = perm[-(round_idx % N):] + perm[:- (round_idx % N)] if round_idx % N else perm
        for idx in range(N - 1):
            a = perm[idx]
            b = perm[idx + 1]
            if a < b:
                comps4.append((a, b))
            else:
                comps4.append((b, a))
    candidates.append(comps4)
    
    # 5. Interleaved odd-even with shuffle
    comps5 = []
    for round_idx in range(N):
        if round_idx % 3 == 0:
            # Shuffle round
            half = N // 2
            perm = []
            for i in range(half):
                perm.append(i)
                perm.append(i + half)
            if N % 2 == 1:
                perm.append(N - 1)
            perm = perm[round_idx % N:] + perm[:round_idx % N]
            for idx in range(N - 1):
                a = perm[idx]
                b = perm[idx + 1]
                if a < b:
                    comps5.append((a, b))
                else:
                    comps5.append((b, a))
        else:
            # Odd-even round
            for i in range(round_idx % 2, N - 1, 2):
                comps5.append((i, i + 1))
    candidates.append(comps5)
    
    # Validate all candidates and pick the best (fewest comparators that sort)
    valid_candidates = []
    for c in candidates:
        if _validate_full(c):
            valid_candidates.append(c)
    
    if not valid_candidates:
        # Fallback: return the odd-even transposition (known correct)
        return _odd_even_transposition()
    
    # Sort by length (fewest comparators first)
    valid_candidates.sort(key=len)
    
    # Allocate time for local search on the top candidates
    best_result = valid_candidates[0]
    best_len = len(best_result)
    
    remaining_time = DEADLINE_SECONDS - (time.time() - start_time)
    
    # Distribute time among top candidates
    num_to_search = min(len(valid_candidates), 4)
    time_per_candidate = max(5.0, remaining_time / num_to_search - 5)  # Leave 5s for final validation
    
    for i in range(num_to_search):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
        
        candidate = valid_candidates[i]
        search_budget = min(time_per_candidate, DEADLINE_SECONDS - 5 - (time.time() - start_time))
        if search_budget < 3:
            break
        
        # Use a different seed for each candidate
        local_rng = random.Random(1000 + i * 7)
        improved = _local_search(candidate, search_budget, local_rng)
        
        if len(improved) < best_len:
            best_result = improved
            best_len = len(best_result)
    
    # Final full validation
    if not _validate_full(best_result):
        # Fallback to first valid candidate
        best_result = valid_candidates[0]
    
    return best_result
