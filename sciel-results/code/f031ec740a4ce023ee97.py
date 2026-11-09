import time
import random

N = 17
TOTAL_MASKS = 1 << N
DEADLINE_SECONDS = 50

# Pre-compute initial and expected wires for verification using big integers.
# This allows O(1) updates for compare-exchange operations across all 2^17 inputs.
_INITIAL_WIRES = None
_EXPECTED_WIRES = None

def _init_wires():
    global _INITIAL_WIRES, _EXPECTED_WIRES
    if _INITIAL_WIRES is not None:
        return
    
    _INITIAL_WIRES = [0] * N
    _EXPECTED_WIRES = [0] * N
    
    # Initial wires: bit k of wire i is 1 if (k >> i) & 1
    # Build efficiently using bit patterns
    for i in range(N):
        # Pattern: blocks of 2^i zeros followed by 2^i ones, repeating
        # Build by setting bits in ranges
        w = 0
        block = 1 << i
        period = 2 * block
        for start in range(block, TOTAL_MASKS, period):
            end = min(start + block, TOTAL_MASKS)
            # Set bits from start to end-1
            mask = (1 << (end - start)) - 1
            w |= mask << start
        _INITIAL_WIRES[i] = w
        
    # Expected wires: bit k of wire i is 1 if popcount(k) >= N-i
    # Group masks by popcount, then set bits for all masks with popcount >= threshold
    # First, compute popcount for all masks
    popcounts = [0] * TOTAL_MASKS
    for k in range(1, TOTAL_MASKS):
        popcounts[k] = popcounts[k >> 1] + (k & 1)
    
    # For each wire i, threshold = N - i
    # Wire i has 1 for all k where popcount(k) >= N - i
    for i in range(N):
        threshold = N - i
        w = 0
        for k in range(TOTAL_MASKS):
            if popcounts[k] >= threshold:
                w |= (1 << k)
        _EXPECTED_WIRES[i] = w

_init_wires()

def _verify_network(net):
    """
    Fast verification using pre-computed big integers.
    """
    wires = _INITIAL_WIRES[:]
    
    for i, j in net:
        wi = wires[i]
        wj = wires[j]
        # min to i, max to j
        # Swap bits where wi=1 and wj=0 (i has 1, j has 0 -> swap to 0, 1)
        # S is the mask of inputs where swap is needed
        S = wi & ~wj
        wires[i] = wi ^ S
        wires[j] = wj ^ S
        
    # Check against expected
    for i in range(N):
        if wires[i] != _EXPECTED_WIRES[i]:
            return False
    return True

def _build_seed_network():
    """
    Constructs Batcher's Odd-Merge sort network for 17 inputs.
    This provides a strong starting point (~105 comparators).
    """
    comps = []

    def _sort_wires(wires, length):
        """Sort elements at given wire positions using odd-merge."""
        if length <= 1:
            return
        if length == 2:
            a, b = wires[0], wires[1]
            if a < b:
                comps.append((a, b))
            else:
                comps.append((b, a))
            return

        half_odd = (length + 1) // 2
        half_even = length // 2

        # Split wires into odd and even positions within this sub-problem
        sub_odd = [wires[2 * i] for i in range(half_odd)]
        sub_even = [wires[2 * i + 1] for i in range(half_even)]

        _sort_wires(sub_odd, half_odd)
        _sort_wires(sub_even, half_even)

        # Merge step
        m = min(half_odd, half_even)
        for i in range(m):
            a = sub_odd[i]
            b = sub_even[i]
            if a < b:
                comps.append((a, b))
            else:
                comps.append((b, a))
        for i in range(m - 1):
            a = sub_odd[i + 1]
            b = sub_even[i]
            if a < b:
                comps.append((a, b))
            else:
                comps.append((b, a))

    _sort_wires(list(range(N)), N)
    return comps

def _greedy_remove(comps, deadline):
    """Try to remove each comparator if the network still sorts."""
    comps = list(comps)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(comps)
        # Iterate backwards to handle index shifts safely if we were modifying in place, 
        # but we are creating new lists. Backwards is often slightly faster for finding removals 
        # if the network was built sequentially, but random order might be better. 
        # Let's stick to a simple scan.
        for idx in range(n - 1, -1, -1):
            if time.time() > deadline:
                return comps
            new_comps = comps[:idx] + comps[idx + 1:]
            if _verify_network(new_comps):
                comps = new_comps
                improved = True
                break
    return comps

def _local_search(comps, deadline):
    """
    Focused local search with perturb-verify-cleanup strategy.
    Uses the fast big-int verifier.
    """
    comps = list(comps)
    best = list(comps)
    best_len = len(comps)
    stall_count = 0
    
    while time.time() < deadline:
        n = len(comps)
        if n == 0:
            break

        # Phase 1: Aggressive removal pass
        moved = False
        for idx in range(n - 1, -1, -1):
            if time.time() > deadline:
                return best
            cand = comps[:idx] + comps[idx + 1:]
            if _verify_network(cand):
                comps = cand
                best = list(comps)
                best_len = len(comps)
                moved = True
                stall_count = 0
                break

        if moved:
            continue

        # Phase 2: Perturb-Verify-Cleanup
        # Try multiple random perturbations per round
        found_improvement = False
        for _ in range(50):
            if time.time() > deadline:
                return best
            
            cand = list(comps)
            idx = random.randint(0, len(cand) - 1)
            i, j = cand[idx]
            
            move_type = random.randint(0, 3)
            if move_type == 0:
                # Change wire i to a nearby wire
                new_i = random.randint(0, j - 1)
                cand[idx] = (new_i, j)
            elif move_type == 1:
                # Change wire j to a nearby wire
                new_j = random.randint(i + 1, N - 1)
                cand[idx] = (i, new_j)
            elif move_type == 2:
                # Move comparator to a different position
                src = idx
                dst = random.randint(0, len(cand) - 1)
                comp = cand[src]
                del cand[src]
                if dst > src:
                    dst -= 1
                cand.insert(dst, comp)
            else:
                # Swap two comparators' positions
                idx2 = random.randint(0, len(cand) - 1)
                if idx2 != idx:
                    cand[idx], cand[idx2] = cand[idx2], cand[idx]
                
            # Quick check: is it valid?
            if not _verify_network(cand):
                continue
            
            # If valid, aggressively clean it up
            cleaned = _greedy_remove(cand, deadline)
            
            if len(cleaned) < best_len:
                comps = cleaned
                best = list(comps)
                best_len = len(comps)
                found_improvement = True
                stall_count = 0
                break
        
        if not found_improvement:
            stall_count += 1
            # If stuck for too long, try a bigger perturbation
            if stall_count > 5 and n > 1:
                # Remove a random comparator and try to re-add useful ones
                cand = list(comps)
                idx = random.randint(0, len(cand) - 1)
                del cand[idx]
                if _verify_network(cand):
                    comps = cand
                    best = list(comps)
                    best_len = len(comps)
                    stall_count = 0
                    continue
                # If that didn't work, try inserting a new comparator
                if n > 0:
                    cand = list(comps)
                    new_i = random.randint(0, N - 2)
                    new_j = random.randint(new_i + 1, N - 1)
                    pos = random.randint(0, len(cand))
                    cand.insert(pos, (new_i, new_j))
                    if _verify_network(cand):
                        cleaned = _greedy_remove(cand, deadline)
                        if len(cleaned) < best_len:
                            comps = cleaned
                            best = list(comps)
                            best_len = len(comps)
                            stall_count = 0
                        else:
                            stall_count += 1
                    else:
                        stall_count += 1
            else:
                stall_count += 1
            
            # If completely stuck, break
            if stall_count > 20:
                break

    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    # Build seed network (Odd-Merge, ~105 comparators)
    comps = _build_seed_network()

    # Verify seed
    if not _verify_network(comps):
        # Fallback to odd-even transposition if seed is broken
        comps = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

    # Initial greedy removal
    comps = _greedy_remove(comps, deadline)
    
    # Run local search
    comps = _local_search(comps, deadline)
    
    # If time remains, do another round of greedy removal + local search
    if time.time() < deadline - 5:
        comps = _greedy_remove(comps, deadline)
        comps = _local_search(comps, deadline)

    return comps
