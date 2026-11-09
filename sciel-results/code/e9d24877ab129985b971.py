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
    # This represents the input value on wire i for all possible input masks k (0 to 2^N - 1)
    for i in range(N):
        w = 0
        for k in range(TOTAL_MASKS):
            if (k >> i) & 1:
                w |= (1 << k)
        _INITIAL_WIRES[i] = w
        
    # Expected wires: bit k of wire i is 1 if popcount(k) >= N-i
    # A sorted network places the largest values (highest popcount) on the lowest indices?
    # Wait, the task says "min to wire i and max to j".
    # Standard sorting network convention: wire 0 is the smallest, wire N-1 is the largest.
    # So for input mask k, the value at wire i should be the i-th smallest element.
    # The elements are 0s and 1s. The sorted sequence is (N - popcount(k)) zeros followed by popcount(k) ones.
    # Wire 0 gets the smallest value.
    # If popcount(k) is P, there are P ones and N-P zeros.
    # Sorted: [0]*(N-P) + [1]*P
    # Wire i gets 1 if i >= N-P.
    # i >= N-P  <=>  P >= N-i.
    # So wire i has 1 if popcount(k) >= N - i.
    for i in range(N):
        w = 0
        threshold = N - i
        for k in range(TOTAL_MASKS):
            if k.bit_count() >= threshold:
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
    Focused local search: try removals, swaps, and insertions iteratively.
    Uses the fast big-int verifier.
    """
    comps = list(comps)
    best = list(comps)
    best_len = len(comps)
    
    # Cache the verification result for the current best to avoid re-verifying if no change
    # Actually, we verify candidates.
    
    while time.time() < deadline:
        moved = False
        n = len(comps)

        # Phase 1: Try removals
        for idx in range(n - 1, -1, -1):
            if time.time() > deadline:
                return best
            cand = comps[:idx] + comps[idx + 1:]
            if _verify_network(cand):
                comps = cand
                best = list(comps)
                best_len = len(comps)
                moved = True
                break

        if moved:
            continue

        # Phase 2: Try moving or modifying a comparator
        # We pick a random comparator and try to change its position or wires
        if n == 0:
            break
            
        # Try a few random moves per iteration to escape local minima
        found_improvement = False
        for _ in range(20): # Try 20 random perturbations
            if time.time() > deadline:
                return best
            
            cand = list(comps)
            idx = random.randint(0, len(cand) - 1)
            i, j = cand[idx]
            
            move_type = random.randint(0, 2)
            if move_type == 0:
                # Change wire i
                new_i = random.randint(0, j - 1)
                cand[idx] = (new_i, j)
            elif move_type == 1:
                # Change wire j
                new_j = random.randint(i + 1, N - 1)
                cand[idx] = (i, new_j)
            else:
                # Move comparator to a different position
                src = idx
                dst = random.randint(0, len(cand) - 1)
                comp = cand[src]
                del cand[src]
                if dst > src: dst -= 1
                cand.insert(dst, comp)
                
            # Quick check: is it valid?
            if not _verify_network(cand):
                continue
            
            # If valid, try to clean it up
            cleaned = _greedy_remove(cand, deadline)
            
            if len(cleaned) < best_len:
                comps = cleaned
                best = list(comps)
                best_len = len(comps)
                found_improvement = True
                break
        
        if not found_improvement:
            # If no improvement found in this batch, we might be stuck.
            # We could break, or continue trying. 
            # Given the deadline, breaking is safer to save time for potential future restarts if we had them,
            # but since we only have one run, we should keep trying if time remains?
            # Actually, if we are stuck, random restarts of the local search from the same point are unlikely to help 
            # without a bigger jump. However, the current loop structure allows continued attempts.
            # Let's add a small break condition to avoid infinite loops if n is small and no moves work.
            # But since we randomize, we might eventually find something.
            # To be safe and efficient, if we fail 100 times in a row, break.
            # For simplicity, we just break if no improvement in a batch, assuming local optimum.
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

    # Focused local search
    comps = _local_search(comps, deadline)

    return comps
