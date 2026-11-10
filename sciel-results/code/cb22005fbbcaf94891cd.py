import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    N = 128
    TARGETS = list(range(128, 257))
    NUM_TARGETS = len(TARGETS)
    
    def compute_multiplicities(A_set):
        """
        Returns an array of multiplicities for targets 0..256.
        Uses convolution for speed.
        """
        A_bool = np.zeros(2 * N + 1, dtype=int)
        for a in A_set:
            A_bool[a] = 1
        conv = np.convolve(A_bool, A_bool)
        mults = np.zeros(2 * N + 1, dtype=int)
        for t in range(128, 257):
            delta = 0
            if t % 2 == 0:
                half = t // 2
                if half in A_set:
                    delta = 1
            mults[t] = (conv[t] + delta) // 2
        return mults

    def is_complete(mults):
        return np.all(mults[128:257] > 0)

    def get_max_mult(mults):
        return np.max(mults[128:257])

    def get_sum_of_squares(mults):
        return np.sum(mults[128:257] ** 2)

    # Efficient delta update: when adding/removing element x, update multiplicities
    # for all affected targets in O(|A|) time instead of O(N) convolution
    
    def make_delta_updater(A_set):
        """Return functions that update mults array in-place for add/remove."""
        # Precompute: for each element x in A, which targets t in [128,256] are affected
        # by adding/removing x? Target t is affected if t-x is in A (for add) 
        # or if removing x breaks pairs.
        
        def add_element(x, mults):
            """Add x to A_set, update mults in-place. Returns list of affected targets."""
            A_set.add(x)
            affected = []
            for t in range(128, 257):
                y = t - x
                if 0 <= y <= N and y in A_set:
                    # The pair (min(x,y), max(x,y)) now exists
                    # Check if it's a new pair
                    if x == y:
                        # (x,x) pair: only counts once, was not present before adding x
                        mults[t] += 1
                    else:
                        # (x,y) with x != y: this unordered pair was not counted before
                        # (since x wasn't in A before), so add 1
                        mults[t] += 1
                    affected.append(t)
            return affected

        def remove_element(x, mults):
            """Remove x from A_set, update mults in-place. Returns list of affected targets."""
            affected = []
            for t in range(128, 257):
                y = t - x
                if 0 <= y <= N and y in A_set:
                    # This pair will be lost
                    mults[t] -= 1
                    affected.append(t)
            A_set.discard(x)
            return affected

        def swap_element(out_x, in_x, mults):
            """Remove out_x, add in_x. Update mults in-place."""
            # First handle removal of out_x
            for t in range(128, 257):
                y = t - out_x
                if 0 <= y <= N and y in A_set:
                    mults[t] -= 1
            A_set.discard(out_x)
            # Then handle addition of in_x
            for t in range(128, 257):
                y = t - in_x
                if 0 <= y <= N and y in A_set:
                    mults[t] += 1
            A_set.add(in_x)

        return add_element, remove_element, swap_element

    best_A = None
    best_mult = float('inf')
    best_sos = float('inf')
    
    # Structured initialization strategies
    def structured_init():
        """Create a structured set with low initial multiplicity."""
        # Strategy: Use a sparse arithmetic progression
        # A = {0, d, 2d, ..., kd} where d is chosen to spread out sums
        # Or use two interleaved APs
        
        # Try: A = {0, 2, 4, ..., 2k} union {some offset elements}
        # Or: A = {i * 3 for i in range(k)} to get gaps
        
        # Best structured approach: start with a set that covers [128,256] with low mult
        # Use: A = {0, 4, 8, ..., 124} (32 elements, step 4)
        # This gives sums that are multiples of 4 in [0, 248]
        # Not enough coverage. Need denser.
        
        # Try: A = {0, 1, 2, ..., 63} union {64, 66, 68, ..., 128}
        # Or simply: random but with structure
        
        # Let's use: A = {i for i in range(0, 129, 2)} = {0,2,4,...,128} (65 elements)
        # Sums are even numbers in [0, 256]. Covers even targets in [128,256].
        # But odd targets not covered.
        
        # Better: A = {0, 1, 4, 5, 8, 9, ...} pattern
        # Or: A = {i for i in range(0, 129) if i % 3 != 2} (86 elements)
        
        # Let's try several structured inits and pick best
        candidates = []
        
        # Init 1: All even + some odds
        A1 = set(range(0, 129, 2))  # {0,2,4,...,128}
        candidates.append(A1)
        
        # Init 2: Step 3
        A2 = set(range(0, 129, 3))
        candidates.append(A2)
        
        # Init 3: Step 2 + offset
        A3 = set(range(1, 129, 2))  # {1,3,5,...,127}
        candidates.append(A3)
        
        # Init 4: Mixed - lower half dense, upper half sparse
        A4 = set(range(0, 65)) | set(range(64, 129, 2))
        candidates.append(A4)
        
        # Init 5: Random structured
        A5 = set(random.sample(range(0, 129), 40))
        candidates.append(A5)
        
        # Pick the one with best max multiplicity after ensuring coverage
        best_init = None
        best_init_mult = float('inf')
        
        for A_cand in candidates:
            # Ensure coverage by greedy adding
            A_work = set(A_cand)
            mults = compute_multiplicities(A_work)
            for _ in range(50):
                if is_complete(mults):
                    break
                uncovered = [t for t in TARGETS if mults[t] == 0]
                if not uncovered:
                    break
                best_gain = 0
                best_elem = None
                for x in range(N + 1):
                    if x in A_work:
                        continue
                    gain = 0
                    for t in uncovered:
                        y = t - x
                        if 0 <= y <= N and y in A_work:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_elem = x
                if best_gain > 0 and best_elem is not None:
                    A_work.add(best_elem)
                    mults = compute_multiplicities(A_work)
                else:
                    break
            
            if is_complete(mults):
                m = get_max_mult(mults)
                if m < best_init_mult:
                    best_init_mult = m
                    best_init = A_work
        
        if best_init is None:
            # Fallback: random
            best_init = set(random.sample(range(0, N + 1), 40))
        
        return best_init

    # Main search loop
    for attempt in range(30):
        if time.time() > deadline - 5:
            break
        
        # Structured initialization
        A_set = structured_init()
        mults = compute_multiplicities(A_set)
        
        if not is_complete(mults):
            # Force complete
            for t in TARGETS:
                if mults[t] == 0:
                    a = t // 2
                    b = t - a
                    A_set.add(a)
                    A_set.add(b)
            mults = compute_multiplicities(A_set)
        
        current_max = get_max_mult(mults)
        current_sos = get_sum_of_squares(mults)
        
        if current_max < best_mult or (current_max == best_mult and current_sos < best_sos):
            best_mult = current_max
            best_sos = current_sos
            best_A = set(A_set)
        
        # Local search with delta updates
        add_elem, remove_elem, swap_elem = make_delta_updater(A_set)
        
        # Targeted local search: focus on high-multiplicity targets
        temp = 5.0
        cooling = 0.97
        steps = 500
        
        for step in range(steps):
            if time.time() > deadline - 2:
                break
            
            # Find targets with max multiplicity
            target_slice = mults[128:257]
            current_max = np.max(target_slice)
            max_indices = np.where(target_slice == current_max)[0]
            
            if len(max_indices) == 0:
                break
            
            # Pick a random max target
            t_idx = max_indices[random.randint(0, len(max_indices) - 1)]
            t = 128 + t_idx
            
            # Find a pair (a, b) in A with a + b = t, a <= b
            pairs = []
            for a in A_set:
                b = t - a
                if a <= b and b in A_set:
                    pairs.append((a, b))
            
            if not pairs:
                continue
            
            # Pick a random pair
            a, b = random.choice(pairs)
            
            # Try swapping a with a random c not in A
            candidates_out = [x for x in range(N + 1) if x not in A_set]
            if not candidates_out:
                break
            
            c = random.choice(candidates_out)
            
            # Save state for rollback
            old_mults = mults.copy()
            old_A = set(A_set)
            
            # Perform swap
            swap_elem(a, c, mults)
            
            # Check coverage
            if not is_complete(mults):
                # Rollback
                mults[:] = old_mults
                A_set.clear()
                A_set.update(old_A)
                continue
            
            new_max = get_max_mult(mults)
            new_sos = get_sum_of_squares(mults)
            
            # Acceptance criterion
            accepted = False
            if new_max < current_max:
                accepted = True
            elif new_max == current_max:
                if new_sos < current_sos:
                    accepted = True
                elif random.random() < 0.3:
                    accepted = True
            else:
                delta = new_max - current_max
                prob = np.exp(-delta / temp)
                if random.random() < prob:
                    accepted = True
            
            if not accepted:
                # Rollback
                mults[:] = old_mults
                A_set.clear()
                A_set.update(old_A)
            else:
                current_max = new_max
                current_sos = new_sos
                if current_max < best_mult or (current_max == best_mult and current_sos < best_sos):
                    best_mult = current_max
                    best_sos = current_sos
                    best_A = set(A_set)
            
            temp *= cooling
            
            if current_max == 1:
                break
    
    if best_A is None:
        return list(range(64, 129))
        
    return list(best_A)
