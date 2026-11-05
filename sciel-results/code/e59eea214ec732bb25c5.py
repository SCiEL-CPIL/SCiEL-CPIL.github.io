import time
import random
import numpy as np

def construct():
    n = 512
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Precompute all 4-APs as arrays of 4 indices (0-based internally)
    # A 4-AP: (a, a+d, a+2d, a+3d) with d>=1, all in [1,n]
    ap_list = []
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    ap_array = np.array(ap_list, dtype=np.int32)  # shape (M, 4)
    M = len(ap_array)
    
    # Precompute for each element x (0-based index), which APs it participates in
    # and at which position (0,1,2,3)
    # For speed, precompute: for each x, list of (ap_index, position)
    elem_aps = [[] for _ in range(n + 1)]
    for i, (a, b, c, d) in enumerate(ap_list):
        elem_aps[a].append((i, 0))
        elem_aps[b].append((i, 1))
        elem_aps[c].append((i, 2))
        elem_aps[d].append((i, 3))
    
    # Convert to numpy arrays for vectorized operations
    # For each element x, store the AP indices where x appears
    elem_ap_indices = [np.array([ap_idx for ap_idx, pos in elem_aps[x]], dtype=np.int32) if elem_aps[x] else np.array([], dtype=np.int32) for x in range(n + 1)]
    elem_ap_positions = [np.array([pos for ap_idx, pos in elem_aps[x]], dtype=np.int32) if elem_aps[x] else np.array([], dtype=np.int32) for x in range(n + 1)]
    
    # Core set: numbers with base-3 digits 0 or 1 (Cantor-like set)
    core = set()
    for mask in range(1, 1 << 9):  # 9 digits: 3^0 to 3^8 = 19683, but we cap at n
        val = 0
        for i in range(9):
            if (mask >> i) & 1:
                val += (3 ** i)
        if 1 <= val <= n:
            core.add(val)
    
    # Also add more base-3 digits (0,1) up to 512
    for mask in range(1, 1 << 6):
        val = 0
        for i in range(6):
            if (mask >> i) & 1:
                val += (3 ** i)
        if 1 <= val <= n:
            core.add(val)
    
    best_set = set(core)
    best_size = len(best_set)
    
    # Vectorized check: given a boolean mask of size n+1 (index 0 unused),
    # check if any AP has all 4 elements present
    def has_4ap(mask):
        """mask is a boolean numpy array of size n+1. Returns True if any 4-AP is fully contained."""
        if M == 0:
            return False
        # For each AP, check if all 4 positions are True in mask
        # ap_array is (M, 4), mask is (n+1,)
        present = mask[ap_array]  # shape (M, 4)
        return np.all(present, axis=1).any()
    
    # Check if adding element x (1-based) to set S (as boolean mask) creates a 4-AP
    def can_add_vec(mask, x):
        """mask is boolean numpy array size n+1. x is 1-based. Returns True if x can be added."""
        if mask[x]:
            return False
        # Get all APs containing x
        aps = elem_ap_indices[x]
        if len(aps) == 0:
            return True
        positions = elem_ap_positions[x]
        # For each such AP, check if the other 3 elements are in mask
        # Get the other 3 positions in each AP
        other_positions = np.array([p for p in range(4) if p not in positions], dtype=np.int32)
        # For each AP index, check the other 3 elements
        # ap_array[aps] gives shape (len(aps), 4)
        # We need to check that for each AP, the elements at positions other than x's position are all in mask
        ap_vals = ap_array[aps]  # shape (K, 4)
        # Create a mask of which positions to check (all except x's position)
        # positions has shape (K,), each value in {0,1,2,3}
        # For each row, we need to check the 3 columns not equal to the position
        # Simple approach: for each AP, check all 4 elements, but x is not yet in mask
        # So we need the other 3 to be in mask
        # Create a boolean matrix: for each AP row, which columns are NOT x's position
        # positions has shape (K,), ap_vals has shape (K,4)
        # Create column indices 0,1,2,3
        col_idx = np.arange(4)
        # For each row i, the columns to check are those where col_idx != positions[i]
        # Build a (K, 4) boolean matrix
        check_mask = col_idx[None, :] != positions[:, None]  # shape (K, 4)
        # For each row, check if all elements at check_mask positions are in mask
        # Get the values at those positions
        vals = ap_vals  # (K, 4)
        # For each row, mask out the x position and check the rest
        # Set x position to True (since we're checking if others are present)
        vals_mod = vals.copy()
        # Set the x position to True
        rows = np.arange(len(aps))
        vals_mod[rows, positions] = True
        # Now check if all positions are True in mask
        present = mask[vals_mod]  # shape (K, 4)
        return not np.all(present, axis=1).any()
    
    # Faster can_add using precomputed structure
    def can_add_fast(mask, x):
        """Faster version using precomputed AP lists."""
        if mask[x]:
            return False
        aps = elem_ap_indices[x]
        if len(aps) == 0:
            return True
        positions = elem_ap_positions[x]
        # For each AP containing x, check if the other 3 elements are in mask
        for ap_idx, pos in zip(aps, positions):
            a, b, c, d = ap_array[ap_idx]
            # Check the 3 elements that are not x
            if pos == 0:
                if mask[b] and mask[c] and mask[d]:
                    return False
            elif pos == 1:
                if mask[a] and mask[c] and mask[d]:
                    return False
            elif pos == 2:
                if mask[a] and mask[b] and mask[d]:
                    return False
            else:  # pos == 3
                if mask[a] and mask[b] and mask[c]:
                    return False
        return True
    
    # Even faster: precompute for each x, the list of (other1, other2, other3) triples
    # where x completes a 4-AP
    x_completes = [[] for _ in range(n + 1)]
    for i, (a, b, c, d) in enumerate(ap_list):
        x_completes[a].append((b, c, d))
        x_completes[b].append((a, c, d))
        x_completes[c].append((a, b, d))
        x_completes[d].append((a, b, c))
    
    def can_add_v2(mask, x):
        """Fastest version: check precomputed triples."""
        if mask[x]:
            return False
        for (o1, o2, o3) in x_completes[x]:
            if mask[o1] and mask[o2] and mask[o3]:
                return False
        return True
    
    # Convert set to mask
    def set_to_mask(s):
        mask = np.zeros(n + 1, dtype=bool)
        for x in s:
            mask[x] = True
        return mask
    
    def mask_to_set(mask):
        return set(np.where(mask)[0].tolist())
    
    # Systematic fill pass: repeatedly try to add any element that can be added
    def fill_pass(mask):
        """Repeatedly add any element that can be added until no more can be added."""
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if not mask[x] and can_add_v2(mask, x):
                    mask[x] = True
                    changed = True
        return mask
    
    # 2-swap move: remove one element, add another
    def try_swaps(mask, iterations=10000):
        """Try swap moves: remove element r, add element a. Accept if it improves or maintains size."""
        current_size = mask.sum()
        for _ in range(iterations):
            if time.time() > deadline:
                break
            # Pick a random element to remove
            r = random.randint(1, n)
            if not mask[r]:
                continue
            # Pick a random element to add
            a = random.randint(1, n)
            if mask[a]:
                continue
            # Try the swap
            mask[r] = False
            if can_add_v2(mask, a):
                mask[a] = True
                new_size = mask.sum()
                if new_size >= current_size:
                    current_size = new_size
                else:
                    # Revert
                    mask[a] = False
                    mask[r] = True
            else:
                mask[r] = True
        return mask
    
    # Main search loop
    random.seed(42)
    
    # Start with core set
    current_mask = set_to_mask(core)
    current_mask = fill_pass(current_mask)
    current_size = current_mask.sum()
    if current_size > best_size:
        best_size = current_size
        best_set = mask_to_set(current_mask)
    
    # Multiple restarts with different initial sets
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Create a perturbed initial set
        if restart == 0:
            current_mask = set_to_mask(core)
        else:
            # Start from best and perturb
            current_mask = set_to_mask(best_set)
            # Remove some random elements
            s = mask_to_set(current_mask)
            for _ in range(random.randint(5, 20)):
                if s:
                    x = random.choice(list(s))
                    s.remove(x)
                    current_mask[x] = False
        
        # Fill pass
        current_mask = fill_pass(current_mask)
        current_size = current_mask.sum()
        
        # Swap-based local search
        for swap_iter in range(5):
            if time.time() > deadline - 3:
                break
            current_mask = try_swaps(current_mask, iterations=5000)
            current_size = current_mask.sum()
            
            # Another fill pass after swaps
            current_mask = fill_pass(current_mask)
            current_size = current_mask.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_set = mask_to_set(current_mask)
    
    # Final fill pass on best
    best_mask = set_to_mask(best_set)
    best_mask = fill_pass(best_mask)
    best_size = best_mask.sum()
    best_set = mask_to_set(best_mask)
    
    return sorted(best_set)
