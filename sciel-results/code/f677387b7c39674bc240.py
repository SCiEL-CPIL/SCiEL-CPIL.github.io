import time
import random

N = 256
DEADLINE = 55.0  # Safety margin before 60s limit

def _precompute():
    """Precompute 3-AP and 4-AP information using bitmasks."""
    # 3-APs: (a, a+d, a+2d)
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            three_aps.append(mask)
    
    # 4-APs: (a, a+d, a+2d, a+3d)
    four_aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
            four_aps.append(mask)
    
    # Map each element to the list of 4-AP indices it belongs to
    four_aps_by_v = [[] for _ in range(N + 1)]
    for idx, mask in enumerate(four_aps):
        for v in range(1, N + 1):
            if mask & (1 << (v-1)):
                four_aps_by_v[v].append(idx)
    
    # Map each element to the list of 3-AP indices it belongs to
    three_aps_by_v = [[] for _ in range(N + 1)]
    for idx, mask in enumerate(three_aps):
        for v in range(1, N + 1):
            if mask & (1 << (v-1)):
                three_aps_by_v[v].append(idx)
    
    return three_aps, four_aps, four_aps_by_v, three_aps_by_v

def _count_3aps(mask, three_aps):
    """Count 3-APs fully contained in the set represented by mask."""
    count = 0
    for ap_mask in three_aps:
        if (mask & ap_mask) == ap_mask:
            count += 1
    return count

def _has_4ap(mask, four_aps_by_v, selected_set):
    """Check if the set has any 4-AP. Uses four_aps_by_v for efficiency."""
    for v in selected_set:
        for ap_idx in four_aps_by_v[v]:
            # Check if all elements of this 4-AP are in the set
            # We need to check the mask
            pass
    # Simpler: check all 4-APs
    # Actually, let's use a different approach
    return False

def _has_4ap_fast(mask, four_aps):
    """Check if the set has any 4-AP using precomputed masks."""
    for ap_mask in four_aps:
        if (mask & ap_mask) == ap_mask:
            return True
    return False

def construct():
    start_time = time.time()
    random.seed(42)
    
    three_aps, four_aps, four_aps_by_v, three_aps_by_v = _precompute()
    
    # Initial greedy: add elements that create 3-APs but no 4-APs
    selected_mask = 0
    selected_set = set()
    
    # Try to build a good initial set
    for x in range(1, N + 1):
        new_mask = selected_mask | (1 << (x-1))
        if not _has_4ap_fast(new_mask, four_aps):
            selected_mask = new_mask
            selected_set.add(x)
    
    best_mask = selected_mask
    best_count = _count_3aps(selected_mask, three_aps)
    
    # Local search: try swapping elements
    def try_swap_remove_add(mask, s, t):
        """Remove s, add t, check 4-AP constraint."""
        new_mask = (mask & ~(1 << (s-1))) | (1 << (t-1))
        return new_mask
    
    # Hill climbing with swaps
    improved = True
    while improved and (time.time() - start_time) < DEADLINE:
        improved = False
        selected_list = sorted(selected_set)
        
        # Try removing one element and adding another
        for s in selected_list:
            if (time.time() - start_time) > DEADLINE - 5:
                break
            mask_without_s = selected_mask & ~(1 << (s-1))
            
            # Try adding each element not in set
            for t in range(1, N + 1):
                if t in selected_set:
                    continue
                new_mask = mask_without_s | (1 << (t-1))
                if _has_4ap_fast(new_mask, four_aps):
                    continue
                new_count = _count_3aps(new_mask, three_aps)
                if new_count > best_count:
                    best_count = new_count
                    best_mask = new_mask
                    selected_mask = new_mask
                    selected_set.remove(s)
                    selected_set.add(t)
                    improved = True
                    break
        
        # Try adding a new element (if possible)
        if (time.time() - start_time) > DEADLINE - 5:
            break
        for t in range(1, N + 1):
            if t in selected_set:
                continue
            new_mask = selected_mask | (1 << (t-1))
            if _has_4ap_fast(new_mask, four_aps):
                continue
            new_count = _count_3aps(new_mask, three_aps)
            if new_count > best_count:
                best_count = new_count
                best_mask = new_mask
                selected_mask = new_mask
                selected_set.add(t)
                improved = True
    
    # Try multiple random restarts with different initial seeds
    if (time.time() - start_time) < DEADLINE - 10:
        for restart in range(20):
            if (time.time() - start_time) > DEADLINE - 5:
                break
            # Random initial set
            random_selected = set(random.sample(range(1, N+1), random.randint(20, 50)))
            rand_mask = 0
            for v in random_selected:
                rand_mask |= (1 << (v-1))
            
            # Remove 4-APs by removing elements
            changed = True
            while changed:
                changed = False
                for v in list(random_selected):
                    if (time.time() - start_time) > DEADLINE - 5:
                        break
                    test_mask = rand_mask & ~(1 << (v-1))
                    if not _has_4ap_fast(test_mask, four_aps):
                        rand_mask = test_mask
                        random_selected.remove(v)
                        changed = True
                    elif _has_4ap_fast(rand_mask, four_aps):
                        # Check if removing v helps
                        pass
            
            if not _has_4ap_fast(rand_mask, four_aps):
                rand_count = _count_3aps(rand_mask, three_aps)
                if rand_count > best_count:
                    best_count = rand_count
                    best_mask = rand_mask
                    selected_mask = rand_mask
                    selected_set = set(random_selected)
            
            # Local search from this starting point
            for _ in range(50):
                if (time.time() - start_time) > DEADLINE - 5:
                    break
                improved_local = False
                selected_list = sorted(selected_set)
                for s in selected_list:
                    mask_without_s = selected_mask & ~(1 << (s-1))
                    for t in range(1, N + 1):
                        if t in selected_set:
                            continue
                        new_mask = mask_without_s | (1 << (t-1))
                        if _has_4ap_fast(new_mask, four_aps):
                            continue
                        new_count = _count_3aps(new_mask, three_aps)
                        if new_count > best_count:
                            best_count = new_count
                            best_mask = new_mask
                            selected_mask = new_mask
                            selected_set.remove(s)
                            selected_set.add(t)
                            improved_local = True
                            break
                    if improved_local:
                        break
                if not improved_local:
                    break
    
    # Convert best_mask back to sorted list
    result = []
    for v in range(1, N + 1):
        if best_mask & (1 << (v-1)):
            result.append(v)
    
    return sorted(result)
