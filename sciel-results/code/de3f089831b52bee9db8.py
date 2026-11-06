"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    n_max = 4096

    def build_sums(selected):
        """Build the set of all triple sums for a given list."""
        sums = set()
        k = len(selected)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    sums.add(selected[i] + selected[j] + selected[l])
        return sums

    def can_add_fast(x, selected, sums):
        """Check if x can be added to selected without triple sum collisions.
        Uses set membership for O(1) checks on precomputed pair sums."""
        # 3x
        if 3 * x in sums:
            return False
        # 2x + s for s in selected
        for s in selected:
            if 2 * x + s in sums:
                return False
        # x + s + t for s <= t in selected
        k = len(selected)
        for i in range(k):
            s = selected[i]
            for j in range(i, k):
                t = selected[j]
                if x + s + t in sums:
                    return False
        return True

    def add_x_fast(x, selected, sums):
        """Add x to selected and update sums incrementally."""
        selected.append(x)
        sums.add(3 * x)
        for s in selected[:-1]:
            sums.add(2 * x + s)
        k = len(selected)
        for i in range(k - 1):
            s = selected[i]
            for j in range(i, k - 1):
                t = selected[j]
                sums.add(x + s + t)

    def greedy_with_order(order, start_list=None):
        """Greedy fill using a custom order of candidates."""
        if start_list is None:
            start_list = []
        selected = list(start_list)
        sums = build_sums(selected)
        existing = set(selected)
        
        for x in order:
            if time.time() > deadline:
                break
            if x in existing:
                continue
            if can_add_fast(x, selected, sums):
                add_x_fast(x, selected, sums)
                existing.add(x)
        return selected

    def greedy_forward(start_list=None):
        """Greedy forward from 1 to n_max."""
        order = list(range(1, n_max + 1))
        return greedy_with_order(order, start_list)

    def greedy_backward(start_list=None):
        """Greedy backward from n_max to 1."""
        order = list(range(n_max, 0, -1))
        return greedy_with_order(order, start_list)

    def greedy_random(order_seed, start_list=None):
        """Greedy with random order."""
        rng = random.Random(order_seed)
        order = list(range(1, n_max + 1))
        rng.shuffle(order)
        return greedy_with_order(order, start_list)

    def remove_and_refill(selected, sums, remove_idx, candidate_order):
        """Remove element at remove_idx and try to refill with best candidate."""
        orig = selected[remove_idx]
        selected_new = selected[:remove_idx] + selected[remove_idx + 1:]
        sums_new = build_sums(selected_new)
        existing_new = set(selected_new)
        
        best_x = None
        for x in candidate_order:
            if time.time() > deadline:
                break
            if x in existing_new:
                continue
            if can_add_fast(x, selected_new, sums_new):
                best_x = x
                break
        
        if best_x is not None:
            selected_new.append(best_x)
            sums_new = build_sums(selected_new)
            return selected_new, sums_new, True
        return None, None, False

    def local_improve(selected, max_iterations=100):
        """Improve by removing one element and refilling."""
        if time.time() > deadline:
            return selected
        
        selected = list(selected)
        sums = build_sums(selected)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            n = len(selected)
            
            # Try removing each element
            for idx in range(n):
                if time.time() > deadline:
                    break
                
                # Try refilling with smallest available candidate
                candidate_order = list(range(1, n_max + 1))
                new_sel, new_sums, success = remove_and_refill(selected, sums, idx, candidate_order)
                
                if success:
                    # Check if this is an improvement (smaller element)
                    if new_sel[idx] < selected[idx]:
                        selected = new_sel
                        sums = new_sums
                        improved = True
                        break
                    elif new_sel[idx] != selected[idx]:
                        # Same size or larger, but different - accept if it helps future fills
                        # For now, only accept if smaller
                        pass
            
            if not improved:
                break
        
        return selected

    # Strategy 1: Start with sparse skeleton (powers of 4) and greedy fill
    base = [4**i for i in range(10) if 4**i <= n_max]
    
    best_set = []
    best_len = 0

    # Try forward greedy from sparse base
    s1 = greedy_forward(start_list=base)
    if len(s1) > best_len:
        best_len = len(s1)
        best_set = s1[:]
    
    if time.time() > deadline:
        return best_set[:80]

    # Try backward greedy from sparse base
    s2 = greedy_backward(start_list=base)
    if len(s2) > best_len:
        best_len = len(s2)
        best_set = s2[:]

    if time.time() > deadline:
        return best_set[:80]

    # Try forward greedy without base
    s3 = greedy_forward()
    if len(s3) > best_len:
        best_len = len(s3)
        best_set = s3[:]

    if time.time() > deadline:
        return best_set[:80]

    # Try backward greedy without base
    s4 = greedy_backward()
    if len(s4) > best_len:
        best_len = len(s4)
        best_set = s4[:]

    if time.time() > deadline:
        return best_set[:80]

    # Local improvement on best solution
    if time.time() < deadline - 10:
        improved = local_improve(best_set, max_iterations=50)
        if len(improved) > best_len:
            best_len = len(improved)
            best_set = improved

    # Randomized restarts with different skeletons and orders
    random.seed(42)
    for attempt in range(40):
        if time.time() > deadline - 3:
            break
        
        # Random sparse skeleton
        skeleton_size = random.randint(3, 12)
        skeleton = random.sample(range(1, n_max + 1), skeleton_size)
        skeleton.sort()
        
        # Verify skeleton is valid
        sk_sums = build_sums(skeleton)
        if len(sk_sums) != skeleton_size * (skeleton_size + 1) * (skeleton_size + 2) // 6:
            continue
        
        # Try different fill strategies
        if attempt % 3 == 0:
            result = greedy_forward(start_list=skeleton)
        elif attempt % 3 == 1:
            result = greedy_backward(start_list=skeleton)
        else:
            result = greedy_random(attempt * 1000 + 42, start_list=skeleton)
        
        if len(result) > best_len:
            best_len = len(result)
            best_set = result[:]

    # Final local improvement pass
    if time.time() < deadline - 2:
        improved = local_improve(best_set, max_iterations=30)
        if len(improved) > best_len:
            best_len = len(improved)
            best_set = improved

    return best_set[:80]
# EVOLVE-BLOCK-END
