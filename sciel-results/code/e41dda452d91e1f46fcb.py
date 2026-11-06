"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 28.0
    best_size = N + 1
    best_set = None

    def can_add(x, selected, sums_mask):
        """Check if x can be added to the Sidon set."""
        if (sums_mask >> (2 * x)) & 1:
            return False
        for a in selected:
            if (sums_mask >> (x + a)) & 1:
                return False
        return True

    def get_addable(selected, sums_mask):
        """Get list of all elements that can be added."""
        addable = []
        selected_set = set(selected)
        for x in range(1, N + 1):
            if x not in selected_set and can_add(x, selected, sums_mask):
                addable.append(x)
        return addable

    def fast_block_score(x, selected, sums_mask, addable):
        """Quick score: count how many addable elements become non-addable after adding x."""
        new_sums = 1 << (2 * x)
        for a in selected:
            new_sums |= 1 << (x + a)
        
        blocked = 0
        for y in addable:
            if y == x:
                continue
            if (new_sums >> (2 * y)) & 1:
                blocked += 1
                continue
            blocked_y = False
            for a in selected:
                if (new_sums >> (y + a)) & 1:
                    blocked_y = True
                    break
            if not blocked_y and (sums_mask >> (y + x)) & 1:
                blocked_y = True
            if blocked_y:
                blocked += 1
        return blocked

    def backtrack(selected, sums_mask, addable_cache):
        nonlocal best_size, best_set
        if time.time() > deadline:
            return

        # Recompute addable if cache is None or stale
        if addable_cache is None:
            addable = get_addable(selected, sums_mask)
        else:
            addable = addable_cache

        if not addable:
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = list(selected)
            return

        if len(selected) >= best_size:
            return

        # Score candidates by blocking potential, but limit work
        scored = []
        for x in addable:
            blocked = fast_block_score(x, selected, sums_mask, addable)
            scored.append((blocked, x))

        scored.sort(key=lambda t: (-t[0], t[1]))

        # Try top candidates - widen branching
        for _, x in scored[:8]:
            new_sums = sums_mask | (1 << (2 * x))
            for a in selected:
                new_sums |= 1 << (x + a)
            
            selected.append(x)
            backtrack(selected, new_sums, None)
            selected.pop()
            
            if time.time() > deadline:
                return

    # Fast greedy baseline
    selected_greedy = []
    sums_mask_greedy = 0
    remaining_set = set(range(1, N + 1))
    
    while remaining_set:
        if time.time() > deadline - 5:
            break
        best_x = None
        best_block = -1
        for x in sorted(remaining_set):
            if not can_add(x, selected_greedy, sums_mask_greedy):
                continue
            new_sums = sums_mask_greedy | (1 << (2 * x))
            for a in selected_greedy:
                new_sums |= 1 << (x + a)
            # Quick block count
            blocked = 0
            for y in remaining_set:
                if y == x:
                    continue
                if (new_sums >> (2 * y)) & 1:
                    blocked += 1
                    continue
                ok = True
                for a in selected_greedy:
                    if (new_sums >> (y + a)) & 1:
                        ok = False
                        break
                if ok and (sums_mask_greedy >> (y + x)) & 1:
                    ok = False
                if not ok:
                    blocked += 1
            if blocked > best_block:
                best_block = blocked
                best_x = x
                if blocked > 50:
                    break  # Early termination if very high blocking
        if best_x is None:
            break
        selected_greedy.append(best_x)
        sums_mask_greedy |= 1 << (2 * best_x)
        for a in selected_greedy[:-1]:
            sums_mask_greedy |= 1 << (best_x + a)
        remaining_set.discard(best_x)

    if len(selected_greedy) < best_size:
        best_size = len(selected_greedy)
        best_set = list(selected_greedy)

    # Multiple backtracking restarts with different seeds
    for seed in [42, 123, 456, 789, 321, 654, 999, 111, 222, 333, 444, 555]:
        if time.time() > deadline:
            break
        random.seed(seed)
        selected_bt = []
        backtrack(selected_bt, 0, None)

    if best_set is not None:
        return best_set
    else:
        return selected_greedy
# EVOLVE-BLOCK-END
