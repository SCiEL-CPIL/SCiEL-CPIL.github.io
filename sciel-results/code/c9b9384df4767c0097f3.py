"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 25.0
    best_size = N + 1
    best_set = None

    def can_add(x, selected, sums_mask, selected_set):
        """Check if x can be added to the Sidon set."""
        if (sums_mask >> (2 * x)) & 1:
            return False
        for a in selected:
            if (sums_mask >> (x + a)) & 1:
                return False
        return True

    def get_addable(selected, sums_mask, selected_set):
        """Get list of all elements that can be added."""
        addable = []
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            if can_add(x, selected, sums_mask, selected_set):
                addable.append(x)
        return addable

    def score_element(x, selected, sums_mask, addable, selected_set):
        """Score element x by how many other addable elements it would block."""
        blocked = 0
        for y in addable:
            if y == x:
                continue
            # y is blocked if 2y is a new sum OR y+a is a new sum for any a in selected
            # New sums: 2x, x+a
            # 2y is new sum if 2y == 2x (y=x) or 2y == x+a => a = 2y-x
            if 2 * y == 2 * x:
                blocked += 1
                continue
            if (2 * y - x) in selected_set:
                blocked += 1
                continue
            
            # y+a is new sum if y+a == 2x => a = 2x-y
            # or y+a == x+b => a-b = x-y => b = a - (x-y)
            
            if (2 * x - y) in selected_set:
                blocked += 1
                continue
            
            d = x - y
            blocked_y = False
            for a in selected:
                if (a - d) in selected_set:
                    blocked_y = True
                    break
            if blocked_y:
                blocked += 1
        return blocked

    def backtrack(selected, sums_mask, selected_set):
        nonlocal best_size, best_set
        if time.time() > deadline:
            return

        addable = get_addable(selected, sums_mask, selected_set)

        if not addable:
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = list(selected)
            return

        if len(selected) >= best_size:
            return

        scored = []
        for x in addable:
            blocked = score_element(x, selected, sums_mask, addable, selected_set)
            scored.append((blocked, x))

        scored.sort(key=lambda t: (-t[0], t[1]))

        # Try top 10 candidates
        for _, x in scored[:10]:
            new_sums = sums_mask | (1 << (2 * x))
            for a in selected:
                new_sums |= 1 << (x + a)
            
            selected.append(x)
            selected_set.add(x)
            backtrack(selected, new_sums, selected_set)
            selected.pop()
            selected_set.remove(x)
            
            if time.time() > deadline:
                return

    # Greedy baseline
    selected_greedy = []
    sums_mask_greedy = 0
    selected_set_greedy = set()
    remaining = list(range(1, N + 1))
    while remaining:
        best_x = None
        best_block = -1
        for x in remaining:
            if not can_add(x, selected_greedy, sums_mask_greedy, selected_set_greedy):
                continue
            # Use a fast heuristic for greedy: just count how many remaining are blocked by x
            # We approximate by checking if x blocks many elements
            # For speed, we just pick the first valid one? No, let's use a simple score
            # Score: number of y in remaining such that y is blocked by x
            # This is O(N^2) per step, which is too slow.
            # Let's just pick the largest valid x? Or random?
            # Let's pick the one with max value to spread them out?
            # Actually, let's just pick the first valid one to make greedy fast.
            best_x = x
            break
        if best_x is None:
            break
        selected_greedy.append(best_x)
        selected_set_greedy.add(best_x)
        sums_mask_greedy |= 1 << (2 * best_x)
        for a in selected_greedy:
            sums_mask_greedy |= 1 << (best_x + a)
        remaining.remove(best_x)

    if len(selected_greedy) < best_size:
        best_size = len(selected_greedy)
        best_set = list(selected_greedy)

    # Backtracking search with multiple restarts
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline:
            break
        
        selected_bt = []
        selected_set_bt = set()
        backtrack(selected_bt, 0, selected_set_bt)

    if best_set is not None:
        return best_set
    else:
        return selected_greedy
# EVOLVE-BLOCK-END
