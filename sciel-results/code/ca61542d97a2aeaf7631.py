"""Sidon set: small inclusion-maximal subset of [1,256]."""
import time
import random

N = 256
DEADLINE = time.time() + 58.0  # safety margin before 60s kill

def construct():
    start = time.time()
    best_size = N + 1
    best_set = None

    rng = random.Random(42)

    # We'll try many random orderings with a greedy "max-block" heuristic.
    # The idea: at each step, pick the candidate x that, when added, blocks
    # the most other remaining candidates (making them no longer addable).
    # This tends to produce smaller maximal sets.

    def is_valid_addition(x, selected, sums):
        """Check if x can be added to the current Sidon set."""
        new_sums = set()
        for a in selected:
            s = x + a
            if s in sums:
                return False
            new_sums.add(s)
        s2 = 2 * x
        if s2 in sums:
            return False
        # Check no duplicate among new_sums and s2
        if s2 in new_sums:
            return False
        return True

    def blocked_count(x, selected, sums, candidates):
        """Count how many candidates y in `candidates` would be blocked if we add x."""
        # After adding x, the new sums are {x+a for a in selected} union {2x}
        new_sums = set()
        for a in selected:
            new_sums.add(x + a)
        new_sums.add(2 * x)
        
        # A candidate y is blocked if adding y to (selected + [x]) would violate Sidon.
        # y is blocked if:
        #  - y + a is in sums or new_sums for some a in selected (i.e., y+a conflicts)
        #  - 2y is in sums or new_sums
        #  - y + x is in sums (already checked since y+x is a new sum that must not be in old sums)
        #  - y + a is in new_sums for some a in selected (y+a = x+a' for some a' => y = x + a' - a)
        
        # More precisely, y is blocked if:
        # 1. For some a in selected: y+a in sums OR y+a in new_sums
        # 2. 2y in sums OR 2y in new_sums
        # 3. y+x in sums (since y+x is a sum involving the new element x)
        # 4. y+x in new_sums (but y+x = x+x' only if y=x', which isn't in candidates)
        
        count = 0
        for y in candidates:
            if y == x:
                continue
            blocked = False
            # Check y+x against existing sums
            if (y + x) in sums:
                blocked = True
            else:
                # Check y+a for a in selected against sums and new_sums
                for a in selected:
                    s = y + a
                    if s in sums or s in new_sums:
                        blocked = True
                        break
                if not blocked:
                    # Check 2y
                    if (2 * y) in sums or (2 * y) in new_sums:
                        blocked = True
            if blocked:
                count += 1
        return count

    def greedy_max_block(order):
        """Greedy: process candidates in given order, but at each step pick the one
        that blocks the most remaining candidates. Returns the constructed set."""
        selected = []
        sums = set()
        candidates = list(order)
        
        while candidates:
            if time.time() > DEADLINE:
                break
            # Pick candidate that blocks the most others
            best_idx = 0
            best_block = -1
            for i, x in enumerate(candidates):
                if not is_valid_addition(x, selected, sums):
                    continue
                bc = blocked_count(x, selected, sums, candidates)
                if bc > best_block:
                    best_block = bc
                    best_idx = i
            if best_block < 0:
                break
            x = candidates.pop(best_idx)
            selected.append(x)
            for a in selected[:-1]:
                sums.add(x + a)
            sums.add(2 * x)
        
        # Check maximality
        selected_set = set(selected)
        all_sums = set()
        for i in range(len(selected)):
            for j in range(i, len(selected)):
                all_sums.add(selected[i] + selected[j])
        
        is_maximal = True
        for y in range(1, N + 1):
            if y in selected_set:
                continue
            # Check if y can be added
            can_add = True
            new_sums = set()
            for a in selected:
                s = y + a
                if s in all_sums:
                    can_add = False
                    break
                new_sums.add(s)
            if can_add:
                s2 = 2 * y
                if s2 in all_sums or s2 in new_sums:
                    can_add = False
            if can_add:
                is_maximal = False
                break
        
        if is_maximal and len(selected) < best_size:
            return selected
        return None

    # Try multiple random permutations
    base = list(range(1, N + 1))
    attempts = 0
    while time.time() < DEADLINE - 2.0:
        attempts += 1
        perm = base[:]
        rng.shuffle(perm)
        result = greedy_max_block(perm)
        if result is not None:
            best_size = min(best_size, len(result))
            best_set = result
    
    if best_set is None:
        # Fallback: ascending greedy
        selected = []
        sums = set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        best_set = selected
        best_size = len(selected)

    return best_set
