"""Rewritten program: greedy B_3[1] construction with incremental triple-sum check."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard: finish well before 60s evaluator limit
    deadline = time.time() + 55.0

    # We want up to 80 distinct positive integers in [1, 4096]
    # such that every unordered triple (with repetitions) has a unique sum.

    selected = []
    existing_sums = set()  # all triple sums from selected elements

    # Helper to add triple sums involving new candidate x to the set
    def try_add(x):
        """Check if x can be added. Returns True if yes, and updates state."""
        nonlocal selected, existing_sums

        # Generate all new triple sums involving x
        # Unordered triples with repetitions from {x} ∪ selected:
        # Types: (x,x,x), (x,x,y), (x,y,y), (x,y,z) where y,z ∈ selected, y<=z
        # We need to check these against existing_sums and among themselves.

        new_sums = set()

        # (x, x, x)
        s = 3 * x
        if s in existing_sums or s in new_sums:
            return False
        new_sums.add(s)

        # (x, x, y) for y in selected
        for y in selected:
            s = 2 * x + y
            if s in existing_sums or s in new_sums:
                return False
            new_sums.add(s)

        # (x, y, y) for y in selected
        for y in selected:
            s = x + 2 * y
            if s in existing_sums or s in new_sums:
                return False
            new_sums.add(s)

        # (x, y, z) for y <= z in selected
        for i, y in enumerate(selected):
            for z in selected[i:]:
                s = x + y + z
                if s in existing_sums or s in new_sums:
                    return False
                new_sums.add(s)

        # All good, commit
        existing_sums.update(new_sums)
        selected.append(x)
        return True

    # Greedy: try each candidate in [1, 4096] in increasing order
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        if len(selected) >= 80:
            break
        try_add(x)

    return selected
# EVOLVE-BLOCK-END
