import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 80.0  # 20s margin
    
    best_peak = 100
    best_A = None
    
    # Target interval
    target_start = 128
    target_end = 256
    target_range = np.arange(target_start, target_end + 1)
    
    # Try different sizes for A
    # Seed size was 65. Let's try smaller to reduce average multiplicity.
    sizes_to_try = [50, 55, 60, 45, 65]
    
    for N in sizes_to_try:
        if time.time() > deadline - 10:
            break
            
        # Initialize A with random elements
        # To help coverage, maybe bias towards middle?
        # Or just random.
        A = set(random.sample(range(129), N))
        
        # Ensure coverage
        # Calculate counts
        counts = np.zeros(257, dtype=np.int32)
        
        def update_counts_add(b, A_set, counts):
            # Add b to A_set
            # Pairs: (b, b) and (y, b) for y in A_set, y < b
            counts[2*b] += 1
            if A_set:
                # Get y < b
                ys = [y for y in A_set if y < b]
                if ys:
                    indices = b + np.array(ys, dtype=np.int32)
                    counts[indices] += 1
        
        def update_counts_remove(a, A_set, counts):
            # Remove a from A_set
            # Pairs: (a, a) and (a, x) for x in A_set, x > a
            counts[2*a] -= 1
            if A_set:
                # Get x > a
                xs = [x for x in A_set if x > a]
                if xs:
                    indices = a + np.array(xs, dtype=np.int32)
                    counts[indices] -= 1
        
        # Initial fill
        for a in A:
            update_counts_add(a, A, counts)
            # Note: update_counts_add assumes a is NOT in A_set yet for the loop part?
            # My logic above: "Add b to A_set". The loop `ys = [y for y in A_set if y < b]` uses the CURRENT A_set.
            # If I iterate `for a in A`, and call `update_counts_add(a, A, counts)`,
            # the first time, A is full. So it will add pairs with all other elements.
            # This is wrong. It will double count or count pairs with elements not yet "processed" in a way that depends on order?
            # No, if I just build it from scratch:
            # It's easier to calculate counts from scratch for initialization.
            
        # Recalculate counts from scratch for correctness
        counts = np.zeros(257, dtype=np.int32)
        A_list = list(A)
        for i in range(len(A_list)):
            a = A_list[i]
            # Pair (a, a)
            counts[2*a] += 1
            # Pairs (a, b) for b in A_list[i+1:]
            for b in A_list[i+1:]:
                counts[a+b] += 1
                
        # Check coverage and fill if necessary
        # If holes, add random elements until covered
        max_fill_steps = 50
        for _ in range(max_fill_steps):
            if time.time() > deadline - 5:
                break
            holes = target_range[counts[target_range] == 0]
            if len(holes) == 0:
                break
            # Pick a random hole
            h = random.choice(holes)
            # Find a candidate b to add such that b + x = h for some x in A
            # Or just add a random b and see if it helps
            # Better: find b such that h-b is in A
            candidates = []
            for x in A:
                b = h - x
                if 0 <= b <= 128 and b not in A:
                    candidates.append(b)
            if candidates:
                b = random.choice(candidates)
                A.add(b)
                # Update counts
                update_counts_add(b, A, counts)
                # Note: update_counts_add logic:
                # It adds (b,b) and (y,b) for y in A (which now includes b? No, A was updated before call?
                # In my loop above: `A.add(b)` then `update_counts_add(b, A, counts)`.
                # Inside `update_counts_add`, it iterates `A_set`. Since `b` is in `A_set`,
                # `ys = [y for y in A_set if y < b]` will include `y` where `y < b`.
                # It will NOT include `b` itself because `y < b`.
                # So it adds pairs `(y, b)` for `y < b`.
                # And `counts[2*b] += 1` adds `(b, b)`.
                # This is correct for adding `b` to a set `A` that does NOT contain `b`.
                # But I added `b` to `A` BEFORE calling.
                # So `A` contains `b`.
                # The loop `ys = [y for y in A_set if y < b]` will correctly find all `y` in the OLD A plus any new ones?
                # No, `A` is the set. It now contains `b`.
                # The pairs involving `b` are `(b, y)` for `y \in A \setminus \{b\}` with `y < b`?
                # And `(b, b)`.
                # The logic in `update_counts_add` adds `(b, b)` and `(y, b)` for `y \in A, y < b`.
                # Since `b \in A`, the condition `y < b` excludes `b` itself.
                # So it adds pairs with all other elements `y < b`.
                # This is correct.
            else:
                # No candidate from existing A. Add random element
                b = random.randint(0, 128)
                if b not in A:
                    A.add(b)
                    update_counts_add(b, A, counts)

        # Local Search
        current_peak = np.max(counts[target_range])
        if current_peak < best_peak:
            best_peak = current_peak
            best_A = set(A)
            
        # SA / Local Search
        # We want to minimize peak.
        # Try swaps
        num_steps = 2000
        for step in range(num_steps):
            if time.time() > deadline - 2:
                break
            
            # Pick random a in A
            a = random.choice(list(A))
            # Pick random b not in A
            # To speed up, precompute list of not_in_A?
            # Or just sample until found
            b = random.randint(0, 128)
            while b in A:
                b = random.randint(0, 128)
                
            # Evaluate swap a -> b
            # Temporarily remove a
            update_counts_remove(a, A, counts)
            A.remove(a)
            
            # Temporarily add b
            update_counts_add(b, A, counts)
            A.add(b)
            
            # Check validity
            if np.all(counts[target_range] > 0):
                new_peak = np.max(counts[target_range])
                
                # Acceptance criteria
                accept = False
                if new_peak < current_peak:
                    accept = True
                elif new_peak == current_peak:
                    if random.random() < 0.3:
                        accept = True
                elif new_peak == current_peak + 1:
                    if random.random() < 0.05:
                        accept = True
                # else reject
                
                if accept:
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = set(A)
                else:
                    # Undo
                    update_counts_remove(b, A, counts)
                    A.remove(b)
                    update_counts_add(a, A, counts)
                    A.add(a)
            else:
                # Invalid, undo
                update_counts_remove(b, A, counts)
                A.remove(b)
                update_counts_add(a, A, counts)
                A.add(a)

    # Return best_A
    if best_A is None:
        # Fallback to seed
        return list(range(64, 129))
    return list(best_A)
