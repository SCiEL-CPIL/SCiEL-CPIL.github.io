import time
import random
import math

def construct():
    N = 256
    elements = list(range(1, N + 1))
    
    # Precompute 4-APs
    aps4 = []
    aps4_by_elem = {i: [] for i in range(1, N + 1)}
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            p = (a, a + d, a + 2 * d, a + 3 * d)
            idx = len(aps4)
            aps4.append(p)
            for x in p:
                aps4_by_elem[x].append(idx)
    num_aps4 = len(aps4)
    
    # Precompute 3-APs
    aps3 = []
    aps3_by_elem = {i: [] for i in range(1, N + 1)}
    for d in range(1, (N - 2) // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            p = (a, a + d, a + 2 * d)
            idx = len(aps3)
            aps3.append(p)
            for x in p:
                aps3_by_elem[x].append(idx)
    num_aps3 = len(aps3)

    # State
    A = set()
    c4 = [0] * num_aps4
    c3 = [0] * num_aps3
    score3 = 0
    
    def add_elem(x):
        nonlocal score3
        # Check 4-AP validity
        for idx in aps4_by_elem[x]:
            if c4[idx] == 3:
                return False
        # Update state
        A.add(x)
        for idx in aps4_by_elem[x]:
            c4[idx] += 1
        for idx in aps3_by_elem[x]:
            c3[idx] += 1
            if c3[idx] == 3:
                score3 += 1
        return True

    def remove_elem(x):
        nonlocal score3
        if x not in A:
            return False
        A.remove(x)
        for idx in aps4_by_elem[x]:
            c4[idx] -= 1
        for idx in aps3_by_elem[x]:
            if c3[idx] == 3:
                score3 -= 1
            c3[idx] -= 1
        return True

    def try_swap(x, y):
        # Remove x, add y
        if x not in A or y in A:
            return False
        # Check if y can be added to A \ {x}
        # Temporarily remove x
        remove_elem(x)
        if add_elem(y):
            return True
        else:
            # Revert: remove y, add x
            remove_elem(y)
            add_elem(x)
            return False

    # Initialization: Greedy random
    random.seed(42)
    # Start with a random subset of ~150 elements, then repair
    # Or just greedy add
    # Greedy add is fast and gives a valid start
    elems = elements.copy()
    random.shuffle(elems)
    for x in elems:
        if add_elem(x):
            pass
        # Stop if we have a decent size? No, let's try to add all possible
    # This greedy approach might be biased. Let's restart with randomization
    # Actually, let's just run SA from this start.
    
    # Better init: Random set of size 100, repair
    A.clear()
    c4 = [0] * num_aps4
    c3 = [0] * num_aps3
    score3 = 0
    
    # Add random elements, skipping if they create 4-AP
    elems = elements.copy()
    random.shuffle(elems)
    for x in elems:
        # Check if adding x creates a 4-AP
        valid = True
        for idx in aps4_by_elem[x]:
            if c4[idx] == 3:
                valid = False
                break
        if valid:
            add_elem(x)
            
    # Simulated Annealing
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    T = 1.0
    best_score = score3
    best_A = A.copy()
    
    # Estimate move cost for cooling
    # Each move touches ~20-50 APs. ~1000 moves/sec is safe.
    
    while time.time() < deadline:
        # Generate move
        if len(A) < 10:
            # Try to add
            candidates = [x for x in elements if x not in A]
            if not candidates:
                break
            x = random.choice(candidates)
            if add_elem(x):
                if score3 > best_score:
                    best_score = score3
                    best_A = A.copy()
            # Always accept add if valid in this phase? No, SA handles it.
            # But add_elem already applied. We need to revert if rejected.
            # Let's structure moves to return delta and allow revert.
            
            # Re-structure for clarity
            # Let's just do Add/Remove/Swap with explicit revert logic.
            pass
        else:
            move_type = random.random()
            if move_type < 0.3:
                # Add
                candidates = [x for x in elements if x not in A]
                if not candidates:
                    continue
                x = random.choice(candidates)
                # Check validity
                valid = True
                for idx in aps4_by_elem[x]:
                    if c4[idx] == 3:
                        valid = False
                        break
                if not valid:
                    continue
                # Apply
                add_elem(x)
                delta = score3 - best_score # Not quite, we need current score vs previous
                # We need to store prev_score
                # Let's track current_score
            elif move_type < 0.6:
                # Remove
                x = random.choice(list(A))
                prev_score = score3
                remove_elem(x)
                delta = score3 - prev_score
                # Acceptance
                if delta > 0:
                    if score3 > best_score:
                        best_score = score3
                        best_A = A.copy()
                else:
                    if random.random() < math.exp(delta / T):
                        if score3 > best_score:
                            best_score = score3
                            best_A = A.copy()
                    else:
                        # Revert
                        add_elem(x)
            else:
                # Swap
                if len(A) == 0:
                    continue
                x = random.choice(list(A))
                candidates = [y for y in elements if y not in A]
                if not candidates:
                    continue
                y = random.choice(candidates)
                prev_score = score3
                if try_swap(x, y):
                    delta = score3 - prev_score
                    if delta > 0:
                        if score3 > best_score:
                            best_score = score3
                            best_A = A.copy()
                    else:
                        if random.random() < math.exp(delta / T):
                            if score3 > best_score:
                                best_score = score3
                                best_A = A.copy()
                        else:
                            # Revert swap
                            try_swap(y, x)
        
        # Cooling
        T *= 0.999
        if T < 0.01:
            T = 0.01

    return sorted(best_A)

# Refining the SA loop to be cleaner and handle the 'Add' case correctly with revert.

def construct():
    N = 256
    elements = list(range(1, N + 1))
    
    # Precompute 4-APs
    aps4 = []
    aps4_by_elem = {i: [] for i in range(1, N + 1)}
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            p = (a, a + d, a + 2 * d, a + 3 * d)
            idx = len(aps4)
            aps4.append(p)
            for x in p:
                aps4_by_elem[x].append(idx)
    num_aps4 = len(aps4)
    
    # Precompute 3-APs
    aps3 = []
    aps3_by_elem = {i: [] for i in range(1, N + 1)}
    for d in range(1, (N - 2) // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            p = (a, a + d, a + 2 * d)
            idx = len(aps3)
            aps3.append(p)
            for x in p:
                aps3_by_elem[x].append(idx)
    num_aps3 = len(aps3)

    # State
    A = set()
    c4 = [0] * num_aps4
    c3 = [0] * num_aps3
    score3 = 0
    
    def can_add(x):
        for idx in aps4_by_elem[x]:
            if c4[idx] == 3:
                return False
        return True

    def do_add(x):
        nonlocal score3
        A.add(x)
        for idx in aps4_by_elem[x]:
            c4[idx] += 1
        for idx in aps3_by_elem[x]:
            c3[idx] += 1
            if c3[idx] == 3:
                score3 += 1

    def do_remove(x):
        nonlocal score3
        A.remove(x)
        for idx in aps4_by_elem[x]:
            c4[idx] -= 1
        for idx in aps3_by_elem[x]:
            if c3[idx] == 3:
                score3 -= 1
            c3[idx] -= 1

    def do_swap(x, y):
        # x in A, y not in A
        do_remove(x)
        do_add(y)

    # Init
    random.seed(42)
    elems = elements.copy()
    random.shuffle(elems)
    for x in elems:
        if can_add(x):
            do_add(x)
    
    start_time = time.time()
    deadline = start_time + 85.0
    T = 1.0
    best_score = score3
    best_A = A.copy()
    
    while time.time() < deadline:
        r = random.random()
        if r < 0.4:
            # Add
            candidates = [x for x in elements if x not in A]
            if not candidates:
                continue
            x = random.choice(candidates)
            if can_add(x):
                prev_score = score3
                do_add(x)
                delta = score3 - prev_score
                if delta > 0:
                    if score3 > best_score:
                        best_score = score3
                        best_A = A.copy()
                else:
                    if random.random() < math.exp(delta / T):
                        if score3 > best_score:
                            best_score = score3
                            best_A = A.copy()
                    else:
                        do_remove(x)
        elif r < 0.7:
            # Remove
            if not A:
                continue
            x = random.choice(list(A))
            prev_score = score3
            do_remove(x)
            delta = score3 - prev_score
            if delta > 0:
                if score3 > best_score:
                    best_score = score3
                    best_A = A.copy()
            else:
                if random.random() < math.exp(delta / T):
                    if score3 > best_score:
                        best_score = score3
                        best_A = A.copy()
                else:
                    do_add(x)
        else:
            # Swap
            if not A:
                continue
            x = random.choice(list(A))
            candidates = [y for y in elements if y not in A]
            if not candidates:
                continue
            y = random.choice(candidates)
            # Check if swap is valid (y can be added to A\{x})
            # Temporarily remove x
            do_remove(x)
            if can_add(y):
                prev_score = score3
                do_add(y)
                delta = score3 - prev_score
                if delta > 0:
                    if score3 > best_score:
                        best_score = score3
                        best_A = A.copy()
                else:
                    if random.random() < math.exp(delta / T):
                        if score3 > best_score:
                            best_score = score3
                            best_A = A.copy()
                    else:
                        # Revert
                        do_remove(y)
                        do_add(x)
            else:
                # Revert remove
                do_add(x)
        
        T *= 0.9995
        if T < 0.01:
            T = 0.01

    return sorted(best_A)

