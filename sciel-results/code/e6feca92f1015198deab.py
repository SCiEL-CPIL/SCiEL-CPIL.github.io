import time
import random

DEADLINE = 85.0

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    N = 512

    # Precompute all 4-APs
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))

    # For each element, which APs does it belong to
    elem_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for y in ap:
            elem_aps[y].append(idx)

    # Fast check: can we add x to bitset without creating a 4-AP?
    def can_add(bitset, x):
        for ap_idx in elem_aps[x]:
            a, b, c, d = aps[ap_idx]
            # Check if the other 3 are all in bitset
            others = [y for y in (a, b, c, d) if y != x]
            if (bitset >> others[0]) & 1 and (bitset >> others[1]) & 1 and (bitset >> others[2]) & 1:
                return False
        return True

    # Initial solution: Cantor set (base-4 digits 0,1) + greedy fill
    def cantor_set():
        result = set()
        for x in range(N):
            v = x
            ok = True
            while v > 0:
                if v % 4 >= 2:
                    ok = False
                    break
                v //= 4
            if ok:
                result.add(x)
        return result

    cant = cantor_set()
    bitset = 0
    for x in cant:
        bitset |= (1 << x)

    rng = random.Random(42)
    order = list(range(N))
    rng.shuffle(order)
    for x in order:
        if (bitset >> x) & 1:
            continue
        if can_add(bitset, x):
            bitset |= (1 << x)

    best_bitset = bitset
    best_size = bin(bitset).count('1')

    # Different strategy: "Repair and Extend" via constraint propagation
    # 1. Start with a random dense set
    # 2. Repair: remove elements to eliminate all 4-APs (using efficient tracking)
    # 3. Extend: greedily add back elements
    # 4. Local search: 1-flip and 2-flip moves

    for restart in range(200):
        if time.time() > deadline - 2.0:
            break

        rng_r = random.Random(42 + restart * 13)

        # Phase 1: Start with a random subset of density ~0.6
        bitset = 0
        for x in range(N):
            if rng_r.random() < 0.6:
                bitset |= (1 << x)

        # Phase 2: Repair - remove elements to make it 4-AP-free
        # Use a queue of violated APs
        # For efficiency, track for each AP how many elements are in the set
        ap_sat = [0] * len(aps)  # number of elements of AP in the set
        for x in range(N):
            if (bitset >> x) & 1:
                for ap_idx in elem_aps[x]:
                    ap_sat[ap_idx] += 1

        # Find violated APs and remove elements
        changed = True
        while changed:
            changed = False
            for ap_idx in range(len(aps)):
                if ap_sat[ap_idx] == 4:
                    # Remove a random element from this AP
                    a, b, c, d = aps[ap_idx]
                    # Pick the element that's in the most other violated APs (most "guilty")
                    # For simplicity, pick randomly
                    x = rng_r.choice(aps[ap_idx])
                    bitset &= ~(1 << x)
                    for ap_idx2 in elem_aps[x]:
                        ap_sat[ap_idx2] -= 1
                    changed = True
                    break  # Restart scan after a removal

        # Phase 3: Extend - greedily add elements
        order = list(range(N))
        rng_r.shuffle(order)
        for x in order:
            if (bitset >> x) & 1:
                continue
            if can_add(bitset, x):
                bitset |= (1 << x)

        sz = bin(bitset).count('1')
        if sz > best_size:
            best_size = sz
            best_bitset = bitset

        # Phase 4: Local search - try to add elements
        for _ in range(5000):
            if time.time() > deadline - 1.0:
                break
            x = rng_r.randint(0, N - 1)
            if (bitset >> x) & 1:
                continue
            if can_add(bitset, x):
                bitset |= (1 << x)
                sz = bin(bitset).count('1')
                if sz > best_size:
                    best_size = sz
                    best_bitset = bitset

        # Phase 5: 2-flip local search (remove one, add two)
        for _ in range(2000):
            if time.time() > deadline - 1.0:
                break
            # Try: remove a random element, then try to add two others
            x_rem = rng_r.randint(0, N - 1)
            if not (bitset >> x_rem) & 1:
                continue
            temp = bitset & ~(1 << x_rem)
            # Try to add two random elements
            added = 0
            for _ in range(2):
                x_add = rng_r.randint(0, N - 1)
                if (temp >> x_add) & 1:
                    continue
                if can_add(temp, x_add):
                    temp |= (1 << x_add)
                    added += 1
            if added == 2:
                sz = bin(temp).count('1')
                if sz > best_size:
                    best_size = sz
                    best_bitset = temp
                    bitset = temp

    # Convert to 1-indexed list
    result = [i + 1 for i in range(N) if (best_bitset >> i) & 1]
    if not result:
        result = [1]
    return result
