import math
import time

def construct():
    N = 512
    start_time = time.time()
    time_limit = 80.0

    # Precompute conflict graph: adj[i] has bit j set if lcm(i,j) > N
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            if i * j // g > N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Compatibility graph: comp[i] has bit j set if lcm(i,j) <= N (i != j)
    full_mask = (1 << (N + 1)) - 2
    comp = [0] * (N + 1)
    for i in range(1, N + 1):
        comp[i] = full_mask & ~adj[i] & ~(1 << i)

    # Strong initial solution: divisors of 504 (lcm of any two divides 504 <= 512)
    divisors_504 = [d for d in range(1, N + 1) if 504 % d == 0]
    # Verify validity
    valid_504 = True
    for i in range(len(divisors_504)):
        for j in range(i + 1, len(divisors_504)):
            a, b = divisors_504[i], divisors_504[j]
            if a * b // math.gcd(a, b) > N:
                valid_504 = False
                break
        if not valid_504:
            break
    if not valid_504:
        # Fallback to greedy
        divisors_504 = [1]

    best_size = len(divisors_504)
    best_set = list(divisors_504)

    # Also try greedy to see if we can do better as initial
    def greedy_clique():
        chosen = []
        remaining = full_mask
        while remaining:
            best_v = -1
            max_deg = -1
            temp = remaining
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (comp[v] & remaining).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= lsb
            if best_v == -1:
                break
            chosen.append(best_v)
            remaining &= ~(1 << best_v)
            remaining &= ~comp[best_v]
        return chosen

    greedy_sol = greedy_clique()
    if len(greedy_sol) > best_size:
        best_size = len(greedy_sol)
        best_set = list(greedy_sol)

    # Fast coloring bound for max clique in compatibility graph
    def coloring_bound(mask, limit):
        """Upper bound on max clique size in comp graph induced by mask."""
        colors = 0
        uncolored = mask
        while uncolored:
            colors += 1
            if colors >= limit:
                return colors
            # Greedy: pick vertex with max degree in uncolored
            best_v = -1
            max_deg = -1
            temp = uncolored
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (comp[v] & uncolored).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= lsb
            if best_v == -1:
                break
            color_class = uncolored & ~comp[best_v]
            color_class |= (1 << best_v)
            uncolored &= ~color_class
        return colors

    # B&B solver for maximum clique in compatibility graph
    def solve(mask, current_size, current_clique):
        nonlocal best_size, best_set

        if time.time() - start_time > time_limit:
            return

        # Upper bound: can we beat best_size?
        count = mask.bit_count()
        if current_size + count <= best_size:
            return

        # Tighter bound via coloring
        needed = best_size - current_size + 1
        ub = coloring_bound(mask, needed)
        if current_size + ub <= best_size:
            return

        if mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_clique)
            return

        # Pick vertex with max compatibility degree in mask for branching
        best_v = -1
        max_deg = -1
        temp = mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (comp[v] & mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= lsb

        if best_v == -1:
            return

        # Branch 1: Include best_v in clique
        new_mask = mask & ~(1 << best_v) & comp[best_v]
        current_clique.append(best_v)
        solve(new_mask, current_size + 1, current_clique)
        current_clique.pop()

        # Branch 2: Exclude best_v
        new_mask = mask & ~(1 << best_v)
        solve(new_mask, current_size, current_clique)

    # Run B&B starting from the initial solution
    init_mask = full_mask
    solve(init_mask, 0, [])

    return sorted(best_set)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            l = a * b // math.gcd(a, b)
            if l > 512:
                print(f"Invalid: {a}, {b}, lcm={l}")
                break
