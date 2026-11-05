import math
import time

def construct():
    N = 512
    start_time = time.time()
    time_limit = 85.0

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

    # Strong initial solution: greedy maximal clique on compatibility graph
    # This is equivalent to greedy independent set on conflict graph
    def greedy_clique():
        chosen = []
        remaining = full_mask
        while remaining:
            # Pick vertex with max compatibility degree in remaining
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
            remaining &= ~comp[best_v]  # Remove incompatible vertices
        return chosen

    # Try multiple greedy orderings to get a better initial solution
    best_init = []
    for order_strategy in range(3):
        chosen = []
        remaining = full_mask
        while remaining:
            best_v = -1
            best_score = -1
            temp = remaining
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (comp[v] & remaining).bit_count()
                if order_strategy == 0:
                    score = deg
                elif order_strategy == 1:
                    score = -v  # Prefer smaller values
                else:
                    score = deg - v // 10
                if score > best_score:
                    best_score = score
                    best_v = v
                temp ^= lsb
            if best_v == -1:
                break
            chosen.append(best_v)
            remaining &= ~(1 << best_v)
            remaining &= ~comp[best_v]
        if len(chosen) > len(best_init):
            best_init = chosen

    best_size = len(best_init)
    best_set = list(best_init)

    # Fast greedy coloring bound for max clique in compatibility graph
    # Returns an upper bound on clique size in the subgraph induced by mask
    def coloring_bound(mask, limit):
        """Upper bound on max clique size in comp graph induced by mask.
        If we can prove bound <= limit, return early."""
        colors = 0
        uncolored = mask
        while uncolored:
            colors += 1
            if colors >= limit:
                return colors
            # Greedy: pick vertex with max degree in uncolored for coloring
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
            # Color class: vertices in uncolored not adjacent to best_v in comp graph
            # In comp graph, adjacent means compatible. So non-adjacent means incompatible.
            # A color class in proper coloring of comp graph: no two vertices are adjacent in comp.
            # So color class = {v in uncolored : v not in comp[best_v]} ∪ {best_v}
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

    # Run B&B
    init_mask = full_mask
    solve(init_mask, 0, [])

    # Ensure we have at least the initial solution
    if len(best_set) < len(best_init):
        best_set = best_init

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
