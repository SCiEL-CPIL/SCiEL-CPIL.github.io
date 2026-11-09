# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 60.0

    # Generate all 5-smooth numbers <= 128
    smooth = []
    a = 1
    while a <= 128:
        b = a
        while b <= 128:
            c = b
            while c <= 128:
                smooth.append(c)
                c *= 5
            b *= 3
        a *= 2
    smooth = sorted(set(smooth))
    n = len(smooth)

    # Precompute divisibility: divs[i] is a bitmask of j such that smooth[i] | smooth[j]
    divs = [0] * n
    for i, vi in enumerate(smooth):
        m = 0
        for j, vj in enumerate(smooth):
            if i != j and vj % vi == 0:
                m |= (1 << j)
        divs[i] = m

    # Union-find for transitive closure of divisibility
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    # Build equivalence classes: i ~ j if i|j or j|i
    for i in range(n):
        m = divs[i]
        while m:
            j = (m & -m).bit_length() - 1
            m &= m - 1
            union(i, j)

    # Group into chains (each chain is a total order under divisibility)
    chain_map = {}
    for i in range(n):
        r = find(i)
        chain_map.setdefault(r, []).append(i)

    chains = []
    for r, indices in chain_map.items():
        indices.sort(key=lambda i: smooth[i])
        chains.append(indices)

    # For each chain, compute prefix OR of divisibility masks
    # chain_prefix[ci][k] = OR of divs[chain[ci][i]] for i in range(k)
    chain_prefix = []
    for ci, chain in enumerate(chains):
        prefix = [0]
        for idx in chain:
            prefix.append(prefix[-1] | divs[idx])
        chain_prefix.append(prefix)

    # Bitmask-based backtracking over chains
    # best[s] = minimum number of terms for sum s
    best = [10000] * 129  # best[s] = min terms for sum s
    best[0] = 0

    # For each target, we need the actual terms, not just the count.
    # So we store the best solution per target.
    best_sol = [None] * 129  # best_sol[s] = list of terms or None

    # We'll do a global backtracking that explores all antichains and records
    # the minimum-size one for each sum.

    # To make this efficient, we process chains in a specific order and use
    # the chain structure to prune: within a chain, we can pick at most one
    # element, and the prefix OR tells us which elements are excluded.

    # Strategy: process chains in order. For each chain, decide which element (if any)
    # to pick. The constraint is that the picked element must not be in the
    # "forbidden" set (union of divs of all previously picked elements).

    # Let's use a recursive approach over chains.
    # state: (chain_idx, current_sum, current_forbidden_mask, current_terms, current_term_count)

    # To keep it tractable, we limit the number of terms and use pruning.

    # Actually, for 128 targets, the max number of terms is small (probably <= 10).
    # Let's enumerate antichains by size, starting from size 1.

    # Better approach: DFS over chains, picking at most one from each chain,
    # tracking forbidden mask, and recording best per sum.

    # The number of chains is small. Let's count:
    # 5-smooth numbers up to 128: 1,2,3,4,5,6,8,9,10,12,15,16,18,20,24,25,27,
    # 30,32,36,40,45,48,50,54,60,64,72,75,80,81,90,96,100,108,120,125,128
    # That's 38 numbers.
    # Chains: numbers related by divisibility.
    # 1: divides everything, so 1 is in a chain with all numbers? No, only if they're
    # connected through a chain of divisibility.
    # Actually, 1|2, 2|4, 4|8, 8|16, 16|32, 32|64, 64|128 -> one chain
    # 1|3, 3|9, 9|27 -> but 3|6, 6|12, etc. So 1 is connected to almost everything.
    # This means there's likely one giant chain containing 1 and most numbers.
    # Hmm, that makes the chain approach less useful.

    # Let me just do a straightforward backtracking with bitmask pruning.

    # Precompute: for each smooth number, its "incompatible" set (numbers it divides or that divide it)
    # incompat[i] = divs[i] | {j : smooth[j] | smooth[i]}
    # But since smooth[i] | smooth[i] is true, and we want j != i:
    # Actually, incompat[i] should be all j such that i|j or j|i, excluding i itself.
    # For the antichain check: when adding element i, we need incompat[i] & forbidden == 0
    # where forbidden contains all previously selected elements.

    # Let's precompute incompat:
    incompat = [0] * n
    for i in range(n):
        m = divs[i]  # j such that smooth[i] | smooth[j]
        # Also need j such that smooth[j] | smooth[i]
        for j in range(n):
            if j != i and smooth[i] % smooth[j] == 0:
                m |= (1 << j)
        incompat[i] = m

    # Backtracking:
    # We maintain:
    # - idx: next smooth number to consider
    # - cur_sum: current sum
    # - forbidden: bitmask of selected elements
    # - terms: list of selected values
    # - count: number of selected elements

    # Pruning:
    # - cur_sum > 128: return
    # - count >= best[cur_sum]: return (if we're trying to improve)
    # - For efficiency, we can also prune if even adding the smallest possible
    #   additional elements can't beat the current best for any achievable sum.

    # Let's just do a full DFS and record the best.

    def dfs(idx, cur_sum, forbidden, terms, count):
        if time.time() - start_time > deadline - 5:
            return

        if cur_sum > 128:
            return

        # Record if this is a valid antichain (non-empty)
        if count > 0 and count < best[cur_sum]:
            best[cur_sum] = count
            best_sol[cur_sum] = list(terms)

        # Try adding more elements
        for i in range(idx, n):
            val = smooth[i]
            new_sum = cur_sum + val
            if new_sum > 128:
                break

            # Check if i is compatible with all selected elements
            if incompat[i] & forbidden:
                continue

            # Also, we need to ensure that future elements won't be incompatible.
            # This is handled by the forbidden mask.

            terms.append(val)
            dfs(i + 1, new_sum, forbidden | (1 << i), terms, count + 1)
            terms.pop()

    dfs(0, 0, 0, [], 0)

    # Build answers
    answers = []
    for t in range(1, 129):
        if best_sol[t] is not None:
            answers.append(best_sol[t])
        else:
            # Fallback: find any valid antichain for t
            # This shouldn't happen, but let's handle it
            found = None
            for k in range(1, 20):
                # Try all combinations of k smooth numbers
                from itertools import combinations
                cands = [x for x in smooth if x <= t]
                for combo in combinations(cands, k):
                    if sum(combo) != t:
                        continue
                    # Check antichain
                    ok = True
                    for a in range(k):
                        for b in range(a + 1, k):
                            if combo[b] % combo[a] == 0 or combo[a] % combo[b] == 0:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        found = list(combo)
                        break
                if found:
                    break
            if found:
                answers.append(found)
            else:
                answers.append([t] if t in smooth else [])

    return answers

# EVOLVE-BLOCK-END
