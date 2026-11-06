import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    # PSL(2,11) acting on P^1(GF(11))
    T = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 11]
    S = [11, 10, 5, 7, 8, 2, 9, 3, 4, 6, 1, 0]

    def compose(p, q):
        return [p[q[i]] for i in range(12)]

    def generate_group(gens):
        e = tuple(range(12))
        group = {e}
        frontier = [e]
        while frontier:
            new_frontier = []
            for g in frontier:
                for h in gens:
                    gh = tuple(compose(g, h))
                    if gh not in group:
                        group.add(gh)
                        new_frontier.append(gh)
                    hg = tuple(compose(h, g))
                    if hg not in group:
                        group.add(hg)
                        new_frontier.append(hg)
            frontier = new_frontier
        return group

    group = generate_group([T, S])

    if time.time() > deadline:
        return []

    base_block = (0, 1, 3, 4, 5, 11)

    blocks_set = set()
    for perm in group:
        new_block = tuple(sorted(perm[i] for i in base_block))
        blocks_set.add(new_block)

    # Verify
    five_subsets = set()
    valid = True
    for block in blocks_set:
        for five in combinations(block, 5):
            if five in five_subsets:
                valid = False
                break
            five_subsets.add(five)
        if not valid:
            break

    if not valid or len(blocks_set) != 132:
        return [list(b) for b in sorted(blocks_set)] if blocks_set else []

    return [list(b) for b in sorted(blocks_set)]
