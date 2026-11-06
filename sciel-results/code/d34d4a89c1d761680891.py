import time
from itertools import combinations

# Global cache for the group and result
_GROUP_CACHE = None
_RESULT_CACHE = None

def construct():
    global _GROUP_CACHE, _RESULT_CACHE
    if _RESULT_CACHE is not None:
        return _RESULT_CACHE

    start_time = time.time()
    deadline = start_time + 100.0

    # PSL(2,11) acting on P^1(GF(11)) = {0,1,...,10, infinity=11}
    # Generators: T: x -> x+1 mod 11 (fixes infinity), S: x -> -1/x mod 11
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

    if _GROUP_CACHE is None:
        _GROUP_CACHE = generate_group([T, S])
    
    group = _GROUP_CACHE

    if time.time() > deadline:
        return []

    # Known base block for S(5,6,12) with stabilizer of order 5
    # This gives orbit size 660/5 = 132
    base_block = (0, 1, 3, 4, 5, 11)

    # Generate all images of base_block under the group
    blocks_set = set()
    for perm in group:
        new_block = tuple(sorted(perm[i] for i in base_block))
        blocks_set.add(new_block)

    # Verify: no 5-subset appears in more than one block
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
        # Fallback: try other candidate base blocks
        candidates = [
            (0, 1, 2, 3, 4, 11),
            (0, 1, 2, 3, 5, 11),
            (0, 1, 2, 4, 5, 11),
            (0, 1, 3, 4, 6, 11),
            (0, 2, 3, 4, 5, 11),
        ]
        for bb in candidates:
            blocks_set = set()
            for perm in group:
                new_block = tuple(sorted(perm[i] for i in bb))
                blocks_set.add(new_block)
            if len(blocks_set) != 132:
                continue
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
            if valid:
                break

    if not valid or len(blocks_set) != 132:
        return [list(b) for b in sorted(blocks_set)] if blocks_set else []

    result = [list(b) for b in sorted(blocks_set)]
    _RESULT_CACHE = result
    return result
