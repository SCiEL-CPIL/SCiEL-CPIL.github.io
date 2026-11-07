import time
import random

_start_time = time.time()

def construct():
    deadline = _start_time + 85
    n = 12
    m = 40
    total_masks = 1 << n  # 4096

    # Precompute comparability matrix: comp[a][b] = True if a and b are comparable
    # Using a 2D list of bytes for memory efficiency
    # comp[a][b] = 1 if a|b == a or a|b == b
    comp = [bytearray(total_masks) for _ in range(total_masks)]
    for a in range(total_masks):
        for b in range(a + 1, total_masks):
            if (a | b) == a or (a | b) == b:
                comp[a][b] = 1
                comp[b][a] = 1

    def is_comparable(a, b):
        return comp[a][b]

    def count_comparables(family):
        cnt = 0
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                if comp[family[i]][family[j]]:
                    cnt += 1
        return cnt

    def get_degrees(family):
        degrees = [0] * len(family)
        for i in range(len(family)):
            fi = family[i]
            for j in range(i + 1, len(family)):
                if comp[fi][family[j]]:
                    degrees[i] += 1
                    degrees[j] += 1
        return degrees

    # Generate all masks by size
    masks_by_size = [[] for _ in range(n + 1)]
    for mask in range(total_masks):
        masks_by_size[bin(mask).count('1')].append(mask)

    # Greedy chain-based initialization
    # Strategy: Build multiple chains that maximize cross-chain comparability
    # Start with a long chain, then add sets that are comparable to many existing sets

    family = []
    seen = set()

    # Build a primary chain: 0, {0}, {0,1}, {0,1,2}, ..., {0,...,11}
    # That's 13 sets
    for sz in range(n + 1):
        mask = (1 << sz) - 1  # {0, 1, ..., sz-1}
        family.append(mask)
        seen.add(mask)

    # Now we have 13 sets. Need 27 more.
    # Greedily add sets that maximize comparability with existing family
    # Use a set for O(1) removal instead of list
    remaining_set = set(mask for mask in range(total_masks) if mask not in seen)

    for _ in range(m - len(family)):
        best_mask = -1
        best_deg = -1
        for c in remaining_set:
            deg = 0
            for s in family:
                if comp[c][s]:
                    deg += 1
            if deg > best_deg:
                best_deg = deg
                best_mask = c
        if best_mask == -1:
            break
        family.append(best_mask)
        seen.add(best_mask)
        remaining_set.discard(best_mask)

    # Ensure exactly 40
    while len(family) < m:
        for mask in range(total_masks):
            if mask not in seen:
                family.append(mask)
                seen.add(mask)
                break
    if len(family) > m:
        family = family[:m]
        seen = set(family)

    # Local Search: exhaustive hill climbing over all 4096 candidates per slot
    # with multi-restart to escape local optima
    current_score = count_comparables(family)
    best_family = list(family)
    best_score = current_score

    def local_search(fam, seen_set, score, time_limit):
        """Exhaustive hill climbing. Returns (family, seen, score)."""
        improved = True
        while improved:
            if time.time() > time_limit:
                break
            improved = False
            degrees = get_degrees(fam)

            for i in range(len(fam)):
                if time.time() > time_limit:
                    break
                old_set = fam[i]
                old_deg = degrees[i]

                best_new = None
                best_gain = 0

                for new_set in range(total_masks):
                    if new_set == old_set or new_set in seen_set:
                        continue

                    new_deg = 0
                    for j in range(len(fam)):
                        if j == i:
                            continue
                        if comp[new_set][fam[j]]:
                            new_deg += 1

                    gain = new_deg - old_deg
                    if gain > best_gain:
                        best_gain = gain
                        best_new = new_set

                if best_new is not None:
                    fam[i] = best_new
                    seen_set.discard(old_set)
                    seen_set.add(best_new)
                    score += best_gain
                    improved = True
                    # Continue scanning remaining slots in this pass
                    # (don't break; the degrees array is stale but we'll recompute next while iteration)
                    break

        return fam, seen_set, score

    # Primary local search
    time_limit = deadline - 15
    family, seen, current_score = local_search(family, seen, current_score, time_limit)

    if current_score > best_score:
        best_score = current_score
        best_family = list(family)

    # Multi-restart: try different initializations to escape local optima
    if time.time() < deadline - 20:
        # Restart 1: different chain direction
        fam2 = []
        seen2 = set()
        for sz in range(n + 1):
            mask = ((1 << n) - 1) ^ ((1 << (n - sz)) - 1)  # complement chain
            fam2.append(mask)
            seen2.add(mask)
        remaining_set2 = set(mask for mask in range(total_masks) if mask not in seen2)
        for _ in range(m - len(fam2)):
            best_mask = -1
            best_deg = -1
            for c in remaining_set2:
                deg = 0
                for s in fam2:
                    if comp[c][s]:
                        deg += 1
                if deg > best_deg:
                    best_deg = deg
                    best_mask = c
            if best_mask == -1:
                break
            fam2.append(best_mask)
            seen2.add(best_mask)
            remaining_set2.discard(best_mask)

        score2 = count_comparables(fam2)
        fam2, seen2, score2 = local_search(fam2, seen2, score2, deadline - 10)
        if score2 > best_score:
            best_score = score2
            best_family = list(fam2)

    # Restart 2: random perturbation of best solution
    if time.time() < deadline - 10:
        fam3 = list(best_family)
        seen3 = set(fam3)
        # Perturb 3 random positions
        rng = random.Random(42)
        positions = rng.sample(range(m), 3)
        for pos in positions:
            old = fam3[pos]
            candidates = [msk for msk in range(total_masks) if msk not in seen3]
            new = rng.choice(candidates)
            fam3[pos] = new
            seen3.discard(old)
            seen3.add(new)

        score3 = count_comparables(fam3)
        fam3, seen3, score3 = local_search(fam3, seen3, score3, deadline - 5)
        if score3 > best_score:
            best_score = score3
            best_family = list(fam3)

    # Final exhaustive polish: multiple passes over all slots
    if time.time() < deadline - 5:
        polish_family = list(best_family)
        polish_seen = set(polish_family)
        polish_score = best_score

        for _ in range(20):
            if time.time() > deadline - 3:
                break

            any_improvement = False
            degrees = get_degrees(polish_family)

            for i in range(len(polish_family)):
                if time.time() > deadline - 3:
                    break
                old_set = polish_family[i]
                old_deg = degrees[i]

                best_new = None
                best_gain = 0

                for new_set in range(total_masks):
                    if new_set == old_set or new_set in polish_seen:
                        continue

                    new_deg = 0
                    for j in range(len(polish_family)):
                        if j == i:
                            continue
                        if comp[new_set][polish_family[j]]:
                            new_deg += 1

                    gain = new_deg - old_deg
                    if gain > best_gain:
                        best_gain = gain
                        best_new = new_set

                if best_new is not None and best_gain > 0:
                    polish_family[i] = best_new
                    polish_seen.discard(old_set)
                    polish_seen.add(best_new)
                    polish_score += best_gain
                    any_improvement = True

                    if polish_score > best_score:
                        best_score = polish_score
                        best_family = list(polish_family)

            if not any_improvement:
                break

    # Convert to lists of ints
    result = []
    for mask in best_family:
        lst = [int(i) for i in range(n) if (mask >> i) & 1]
        result.append(lst)

    return result

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    assert len(res) == 40
    seen = set()
    for r in res:
        fr = frozenset(r)
        assert fr not in seen
        seen.add(fr)
        for v in r:
            assert isinstance(v, int)
            assert 0 <= v <= 11
    masks = []
    for r in res:
        m = 0
        for v in r:
            m |= (1 << v)
        masks.append(m)
    cnt = 0
    for i in range(40):
        for j in range(i+1, 40):
            a, b = masks[i], masks[j]
            if a | b == a or a | b == b:
                cnt += 1
    print(f"Time: {t1-t0:.2f}s, Comparables: {cnt}")
