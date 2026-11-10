"""Construct a linear 3-uniform hypergraph on 16 points with small independence number."""
import time
import random
from itertools import combinations

# Deadline guard: 150s wall-clock limit, use 140s to be safe
_DEADLINE = 140.0
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _remaining_time():
    return max(0.0, _DEADLINE - _elapsed())


def _is_linear(triples):
    """Check that any two triples share at most one point."""
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True


def _independence_number(triples):
    """Compute the independence number: largest set of points containing no triple."""
    n = 16
    # Bitmask representation for speed
    triple_masks = []
    for t in triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks.append(m)

    ntriples = len(triple_masks)

    # Check all subsets by size, from largest to smallest
    # For n=16, we can iterate all 2^16 = 65536 subsets
    # A subset S is independent if no triple is fully contained in S
    best = 0
    for mask in range(1 << n):
        # Check if any triple is contained in mask
        independent = True
        for tm in triple_masks:
            if (tm & mask) == tm:
                independent = False
                break
        if independent:
            sz = bin(mask).count('1')
            if sz > best:
                best = sz
    return best


def _independence_number_fast(triples):
    """Faster independence number computation using subset enumeration with pruning."""
    n = 16
    triple_masks = []
    for t in triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks.append(m)

    ntriples = len(triple_masks)
    if ntriples == 0:
        return n

    best = 0
    # Enumerate all subsets
    for mask in range(1 << n):
        sz = bin(mask).count('1')
        if sz <= best:
            continue
        independent = True
        for tm in triple_masks:
            if (tm & mask) == tm:
                independent = False
                break
        if independent:
            best = sz
    return best


def _greedy_construct(rng, target_alpha=5):
    """Greedy construction trying to achieve a given independence number."""
    n = 16
    triples = []
    used_pairs = set()

    # Precompute all possible triples
    all_triples = list(combinations(range(n), 3))
    rng.shuffle(all_triples)

    # Build a conflict graph: which pairs are used
    for tri in all_triples:
        if _elapsed() > _DEADLINE - 5:
            break

        # Check if this triple conflicts with existing ones
        ok = True
        pairs_in_tri = set()
        for i in range(3):
            for j in range(i + 1, 3):
                p = (tri[i], tri[j])
                if p < (tri[j], tri[i]):
                    p = (tri[j], tri[i])
                if p in used_pairs:
                    ok = False
                    break
                pairs_in_tri.add(p)
            if not ok:
                break

        if ok:
            triples.append(list(tri))
            used_pairs.update(pairs_in_tri)

    return triples


def _local_search(triples, rng, max_iters=500):
    """Local search: try to improve the system by swapping triples."""
    n = 16
    current_alpha = _independence_number_fast(triples)
    best_triples = triples[:]
    best_alpha = current_alpha

    # Build pair usage
    def get_pairs(tri):
        return frozenset((min(a,b), max(a,b)) for a, b in combinations(tri, 2))

    used_pairs = set()
    for t in triples:
        used_pairs.update(get_pairs(t))

    all_triples = list(combinations(range(n), 3))

    for it in range(max_iters):
        if _remaining_time() < 2:
            break
        if it % 20 == 0 and _remaining_time() < 10:
            break

        improved = False
        # Try removing one triple and adding a different one
        for idx in range(len(triples)):
            if _remaining_time() < 2:
                break
            orig = triples[idx]
            orig_pairs = get_pairs(orig)
            used_pairs -= orig_pairs

            # Try candidate replacements
            candidates = [t for t in all_triples if set(t) != set(orig)]
            rng.shuffle(candidates)
            for cand in candidates[:200]:
                cand_pairs = get_pairs(cand)
                if cand_pairs & used_pairs:
                    continue
                triples[idx] = list(cand)
                used_pairs |= cand_pairs

                alpha = _independence_number_fast(triples)
                if alpha < best_alpha:
                    best_alpha = alpha
                    best_triples = triples[:]
                    improved = True
                    break
                else:
                    triples[idx] = orig
                    used_pairs -= cand_pairs
                    used_pairs |= orig_pairs

            # Restore
            if not improved:
                triples[idx] = orig
                used_pairs |= orig_pairs

        if not improved:
            break

    return best_triples, best_alpha


def _simulated_annealing(triples, rng, max_iters=2000):
    """Simulated annealing to improve the system."""
    n = 16
    current_alpha = _independence_number_fast(triples)
    best_triples = triples[:]
    best_alpha = current_alpha

    def get_pairs(tri):
        return frozenset((min(a,b), max(a,b)) for a, b in combinations(tri, 2))

    used_pairs = set()
    for t in triples:
        used_pairs.update(get_pairs(t))

    all_triples = list(combinations(range(n), 3))

    temp = 1.0
    cooling = 0.999

    for it in range(max_iters):
        if _remaining_time() < 3:
            break

        temp *= cooling
        if temp < 0.01:
            temp = 0.01

        # Pick a random triple to modify
        idx = rng.randint(0, len(triples) - 1)
        orig = triples[idx]
        orig_pairs = get_pairs(orig)
        used_pairs -= orig_pairs

        # Pick a random replacement
        cand = all_triples[rng.randint(0, len(all_triples) - 1)]
        cand_pairs = get_pairs(cand)
        if cand_pairs & used_pairs:
            continue

        triples[idx] = list(cand)
        used_pairs |= cand_pairs

        new_alpha = _independence_number_fast(triples)
        delta = new_alpha - current_alpha

        if delta < 0 or (delta == 0 and rng.random() < 0.1) or (rng.random() < (2.718281828 ** (-delta / max(temp, 1e-6)))):
            current_alpha = new_alpha
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_triples = triples[:]
        else:
            triples[idx] = orig
            used_pairs -= cand_pairs
            used_pairs |= orig_pairs

    return best_triples, best_alpha


def construct():
    global _START_TIME
    _START_TIME = time.time()
    rng = random.Random(42)

    best_triples = None
    best_alpha = 16  # worst case

    # Start with a greedy construction
    triples = _greedy_construct(rng)
    if triples:
        alpha = _independence_number_fast(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]

    # Local search from greedy result
    if _remaining_time() > 5 and best_triples:
        ls_triples, ls_alpha = _local_search(best_triples[:], rng, max_iters=300)
        if ls_alpha < best_alpha:
            best_alpha = ls_alpha
            best_triples = ls_triples

    # Simulated annealing
    if _remaining_time() > 10 and best_triples:
        sa_triples, sa_alpha = _simulated_annealing(best_triples[:], rng, max_iters=3000)
        if sa_alpha < best_alpha:
            best_alpha = sa_alpha
            best_triples = sa_triples

    # Multiple restarts with different random seeds
    restarts = 0
    while _remaining_time() > 20 and restarts < 50:
        restarts += 1
        triples = _greedy_construct(rng)
        if not triples:
            continue
        alpha = _independence_number_fast(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]

        if _remaining_time() > 15:
            ls_triples, ls_alpha = _local_search(best_triples[:], rng, max_iters=200)
            if ls_alpha < best_alpha:
                best_alpha = ls_alpha
                best_triples = ls_triples

    # Final local search pass if time permits
    if _remaining_time() > 5 and best_triples:
        final_triples, final_alpha = _local_search(best_triples[:], rng, max_iters=400)
        if final_alpha < best_alpha:
            best_alpha = final_alpha
            best_triples = final_triples

    if best_triples is None:
        # Fallback: return a simple valid system
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

    return best_triples
