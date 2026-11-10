import random
import time
from collections import Counter

# EVOLVE-BLOCK-START

def _precompute_triples(N=180):
    """
    Precompute all distinct triples (c, a, b) with c < a < b
    such that 1/c = 1/a + 1/b and a, b, c <= N.
    Returns a list of tuples (c, a, b).
    """
    triples = []
    # c is the smallest element
    for c in range(1, N + 1):
        # (a-c)(b-c) = c^2
        # Let u = a-c, v = b-c. u*v = c^2.
        # a = c+u, b = c+v.
        # We need a, b <= N.
        # So c+u <= N and c+v <= N.
        # Also u, v >= 1.
        # u <= v WLOG (so a <= b).
        # u <= c.
        c_sq = c * c
        # Iterate u from 1 to c
        for u in range(1, c + 1):
            if c_sq % u != 0:
                continue
            v = c_sq // u
            a = c + u
            b = c + v
            if a <= N and b <= N:
                # Ensure distinct
                if a != b and a != c and b != c:
                    triples.append((c, a, b))
    return triples


def _is_valid_subset(S, triples):
    """Return True if S contains no complete precomputed triple."""
    S = set(S)
    for t in triples:
        if t[0] in S and t[1] in S and t[2] in S:
            return False
    return True


def _greedy_vertex_cover(N, triples, deadline, deterministic=True):
    """
    Fast Greedy Vertex Cover (Hitting Set).
    Repeatedly picks the vertex involved in the most uncovered triples.
    """
    if not triples:
        return set(), set(range(1, N + 1))

    elem_to_triples = [[] for _ in range(N + 1)]
    for tri_idx, t in enumerate(triples):
        for x in t:
            elem_to_triples[x].append(tri_idx)

    active_triples = [True] * len(triples)
    degrees = [0] * (N + 1)
    for tri_idx, t in enumerate(triples):
        for x in t:
            degrees[x] += 1

    removed = set()
    active_count = len(triples)

    while active_count > 0:
        if time.time() >= deadline:
            break

        # Find vertex with max degree among those not removed
        best_elem = -1
        best_deg = -1
        for x in range(1, N + 1):
            if x not in removed and degrees[x] > best_deg:
                best_deg = degrees[x]
                best_elem = x

        if best_elem == -1 or best_deg <= 0:
            break

        removed.add(best_elem)

        # Update degrees and active status
        for tri_idx in elem_to_triples[best_elem]:
            if active_triples[tri_idx]:
                active_triples[tri_idx] = False
                active_count -= 1
                # Decrease degree of other vertices in this triple
                for z in triples[tri_idx]:
                    if z != best_elem:
                        degrees[z] -= 1

    current_set = set(range(1, N + 1)) - removed
    return removed, current_set


def _local_search_add_back(N, triples, removed, current_set, deadline):
    """
    Repeatedly add back removed elements if doing so does not complete any triple.
    Uses per-triple active counts for O(1) feasibility checks.
    """
    # active_count[t] = number of elements of triple t currently in current_set.
    active_count = [0] * len(triples)
    for tri_idx, t in enumerate(triples):
        cnt = 0
        for x in t:
            if x in current_set:
                cnt += 1
        active_count[tri_idx] = cnt

    # For each element, list of incident triple indices.
    elem_to_triples = [[] for _ in range(N + 1)]
    for tri_idx, t in enumerate(triples):
        for x in t:
            elem_to_triples[x].append(tri_idx)

    changed = True
    while changed:
        if time.time() >= deadline:
            break
        changed = False

        # Deterministic order by element value gives stable behavior.
        for x in sorted(removed):
            # Adding x is safe iff no incident triple already has the other two elements.
            safe = True
            for tri_idx in elem_to_triples[x]:
                if active_count[tri_idx] >= 2:
                    safe = False
                    break
            if safe:
                current_set.add(x)
                removed.discard(x)
                for tri_idx in elem_to_triples[x]:
                    active_count[tri_idx] += 1
                changed = True

    return removed, current_set


def _local_search_swaps(N, triples, removed, current_set, deadline):
    """
    Try size-preserving swaps that may enable subsequent add-back improvements.
    This is a local refinement within the same design family.
    """
    elem_to_triples = [[] for _ in range(N + 1)]
    for tri_idx, t in enumerate(triples):
        for x in t:
            elem_to_triples[x].append(tri_idx)

    # active_count for current_set.
    active_count = [0] * len(triples)
    for tri_idx, t in enumerate(triples):
        cnt = 0
        for x in t:
            if x in current_set:
                cnt += 1
        active_count[tri_idx] = cnt

    def can_add_with_counts(x, counts):
        for tri_idx in elem_to_triples[x]:
            if counts[tri_idx] >= 2:
                return False
        return True

    max_swap_passes = 5
    passes = 0
    improved = True
    while improved and passes < max_swap_passes:
        if time.time() >= deadline:
            break
        improved = False
        passes += 1

        # Try each removed x.
        for x in sorted(removed):
            if time.time() >= deadline:
                break

            # Find blocking triples for x: those with count >= 2.
            blockers = []
            for tri_idx in elem_to_triples[x]:
                if active_count[tri_idx] >= 2:
                    blockers.append(tri_idx)

            if not blockers:
                # Should have been added already; do it just in case.
                current_set.add(x)
                removed.discard(x)
                for tri_idx in elem_to_triples[x]:
                    active_count[tri_idx] += 1
                improved = True
                continue

            # Candidate y to remove from current_set: one that appears in all blockers.
            # Removing y may unblock x.
            common = None
            for tri_idx in blockers:
                t = triples[tri_idx]
                # y must be in current_set and not x.
                cand = set()
                for z in t:
                    if z != x and z in current_set:
                        cand.add(z)
                if common is None:
                    common = cand
                else:
                    common &= cand
                if not common:
                    break

            if not common:
                continue

            # Try each candidate y in common.
            for y in sorted(common):
                if time.time() >= deadline:
                    break

                # Simulate removing y and adding x.
                # First remove y from counts.
                for tri_idx in elem_to_triples[y]:
                    active_count[tri_idx] -= 1

                # Check if x is now addable.
                if can_add_with_counts(x, active_count):
                    # Also check that removing y does not immediately create a new complete triple.
                    # Since y was in current_set, removing it cannot create a triple.
                    # Add x.
                    current_set.add(x)
                    removed.discard(x)
                    current_set.discard(y)
                    removed.add(y)
                    for tri_idx in elem_to_triples[x]:
                        active_count[tri_idx] += 1
                    improved = True
                    break
                else:
                    # Undo removal of y.
                    for tri_idx in elem_to_triples[y]:
                        active_count[tri_idx] += 1

    return removed, current_set


def construct():
    start = time.time()
    # Explicit deadline guard with safety margin before evaluator kill.
    deadline = start + 45.0
    N = 180

    triples = _precompute_triples(N)

    # Initial greedy vertex cover.
    removed, current_set = _greedy_vertex_cover(N, triples, deadline)

    # Local add-back passes.
    removed, current_set = _local_search_add_back(N, triples, removed, current_set, deadline)

    # Size-preserving swaps to potentially unlock more add-backs.
    removed, current_set = _local_search_swaps(N, triples, removed, current_set, deadline)

    # Final add-back after swaps.
    removed, current_set = _local_search_add_back(N, triples, removed, current_set, deadline)

    # Safety: verify validity. If somehow invalid, fall back to a conservative set.
    if not _is_valid_subset(current_set, triples):
        # Fallback: take all numbers > 90. This avoids many small c triples,
        # but we still verify; if invalid, use a very safe subset.
        fallback = set(range(91, N + 1))
        if not _is_valid_subset(fallback, triples):
            # Extremely conservative: primes > 60 (no small divisors)
            fallback = set()
            for x in range(61, N + 1):
                is_prime = True
                for d in range(2, int(x ** 0.5) + 1):
                    if x % d == 0:
                        is_prime = False
                        break
                if is_prime:
                    fallback.add(x)
        current_set = fallback

    # Ensure nonempty.
    if not current_set:
        current_set = {180}

    return list(current_set)

# EVOLVE-BLOCK-END
