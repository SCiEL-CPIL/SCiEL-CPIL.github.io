"""Construct a large graph with no regular induced subgraph on 6+ vertices."""
import time
import random
import numpy as np
from itertools import combinations


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Try greedy construction for n=21, then n=20
    for target_n in [21, 20]:
        if time.time() > deadline - 15:
            break
        result = _greedy_construct(target_n, deadline)
        if result is not None:
            return result

    # Fallback: try random search for n=20
    if time.time() < deadline - 10:
        result = _random_search(20, deadline)
        if result is not None:
            return result

    # Fallback: return a small valid graph (n=5 is always valid)
    return {'n': 5, 'edges': []}


def _greedy_construct(n, deadline):
    """
    Greedy sequential vertex addition: start empty, add vertices one by one,
    choosing edge connections to existing vertices that minimize new regular
    induced subgraphs of size 6+.
    """
    best_result = None
    best_violations = float('inf')

    # Try multiple random orderings and tie-breaking strategies
    for attempt in range(30):
        if time.time() > deadline - 5:
            break

        rng = random.Random(attempt * 7919 + 13)
        adj_mask = [0] * n

        for new_v in range(1, n):
            # Determine which subsets of existing vertices (size 5) would form
            # a regular induced subgraph with new_v if we connect new_v to them
            # Strategy: for each possible connection pattern, count violations

            # We need to choose which existing vertices to connect to new_v
            # Try a few candidate patterns and pick the best

            best_pattern = None
            best_score = float('inf')

            # Generate candidate patterns
            candidates = []

            # Pattern 1: random with p=0.5
            pat1 = 0
            for v in range(new_v):
                if rng.random() < 0.5:
                    pat1 |= (1 << v)
            candidates.append(pat1)

            # Pattern 2: random with p=0.4
            pat2 = 0
            for v in range(new_v):
                if rng.random() < 0.4:
                    pat2 |= (1 << v)
            candidates.append(pat2)

            # Pattern 3: random with p=0.6
            pat3 = 0
            for v in range(new_v):
                if rng.random() < 0.6:
                    pat3 |= (1 << v)
            candidates.append(pat3)

            # Pattern 4: connect to exactly half (rounded)
            half = new_v // 2
            chosen = rng.sample(range(new_v), half)
            pat4 = 0
            for v in chosen:
                pat4 |= (1 << v)
            candidates.append(pat4)

            # Pattern 5: connect to fewer (new_v//3)
            third = max(1, new_v // 3)
            chosen5 = rng.sample(range(new_v), third)
            pat5 = 0
            for v in chosen5:
                pat5 |= (1 << v)
            candidates.append(pat5)

            # Pattern 6: connect to more (2*new_v//3)
            two_thirds = min(new_v - 1, 2 * new_v // 3)
            chosen6 = rng.sample(range(new_v), two_thirds)
            pat6 = 0
            for v in chosen6:
                pat6 |= (1 << v)
            candidates.append(pat6)

            # Evaluate each candidate
            for pat in candidates:
                # Temporarily add the edges
                old_masks = list(adj_mask)
                for v in range(new_v):
                    if pat & (1 << v):
                        adj_mask[v] |= (1 << new_v)
                        adj_mask[new_v] |= (1 << v)

                # Count how many 6-subsets involving new_v are regular
                violations = _count_new_violations(adj_mask, new_v, n)

                # Restore
                adj_mask = old_masks

                if violations < best_score:
                    best_score = violations
                    best_pattern = pat

            # Apply best pattern
            if best_pattern is not None:
                for v in range(new_v):
                    if best_pattern & (1 << v):
                        adj_mask[v] |= (1 << new_v)
                        adj_mask[new_v] |= (1 << v)

        # Final verification
        if _is_valid_fast(adj_mask, n, deadline):
            edges = _masks_to_edges(adj_mask, n)
            return {'n': n, 'edges': edges}

        # Track best attempt
        total_violations = _total_violations(adj_mask, n, deadline)
        if total_violations < best_violations:
            best_violations = total_violations
            best_result = _masks_to_edges(adj_mask, n)

    return None


def _count_new_violations(adj_mask, new_v, n):
    """Count 6-subsets containing new_v that induce a regular subgraph."""
    count = 0
    other = [v for v in range(n) if v != new_v]

    for subset in combinations(other, 5):
        # Check if {new_v} ∪ subset induces a regular subgraph
        mask = (1 << new_v)
        for v in subset:
            mask |= (1 << v)

        vertices = [new_v] + list(subset)
        first_deg = (adj_mask[vertices[0]] & mask).bit_count()
        is_regular = True
        for v in vertices[1:]:
            if (adj_mask[v] & mask).bit_count() != first_deg:
                is_regular = False
                break

        if is_regular:
            count += 1

    return count


def _total_violations(adj_mask, n, deadline):
    """Count total regular induced subgraphs of size 6+."""
    count = 0
    for size in range(6, n + 1):
        for subset in combinations(range(n), size):
            if time.time() > deadline:
                return count
            mask = 0
            for v in subset:
                mask |= (1 << v)

            first_deg = (adj_mask[subset[0]] & mask).bit_count()
            is_regular = True
            for v in subset[1:]:
                if (adj_mask[v] & mask).bit_count() != first_deg:
                    is_regular = False
                    break

            if is_regular:
                count += 1
    return count


def _is_valid_fast(adj_mask, n, deadline):
    """
    Check if no subset of 6+ vertices induces a regular subgraph.
    Uses bitmask operations for fast degree computation within subsets.
    """
    if n < 6:
        return True

    # Check all subset sizes from 6 to n
    for size in range(6, n + 1):
        for subset in combinations(range(n), size):
            # Time check
            if time.time() > deadline:
                return False

            # Build mask for this subset
            mask = 0
            for v in subset:
                mask |= (1 << v)

            # Compute degrees within the induced subgraph
            first_deg = (adj_mask[subset[0]] & mask).bit_count()
            for v in subset[1:]:
                if (adj_mask[v] & mask).bit_count() != first_deg:
                    break
            else:
                # All degrees equal -> regular induced subgraph found
                return False

    return True


def _masks_to_edges(adj_mask, n):
    """Convert bitmask adjacency to edge list."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_mask[i] & (1 << j):
                edges.append([i, j])
    return edges


def _random_search(n, deadline):
    """Fallback random search for valid graph."""
    p_values = [0.5, 0.45, 0.55, 0.4, 0.6, 0.35, 0.65, 0.3, 0.7, 0.52, 0.48]
    max_trials = 100

    for trial in range(max_trials):
        if time.time() > deadline - 3:
            break

        p = p_values[trial % len(p_values)]
        seed = trial * 2654435761 % (2**32)
        rng = random.Random(seed)

        adj_mask = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)

        if _is_valid_fast(adj_mask, n, deadline):
            edges = _masks_to_edges(adj_mask, n)
            return {'n': n, 'edges': edges}

    return None
