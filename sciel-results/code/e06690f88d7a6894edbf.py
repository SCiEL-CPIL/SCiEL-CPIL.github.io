"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def _neighbors(cw, q, n, radius):
    """Yield (word_int, cw_int) for words within radius of cw. Returns list of ints."""
    out = []
    # all words within radius r of cw
    # enumerate all positions to change up to radius
    for r in range(radius + 1):
        for pos in _combinations_with_replacement_range(n, r):
            if len(set(pos)) != r and r > 0:
                continue
            # actually we want distinct positions; use combinations
            break
    # simpler: generate via itertools
    from itertools import combinations, product
    seen = set()
    for r in range(radius + 1):
        for pos in combinations(range(n), r):
            for vals in product(range(q), repeat=r):
                w = list(cw)
                for i, v in zip(pos, vals):
                    w[i] = v
                wi = _to_int(w, q)
                if wi not in seen:
                    seen.add(wi)
                    out.append(wi)
    return out


def _combinations_with_replacement_range(n, r):
    from itertools import combinations
    return list(combinations(range(n), r))


def _to_int(w, q):
    v = 0
    for x in w:
        v = v * q + x
    return v


def _to_word(i, q, n):
    w = [0] * n
    for j in range(n - 1, -1, -1):
        w[j] = i % q
        i //= q
    return w


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill
    total_space = q ** n

    # Precompute neighbor lists for all words? Too big for n=8,q=3 (6561 words, each ~1+8*2+28*4=... ~ 1+16+112=129 neighbors)
    # That's 6561*129 ~ 846k entries, feasible.
    # Build adjacency: for each word, list of words within radius
    all_words = list(range(total_space))
    # Build neighbor sets
    neighbor_map = {}
    for wi in all_words:
        w = _to_word(wi, q, n)
        nbrs = set()
        from itertools import combinations, product
        for r in range(radius + 1):
            for pos in combinations(range(n), r):
                for vals in product(range(q), repeat=r):
                    nw = list(w)
                    for i, v in zip(pos, vals):
                        nw[i] = v
                    ni = _to_int(nw, q)
                    nbrs.add(ni)
        neighbor_map[wi] = nbrs

    # Greedy: pick codeword covering most uncovered words
    uncovered = set(all_words)
    codewords = []
    # For greedy, we need for each candidate cw, count of uncovered words it covers
    # = |neighbor_map[cw] ∩ uncovered|
    # We'll maintain a count array
    # counts[w] = number of codewords that cover w
    counts = np.zeros(total_space, dtype=np.int32)
    # Initially all uncovered
    # Greedy step: find cw maximizing uncovered covered
    # To be efficient, maintain for each cw the number of uncovered words in its neighborhood
    # uncovered_count[cw] = |neighbor_map[cw] ∩ uncovered|
    uncovered_count = np.zeros(total_space, dtype=np.int32)
    for cw in all_words:
        uncovered_count[cw] = len(neighbor_map[cw])

    # We'll use a simple approach: at each step, pick cw with max uncovered_count[cw]
    # Then update: for each word w in neighbor_map[cw], if it was uncovered, decrement
    # uncovered_count for all cw' that cover w
    # This is expensive. Instead, we'll recompute or use a different strategy.

    # Alternative: since total_space is only 6561, we can afford O(total_space^2) per step
    # but that's 43M per step, and we need ~486 steps... too slow.

    # Better: use the fact that neighbor_map is small (~129 entries)
    # For each step:
    #   best_cw = argmax uncovered_count[cw]
    #   for each w in neighbor_map[best_cw]:
    #       if w was uncovered (counts[w]==0):
    #           for each cw2 in codewords_covering[w]:
    #               uncovered_count[cw2] -= 1
    #   We need reverse mapping: for each word w, which codewords cover it = neighbor_map is symmetric
    #   Actually neighbor_map[w] = set of codewords within radius of w = same as words within radius of w
    #   So codewords covering w = neighbor_map[w]

    # Precompute reverse: for each w, list of cw such that w in neighbor_map[cw]
    # Since neighbor_map is symmetric (distance is symmetric), neighbor_map[w] = {cw : w in neighbor_map[cw]}
    # So we can just use neighbor_map[w] as the list of codewords that cover w.

    while uncovered and time.time() < deadline - 5.0:
        # Find best codeword
        best_cw = int(np.argmax(uncovered_count))
        if uncovered_count[best_cw] == 0:
            break
        codewords.append(best_cw)
        # Update counts and uncovered_count
        for w in neighbor_map[best_cw]:
            if counts[w] == 0:
                # w was uncovered, now covered
                counts[w] += 1
                # For each cw2 that covers w, decrement uncovered_count[cw2]
                for cw2 in neighbor_map[w]:
                    uncovered_count[cw2] -= 1
        # Remove from uncovered
        # We don't actually need the uncovered set if we track counts
        # But we need to know when all are covered
        # Let's track covered_count
        # Actually, let's just check if min(counts) > 0 at the end

    # Local search: try to remove codewords or swap
    # 1-exchange: try removing a codeword and see if coverage is maintained
    # Or try swapping one codeword for another to reduce total count
    # Start with removal: can we remove any codeword?
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        # Recompute counts from scratch for simplicity
        counts2 = np.zeros(total_space, dtype=np.int32)
        for cw in codewords:
            for w in neighbor_map[cw]:
                counts2[w] += 1
        # Check if all covered
        if np.min(counts2) == 0:
            break
        # Try removing each codeword
        for i in range(len(codewords) - 1, -1, -1):
            cw = codewords[i]
            # Check if removing cw leaves all words covered
            # A word w is still covered if counts2[w] > 1 or w is covered by another cw
            # Simpler: for each w in neighbor_map[cw], check if counts2[w] > 1
            removable = True
            for w in neighbor_map[cw]:
                if counts2[w] <= 1:
                    removable = False
                    break
            if removable:
                codewords.pop(i)
                changed = True
                break

    # Convert back to lists
    result = [_to_word(cw, q, n) for cw in codewords]
    return result
# EVOLVE-BLOCK-END
