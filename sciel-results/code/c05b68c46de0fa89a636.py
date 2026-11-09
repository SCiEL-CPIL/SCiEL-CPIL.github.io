"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations, product


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


def _build_neighbor_map(q, n, radius):
    """Build a list of neighbor lists for each word in the space."""
    total_space = q ** n
    # Precompute all possible position combinations and value combinations for each radius
    # For radius r, we need all subsets of positions of size r, and all value assignments
    # of size r (each position can be changed to any of q-1 other values, or kept same)
    # Actually, for a word within Hamming distance r, we change up to r positions
    # to different values.
    
    # Precompute all (positions, values) pairs for each radius
    pos_val_pairs = {}
    for r in range(radius + 1):
        pairs = []
        if r == 0:
            pairs.append(((), ()))
        else:
            for pos in combinations(range(n), r):
                # For each position, we can change to any of the q-1 other values
                # But we also want to include the "same" value? No, Hamming distance
                # counts positions where values differ. So for each position, we
                # change to a different value.
                # Actually, to be safe, let's include all q values (including same)
                # but that would overcount. Let's use q-1 different values.
                # Wait, if we change to the same value, the Hamming distance doesn't increase.
                # So for r positions, we need to change each to a value different from original.
                # But the "original" depends on the word. So we can't precompute values
                # without knowing the word.
                # 
                # Alternative: precompute just the positions, and generate values on the fly.
                # Or: precompute all (pos, vals) where vals are relative changes.
                # For each position, the change is in {1, 2, ..., q-1} mod q (i.e., add 1 to q-1 mod q)
                # But that's not quite right either, because the target value depends on original.
                #
                # Let's just precompute positions for each radius, and generate neighbors on the fly.
                for vals in product(range(1, q), repeat=r):
                    # vals are the "additions" mod q to get the new value
                    pairs.append((pos, vals))
        pos_val_pairs[r] = pairs
    
    neighbor_map = [None] * total_space
    for wi in range(total_space):
        w = _to_word(wi, q, n)
        nbrs = set()
        for r in range(radius + 1):
            for pos, vals in pos_val_pairs[r]:
                nw = list(w)
                if r > 0:
                    for i, v in zip(pos, vals):
                        nw[i] = (w[i] + v) % q
                ni = _to_int(nw, q)
                nbrs.add(ni)
        neighbor_map[wi] = list(nbrs)
    return neighbor_map


def _greedy_cover(neighbor_map, total_space, deadline, seed=42):
    """Greedy set cover with efficient updates."""
    rng = random.Random(seed)
    # counts[w] = number of selected codewords that cover w
    counts = np.zeros(total_space, dtype=np.int32)
    # uncovered_count[cw] = number of uncovered words in neighbor_map[cw]
    uncovered_count = np.zeros(total_space, dtype=np.int32)
    for cw in range(total_space):
        uncovered_count[cw] = len(neighbor_map[cw])
    
    codewords = []
    
    # We need to efficiently find the best codeword.
    # We'll maintain a max-heap or just scan (total_space is only 6561)
    # Since total_space is small, scanning is fine.
    
    while True:
        if time.time() >= deadline - 5.0:
            break
        
        # Find codeword with max uncovered_count
        best_cw = int(np.argmax(uncovered_count))
        best_count = uncovered_count[best_cw]
        if best_count == 0:
            break
        
        codewords.append(best_cw)
        
        # Update: for each word w in neighbor_map[best_cw], if it was uncovered:
        #   increment counts[w]
        #   for each cw2 in neighbor_map[w], decrement uncovered_count[cw2]
        for w in neighbor_map[best_cw]:
            if counts[w] == 0:
                counts[w] += 1
                for cw2 in neighbor_map[w]:
                    uncovered_count[cw2] -= 1
    
    return codewords


def _local_search_reduce(codewords, neighbor_map, total_space, deadline):
    """Try to remove redundant codewords."""
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        # Recompute coverage counts
        counts = np.zeros(total_space, dtype=np.int32)
        for cw in codewords:
            for w in neighbor_map[cw]:
                counts[w] += 1
        
        # Try removing codewords from the end
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() >= deadline - 2.0:
                break
            cw = codewords[i]
            removable = True
            for w in neighbor_map[cw]:
                if counts[w] <= 1:
                    removable = False
                    break
            if removable:
                codewords.pop(i)
                changed = True
                break
    return codewords


def _local_search_swap(codewords, neighbor_map, total_space, deadline, max_swaps=50):
    """Try to swap a codeword with a better one to reduce count or improve coverage."""
    # This is more complex; for now, just try 1-exchange (remove one, add another)
    # Actually, let's try: remove codeword i, then greedily add back if needed
    # This is expensive, so let's just do a simple version
    pass


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill
    total_space = q ** n
    
    # Build neighbor map
    neighbor_map = _build_neighbor_map(q, n, radius)
    
    # Phase 1: Greedy cover
    codewords = _greedy_cover(neighbor_map, total_space, deadline, seed=42)
    
    # Phase 2: Local search - remove redundant codewords
    codewords = _local_search_reduce(codewords, neighbor_map, total_space, deadline)
    
    # Phase 3: Try a few more greedy runs with different seeds and take the best
    best_codewords = codewords
    best_len = len(codewords)
    
    for seed in [1, 2, 3, 7, 13, 42, 100, 999]:
        if time.time() >= deadline - 20.0:
            break
        # Skip seed 42 since we already did it
        if seed == 42:
            continue
        cw2 = _greedy_cover(neighbor_map, total_space, deadline, seed=seed)
        cw2 = _local_search_reduce(cw2, neighbor_map, total_space, deadline)
        if len(cw2) < best_len:
            best_len = len(cw2)
            best_codewords = cw2
    
    # Final reduction pass
    best_codewords = _local_search_reduce(best_codewords, neighbor_map, total_space, deadline)
    
    # Convert back to lists
    result = [_to_word(cw, q, n) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
