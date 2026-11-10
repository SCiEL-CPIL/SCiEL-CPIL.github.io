"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations

def _find_overlap_suffix_prefix(new_tail, block_prefix):
    """Find the longest suffix of new_tail that equals a prefix of block_prefix."""
    max_ov = min(len(new_tail), len(block_prefix))
    for o in range(max_ov, 0, -1):
        if new_tail[-o:] == block_prefix[:o]:
            return o
    return 0

def construct(n):
    start_time = time.time()
    deadline = start_time + 110

    word = "1"
    for m in range(2, n + 1):
        if time.time() > deadline:
            break

        # Extract all distinct (m-1)-permutations from word
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = word[i:i + m - 1]
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)

        if m < n:
            new = ""
            for p in perms:
                block = p + str(m) + p
                # Limit overlap check to last len(block) chars of new
                tail = new[-len(block):] if len(new) >= len(block) else new
                ov = _find_overlap_suffix_prefix(tail, block)
                new += block[ov:]
            word = new
        else:
            candidates = [perms, list(reversed(perms)), sorted(perms)]
            best_word, best_len = None, float('inf')
            for ordering in candidates:
                if time.time() > deadline - 5:
                    break
                new = ""
                for p in ordering:
                    block = p + str(m) + p
                    tail = new[-len(block):] if len(new) >= len(block) else new
                    ov = _find_overlap_suffix_prefix(tail, block)
                    new += block[ov:]
                if len(new) < m:
                    continue
                # Verify all permutations are present
                all_perms = set(permutations(range(1, m + 1)))
                found = set()
                for i in range(len(new) - m + 1):
                    w = tuple(int(c) for c in new[i:i + m])
                    if len(set(w)) == m:
                        found.add(w)
                if found == all_perms and len(new) < best_len:
                    best_len, best_word = len(new), new
            if best_word is not None:
                word = best_word
            else:
                new = ""
                for p in perms:
                    block = p + str(m) + p
                    tail = new[-len(block):] if len(new) >= len(block) else new
                    ov = _find_overlap_suffix_prefix(tail, block)
                    new += block[ov:]
                word = new

    return word
# EVOLVE-BLOCK-END
