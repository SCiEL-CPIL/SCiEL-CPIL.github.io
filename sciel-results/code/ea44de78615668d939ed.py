import itertools
import time
import sys

# Hardcoded optimal superpermutations for small n
# n=1: 1
# n=2: 121
# n=3: 123121321 (len 9) - optimal
# n=4: 123412431234132413421432142314321 (len 33) - standard, optimal is 19
# Let's use standard for building up, but for n<=5 we can use known optimal.
# Actually, for n=7, the construction from n=6 is key.
# Let's generate standard S_1..S_5, then build S_6 and S_7 with optimization.

def _standard_superperm(n):
    """Generate standard superpermutation for n using recursive construction."""
    if n == 1:
        return [1]
    if n == 2:
        return [1, 2, 1]
    if n == 3:
        return [1, 2, 3, 1, 2, 1, 3, 2, 1]
    
    # For n >= 4, use recursive construction
    # S_n is built from S_{n-1} by finding all (n-1)-permutations in S_{n-1}
    # and replacing each with [perm, n, perm]
    prev = _standard_superperm(n - 1)
    
    # Find all distinct (n-1)-length windows that are permutations of 1..n-1
    # and record their order of first appearance
    seen = set()
    order = []
    for i in range(len(prev) - (n - 1) + 1):
        window = tuple(prev[i:i + n - 1])
        if window not in seen:
            seen.add(window)
            # Check if it's a permutation of 1..n-1
            if sorted(window) == list(range(1, n)):
                order.append(window)
    
    # Now build the new superpermutation
    new = []
    for perm in order:
        block = list(perm) + [n] + list(perm)
        # Find overlap with end of current new
        # Try maximum overlap
        max_ov = min(len(new), len(block))
        overlap = 0
        for ov in range(max_ov, -1, -1):
            if ov == 0 or new[len(new) - ov:] == block[:ov]:
                overlap = ov
                break
        new.extend(block[overlap:])
    
    return new

def _get_all_perms(n):
    return list(itertools.permutations(range(1, n + 1)))

def _verify_superpermutation(word, n):
    """Check if word contains all permutations of 1..n as consecutive subsequences."""
    all_perms = set(itertools.permutations(range(1, n + 1)))
    found = set()
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i + n])
        if p in all_perms:
            found.add(p)
        if len(found) == len(all_perms):
            return True
    return len(found) == len(all_perms)

def _construct_from_order(order, n, base):
    """
    Construct superpermutation for n from a base superpermutation for n-1,
    using the given order of (n-1)-permutations.
    """
    new = []
    for perm in order:
        block = list(perm) + [n] + list(perm)
        max_ov = min(len(new), len(block))
        overlap = 0
        for ov in range(max_ov, -1, -1):
            if ov == 0 or new[len(new) - ov:] == block[:ov]:
                overlap = ov
                break
        new.extend(block[overlap:])
    return new

def _extract_order_from_word(word, n):
    """Extract the order of (n-1)-permutations from a superpermutation for n."""
    seen = set()
    order = []
    for i in range(len(word) - (n - 1) + 1):
        window = tuple(word[i:i + n - 1])
        if window not in seen:
            seen.add(window)
            if sorted(window) == list(range(1, n)):
                order.append(window)
    return order

def _overlap_score(order, n):
    """
    Compute the total length of the superpermutation that would be built
    from the given order of (n-1)-permutations for symbol n.
    Smaller is better.
    """
    total = 0
    prev_end = None
    for i, perm in enumerate(order):
        block = list(perm) + [n] + list(perm)
        if i == 0:
            total = len(block)
        else:
            # Compute overlap between end of previous block and start of this block
            # But this is simplified - the actual overlap depends on the full string
            # For a quick heuristic, we can estimate based on consecutive pairs
            pass
    
    # Better approach: actually simulate the construction but only track length
    # This is expensive, so let's use a pairwise heuristic
    # For each consecutive pair (perm_i, perm_{i+1}), the overlap is determined by
    # how much suffix of (perm_i + [n] + perm_i) matches prefix of (perm_{i+1} + [n] + perm_{i+1})
    
    score = 0
    for i in range(len(order)):
        perm = list(order[i])
        block = perm + [n] + perm
        if i == 0:
            score += len(block)
        else:
            prev_perm = list(order[i - 1])
            prev_block = prev_perm + [n] + prev_perm
            # Find max overlap
            max_ov = min(len(prev_block), len(block))
            ov = 0
            for k in range(max_ov, -1, -1):
                if k == 0 or prev_block[len(prev_block) - k:] == block[:k]:
                    ov = k
                    break
            score += len(block) - ov
    return score

def _two_opt_improve_order(order, n, deadline):
    """
    Improve the order using 2-opt local search to minimize the resulting superpermutation length.
    """
    best_order = list(order)
    best_score = _overlap_score(best_order, n)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_order) - 1):
            if time.time() > deadline:
                break
            # Try reversing segment from i to j
            for j in range(i + 1, min(i + 20, len(best_order))):
                new_order = best_order[:i] + best_order[i:j+1][::-1] + best_order[j+1:]
                new_score = _overlap_score(new_order, n)
                if new_score < best_score:
                    best_order = new_order
                    best_score = new_score
                    improved = True
                    break
    
    return best_order

def construct(n):
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # For n <= 5, return standard (which is optimal for n<=3, near-optimal for n=4,5)
    if n <= 5:
        word = _standard_superperm(n)
        return "".join(map(str, word))
    
    # For n=6,7: build from standard base, then optimize the ordering step
    # Step 1: Generate standard superpermutation for n-1
    base = _standard_superperm(n - 1)
    
    # Step 2: Extract the order of (n-1)-permutations
    order = _extract_order_from_word(base, n - 1)
    
    # Step 3: Optimize the order using 2-opt
    if time.time() < deadline - 20:
        deadline_opt = deadline - 20
        order = _two_opt_improve_order(order, n, deadline_opt)
    
    # Step 4: Construct the final superpermutation
    word = _construct_from_order(order, n, base)
    
    # Verify
    if not _verify_superpermutation(word, n):
        # Fallback to standard
        word = _standard_superperm(n)
    
    return "".join(map(str, word))

# Quick self-test
if __name__ == "__main__":
    for n in range(1, 6):
        w = construct(n)
        valid = _verify_superpermutation(list(map(int, w)), n)
        print(f"n={n}, len={len(w)}, valid={valid}")
    
    # For n=6 and 7, just check timing and validity
    for n in [6, 7]:
        start = time.time()
        w = construct(n)
        elapsed = time.time() - start
        valid = _verify_superpermutation(list(map(int, w)), n)
        print(f"n={n}, len={len(w)}, valid={valid}, time={elapsed:.2f}s")
