"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct(n):
    # Deadline guard: evaluator kills after 150s; leave safety margin
    deadline = time.time() + 130.0
    
    # For small n, use the classic recursive construction (optimal for n<=4)
    if n <= 4:
        return _classic_superperm(n)
    
    # For n >= 5, use a pattern-based constructive approach
    # The classic construction gives length sum_{k=1}^n k!
    # For n=5: 1!+2!+3!+4!+5! = 1+2+6+24+120 = 153
    # But known shorter superpermutations exist.
    
    # Strategy: Use the "symbol rotation" constructive algorithm.
    # Build a superpermutation by starting with a base pattern and
    # using cyclic shifts to generate new permutations with maximal overlap.
    
    # For n=5, the optimal length is 152 (one less than the classic 153).
    # For n=6, the optimal is unknown but the classic gives 1+2+6+24+120+720 = 873.
    # For n=7, classic gives 1+2+6+24+120+720+5040 = 5913.
    # The task says shortest known is 5905, lower bound 5884.
    
    # Let's try to improve upon the classic construction.
    # The classic construction for step m inserts the new symbol m between
    # each occurrence of a (m-1)-permutation, creating blocks of length 2*(m-1)+1 = 2m-1.
    # Total length = 1 + sum_{m=2}^n (m-1)! * (2m-1) ... wait, let me recalculate.
    # Actually the classic recursive construction:
    # S_1 = "1" (length 1)
    # S_m = for each permutation p of S_{m-1} in order of appearance: p, m, p
    # with maximum overlap between consecutive blocks.
    # Length of S_m = (m-1)! * (2*(m-1) + 1) - (m-1)! + 1 ... 
    # Actually it's known that |S_n| = 1! + 2! + 3! + ... + n!
    # |S_5| = 153, |S_6| = 873, |S_7| = 5913
    
    # The task mentions 5905 as the shortest known, which is 5913 - 8.
    # This suggests there's a way to save 8 characters.
    
    # Let me try a different approach: use a greedy/hill-climbing method
    # to find a shorter superpermutation.
    
    # First, generate the classic superpermutation as a starting point
    word = _classic_superperm(n)
    current_len = len(word)
    
    # Try to shorten via local search: find and remove redundant characters
    # A character can be removed if every permutation still appears after removal.
    # This is expensive to check, so let's try a smarter approach.
    
    # Alternative: use the known fact that for n=5, optimal is 152.
    # The improvement comes from a specific pattern modification.
    
    # For now, return the classic construction which is guaranteed valid.
    # The length for n=7 is 5913, which is within the n*n! = 35280 limit.
    
    return word


def _classic_superperm(n):
    """Classic recursive construction of a superpermutation of length 1!+2!+...+n!"""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))


def _try_shorten(word, n, deadline):
    """Try to shorten the superpermutation by removing redundant characters."""
    word = list(word)
    length = len(word)
    
    # Build a set of all permutations needed
    target_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Track which permutations are covered
    def get_covered_perms(w):
        covered = set()
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            if len(set(p)) == n:
                covered.add(p)
        return covered
    
    # This is too slow for n=7. Let's use a different approach.
    # Instead, try specific known improvements.
    
    # For n=5, the improvement from 153 to 152 is well-known.
    # The pattern involves removing one character from a specific position.
    
    # Let's try a greedy approach: try removing each character and check
    # if all permutations are still covered. Use a bitmap for efficiency.
    
    if n > 6:
        # Too slow for n=7 with full verification
        return "".join(map(str, word))
    
    # Use a faster verification method
    # Represent each permutation as an integer for fast lookup
    perm_to_id = {}
    id_to_perm = []
    for i, p in enumerate(itertools.permutations(range(1, n + 1))):
        perm_to_id[p] = i
        id_to_perm.append(p)
    num_perms = len(id_to_perm)
    
    def perm_to_int(p):
        # Encode permutation as integer using factorial number system
        result = 0
        for i, val in enumerate(p):
            result = result * n + val
        return result
    
    # This encoding is not unique for all tuples, but for permutations of 1..n it is
    # Actually, let's just use the tuple directly in a set.
    
    # For n=5, num_perms = 120, which is manageable.
    # For n=6, num_perms = 720, still manageable.
    
    if n <= 6:
        covered = get_covered_perms(word)
        
        # Try removing each position
        improved = False
        for pos in range(length - 1, -1, -1):
            if time.time() > deadline:
                break
            # Try removing position pos
            test_word = word[:pos] + word[pos+1:]
            new_covered = get_covered_perms(test_word)
            if new_covered == target_perms:
                word = test_word
                covered = new_covered
                improved = True
                # Continue trying to remove more
                # Reset position since word changed
                pos = len(word) - 1
        
        if improved:
            return "".join(map(str, word))
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
