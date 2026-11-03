"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from math import factorial

def construct(n):
    """
    Construct the shortest superpermutation on n symbols.
    Returns a string of digits 1..n.
    """
    if n <= 1:
        return "1"
    
    # For n up to 4, we can use the exact recursive construction or simple search.
    # For n >= 5, we use a heuristic approach based on the "factorial construction" 
    # which yields length 1! + 2! + ... + n!. This is known to be optimal for n<=4, 
    # but not for n>=5. However, it is a valid superpermutation.
    # The task asks for the SHORTEST word. 
    # The known shortest for n=7 is 5905. The factorial sum is 1+2+6+24+120+720+5040 = 5913.
    # So we can save 8 characters potentially.
    
    # For n=7, the gap between 5913 and 5905 is small.
    # A general approach for shortening:
    # The standard construction builds a word W_n from W_{n-1}.
    # W_n is formed by taking each permutation P of 1..n-1 in the order they appear in W_{n-1}, 
    # and forming P, n, P.
    # Specifically, if W_{n-1} = ... P_1 ... P_2 ... P_3 ...
    # W_n = P_1 n P_1 n P_2 n P_2 ... P_k n P_k
    # Wait, the overlap is:
    # End of previous block is P_i.
    # Start of next block is P_{i+1}.
    # If P_{i+1} is a suffix of P_i, we can overlap.
    # Actually, the standard construction is:
    # Start with W_1 = "1"
    # W_m is built from W_{m-1}.
    # Let W_{m-1} contain permutations of 1..m-1.
    # We insert 'm' between every pair of consecutive permutations in W_{m-1} that are "adjacent" in the cycle?
    # No, the standard "factorial construction" works like this:
    # To add symbol m, we take the list of permutations of 1..m-1 in the order they appear in the current superpermutation.
    # Let the sequence of permutations be P_1, P_2, ..., P_{(m-1)!}.
    # The new superpermutation is:
    # P_1, m, P_1, m, P_2, m, P_2, ..., m, P_k, m, P_k?
    # No, it's:
    # Start with P_1.
    # Then append m.
    # Then append P_2 (overlapping with P_1 if possible? No, P_1 and P_2 are distinct permutations of m-1 elements, so they can't overlap completely, but they might share a suffix/prefix).
    # Actually, the construction is:
    # W_m = P_1 m P_1 m P_2 m P_2 ... m P_{(m-1)!} m P_{(m-1)!}
    # But we can overlap P_i and P_{i+1}.
    # Specifically, if P_{i+1} starts with a suffix of P_i, we can merge.
    # Since P_i and P_{i+1} are permutations of the same set, they have the same length.
    # The overlap is determined by the longest suffix of P_i that is a prefix of P_{i+1}.
    
    # Let's implement this construction and then try to optimize it.
    # For n=7, the length is 5913.
    # The optimal is 5905.
    # We can try to find a better ordering of permutations or better overlaps.
    
    # Given the time limit (120s) and the complexity, finding the absolute optimum for n=7 is likely impossible from scratch without specific knowledge of the 5905 construction.
    # However, we can return the factorial construction which is valid and has length 5913.
    # Score = 5913.
    # The seed program returns the factorial construction as well (score 1.0).
    # To improve, we need to find a shorter superpermutation.
    
    # Let's try to improve the overlap in the construction.
    # The standard construction assumes a specific order of permutations.
    # We can choose a different order of permutations of 1..n-1 to minimize the total length.
    # This is equivalent to finding a Hamiltonian path in a directed graph where nodes are permutations and edge weight is (m-1) - overlap.
    # But we also have to account for the 'm' insertions.
    
    # Actually, the length of the superpermutation constructed this way is:
    # (m-1)! * (m-1) + (m-1)! * 1 + sum(overlap savings)
    # Wait, each block P_i m P_i has length 2*(m-1) + 1.
    # But they are concatenated.
    # Let's just build it and measure.
    
    def build_superpermutation(n):
        if n == 1:
            return [1]
        
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            # Extract permutations of 1..m-1 from the current word
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            # Now construct the new word
            # We want to order perms to minimize the total length.
            # The length added by appending P is:
            # If this is the first P, we add len(P) + 1 + len(P) = 2*len(P) + 1?
            # No, the structure is:
            # Start with P_1.
            # Then add m.
            # Then add P_2 (overlapping with P_1).
            # Then add m.
            # Then add P_3 (overlapping with P_2).
            # ...
            # Then add m.
            # Then add P_k (overlapping with P_{k-1}).
            # Then add m.
            # Then add P_k (again? No, the construction is P_1 m P_1 m P_2 m P_2 ... m P_k m P_k).
            # Wait, the standard construction is:
            # W_m = P_1 m P_1 m P_2 m P_2 ... m P_k m P_k
            # But P_i m P_i is a block.
            # Actually, it's simpler:
            # The word is a sequence of permutations separated by m's.
            # Specifically: P_1, m, P_1, m, P_2, m, P_2, ..., m, P_k, m, P_k.
            # No, that's not right.
            
            # Let's look at the seed code:
            # block = list(p) + [m] + list(p)
            # new += block[overlap:]
            # So it appends P, m, P.
            # And overlaps with the previous block.
            
            # To optimize, we can reorder the permutations.
            # Let's try a greedy approach:
            # Start with a random permutation.
            # Then repeatedly choose the next permutation that maximizes the overlap with the current end.
            
            # Since (m-1)! can be large (720 for m=7), we can't do exhaustive search.
            # Greedy is a good heuristic.
            
            # Let's implement a greedy construction.
            new = []
            remaining = list(perms)
            random.shuffle(remaining)
            
            # Start with the first permutation
            current_p = remaining.pop(0)
            new.extend(current_p)
            new.append(m)
            new.extend(current_p)
            
            # Now, we need to add the remaining permutations.
            # The current end is current_p.
            # We want to append P_next such that it overlaps with current_p.
            # The block to append is P_next m P_next.
            # But we only append the part that doesn't overlap with the current end.
            # The current end is a permutation of 1..m-1.
            # P_next is also a permutation of 1..m-1.
            # The overlap is the longest suffix of current_p that is a prefix of P_next.
            
            # Wait, the block is P_next m P_next.
            # The first part is P_next.
            # So we check overlap between current end (current_p) and P_next.
            # If overlap is L, we append P_next[L:] + [m] + P_next.
            
            # After appending, the new end is P_next.
            
            while remaining:
                best_idx = -1
                best_overlap = -1
                for i, p in enumerate(remaining):
                    # Calculate overlap between current_p and p
                    # current_p is a tuple, p is a tuple
                    max_ov = len(current_p)
                    for l in range(max_ov, -1, -1):
                        if current_p[-l:] == p[:l]:
                            if l > best_overlap:
                                best_overlap = l
                                best_idx = i
                            break
                if best_idx == -1:
                    best_idx = 0
                    best_overlap = 0
                p = remaining.pop(best_idx)
                # Append p[best_overlap:] + [m] + p
                new.extend(p[best_overlap:])
                new.append(m)
                new.extend(p)
                current_p = p
            
            word = new
        return word
    
    # For n=7, this greedy approach might find a shorter word than the standard order.
    # Let's run it multiple times with different seeds to find a good one.
    
    best_word = None
    best_len = float('inf')
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    for seed in range(100):
        if time.time() > deadline:
            break
        random.seed(seed)
        word = build_superpermutation(n)
        if len(word) < best_len:
            best_len = len(word)
            best_word = word
    
    if best_word is None:
        # Fallback to standard construction
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
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        best_word = word
    
    return "".join(map(str, best_word))
# EVOLVE-BLOCK-END
