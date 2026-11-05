import time
import random
from itertools import permutations

_START_TIME = time.time()
DEADLINE = 145.0

def _remaining_time():
    return DEADLINE - (time.time() - _START_TIME)

def _standard_construct(n):
    if n == 1:
        return [1]
    if n == 2:
        return [1, 2, 1]
    if n == 3:
        return [1, 3, 1, 2, 1, 3, 1]
    if n == 4:
        return [1, 4, 1, 3, 1, 2, 1, 4, 1, 3, 1, 2, 4, 1, 3, 1, 2, 4, 1]
    # Recursive standard construction
    word = _standard_construct(n-1)
    perms = list(permutations(range(1, n)))
    # The standard construction for n uses the order of permutations of 1..n-1
    # But _standard_construct(n-1) is a word, not a list of perms.
    # We need to extract the order of (n-1)-permutations from the word?
    # Or just use the lexicographic order of (n-1)-permutations.
    
    # Let's just use lexicographic order for the standard construct fallback
    # For n=5,6,7, the "standard" construct usually implies the specific recursive
    # ordering that generates the known lengths.
    # However, for the purpose of this task, we are optimizing n=7.
    # For n<7, we can return the known shortest or standard.
    
    # Actually, for n=5 and 6, the standard construction is optimal.
    # Lengths: n=1:1, n=2:3, n=3:7, n=4:13, n=5:21, n=6:31, n=7:59? No.
    # Standard lengths: 1, 3, 7, 13, 21, 31, 59 is NOT standard.
    # Standard n=7 length is 8527.
    # Standard n=6 length is 1957? No.
    # Let's calculate standard length:
    # L(n) = (n-1)! (2n-1) - sum overlaps.
    # For n=4: 6 * 7 - overlaps.
    # Standard order: 123, 132, 213, 231, 312, 321.
    # Overlaps:
    # 123 -> 132: suf 123, pref 132. Match '1'? No, '1'=='1', '2'!='3'. Overlap 1.
    # 132 -> 213: suf 132, pref 213. Match '2'=='2'? No. '3'!='1'. Overlap 0.
    # This is getting complicated to hardcode.
    
    # For n=5,6, we can just use the recursive definition with lexicographic order.
    if n == 5:
        return _construct_from_lex_order(5)
    if n == 6:
        return _construct_from_lex_order(6)
    return []

def _construct_from_lex_order(n):
    if n == 1:
        return [1]
    if n == 2:
        return [1, 2, 1]
    if n == 3:
        return [1, 3, 1, 2, 1, 3, 1]
    if n == 4:
        return [1, 4, 1, 3, 1, 2, 1, 4, 1, 3, 1, 2, 4, 1, 3, 1, 2, 4, 1]
    
    # For n=5, use perms of 1..4
    perms = list(permutations(range(1, n)))
    word = []
    for i, p in enumerate(perms):
        block = list(p) + [n] + list(p)
        if i == 0:
            word = block
        else:
            # Find overlap
            prev = perms[i-1]
            max_ov = min(len(prev), len(p))
            ov = 0
            for o in range(max_ov, 0, -1):
                if prev[-o:] == p[:o]:
                    ov = o
                    break
            word += block[ov:]
    return word

def _overlap_len(a, b):
    max_ov = min(len(a), len(b))
    for ov in range(max_ov, 0, -1):
        if a[-ov:] == b[:ov]:
            return ov
    return 0

def _construct_from_order(order, n):
    if not order:
        return list(range(1, n + 1))
    
    # order is a list of tuples (permutations of 1..n-1)
    word = list(order[0]) + [n] + list(order[0])
    
    for i in range(1, len(order)):
        prev = order[i-1]
        curr = order[i]
        
        # The block to add is curr + [n] + curr
        # We append curr + [n] + curr to word
        # Overlap is between suffix of word (which ends with prev) and prefix of curr
        # So overlap is between prev and curr
        
        max_ov = min(len(prev), len(curr))
        ov = 0
        for o in range(max_ov, 0, -1):
            if prev[-o:] == curr[:o]:
                ov = o
                break
        
        word += list(curr)[ov:]
        word += [n]
        word += list(curr)
        
    return word

def _verify_superpermutation(word, n):
    if len(word) < n:
        return False
    found = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        found.add(window)
    return len(found) == _factorial(n)

def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def _construct_n7_optimized():
    # Generate permutations of 1..6
    perms_6 = list(permutations(range(1, 7)))
    n_perms = len(perms_6)
    
    # Compute overlap matrix
    # overlap_matrix[i][j] = length of longest suffix of perms_6[i] that is prefix of perms_6[j]
    overlap_matrix = [[0] * n_perms for _ in range(n_perms)]
    for i in range(n_perms):
        p_i = perms_6[i]
        for j in range(n_perms):
            if i != j:
                p_j = perms_6[j]
                # Max possible overlap is 5 (since length is 6, and they are distinct, 
                # but wait, if they share 5, they are identical? 
                # If suffix of length 5 of A is prefix of length 5 of B, 
                # and A and B are permutations, then A[1:] == B[:5].
                # Since A is a permutation, A[0] is the missing element.
                # B[5] is the missing element for B.
                # If A[1:] == B[:5], then A is a rotation of B?
                # No, A is a permutation of 1..6.
                # Example: 123456, 234561. Suffix 23456, Prefix 23456. Overlap 5.
                # So max overlap is 5.
                max_ov = 5
                for ov in range(max_ov, 0, -1):
                    if p_i[-ov:] == p_j[:ov]:
                        overlap_matrix[i][j] = ov
                        break
    
    # Initial order: lexicographic
    best_path = list(range(n_perms))
    
    # Calculate initial score (sum of overlaps)
    def path_score(path):
        total = 0
        for i in range(len(path) - 1):
            total += overlap_matrix[path[i]][path[i+1]]
        return total
    
    current_score = path_score(best_path)
    
    # Local Search: Or-Opt (k=1, 2, 3) and 2-Opt
    # We want to maximize score.
    
    improved = True
    # Time budget for search
    search_deadline = _START_TIME + 130.0
    
    while improved and time.time() < search_deadline:
        improved = False
        
        # Try Or-Opt moves for k=1, 2, 3
        for k in [1, 2, 3]:
            for i in range(n_perms - k):
                if time.time() > search_deadline:
                    break
                
                # Segment to move: path[i : i+k]
                # Old neighbors:
                # Left: path[i-1] (if i>0)
                # Right: path[i+k] (if i+k < n_perms)
                
                old_left = overlap_matrix[best_path[i-1]][best_path[i]] if i > 0 else 0
                old_right = overlap_matrix[best_path[i+k-1]][best_path[i+k]] if i+k < n_perms else 0
                old_cost = old_left + old_right
                
                # Try inserting at j
                for j in range(n_perms - k + 1):
                    if i <= j < i + k:
                        continue
                    
                    # New neighbors for the segment at j:
                    # Left: path[j-1] (if j>0)
                    # Right: path[j] (if j+k < n_perms, but careful with index shift)
                    
                    # If j < i:
                    #   The segment is moved left.
                    #   New Left: path[j-1] (if j>0)
                    #   New Right: path[j] (the element that was at j, now at j+k)
                    #   Wait, if we remove i..i+k-1, and insert at j.
                    #   If j < i:
                    #     Elements j..i-1 shift right by k.
                    #     The element after the inserted segment is the one that was at j.
                    #     So New Right Neighbor is path[j].
                    #   If j > i+k:
                    #     Elements i+k..j-1 shift left by k.
                    #     The element before the insertion point is path[j-1].
                    #     The element after the insertion point is path[j].
                    #     So New Left: path[j-1], New Right: path[j].
                    
                    # Let's verify the "New Right" neighbor.
                    # The segment is P[i..i+k-1].
                    # It is placed at j.
                    # The element immediately following the segment in the new path is the element that was at j in the old path (if j < i) or the element that was at j in the old path (if j > i+k).
                    # In both cases, the element at index j in the OLD path is the one that follows the segment in the NEW path?
                    # If j < i:
                    #   Old: ... P[j-1] P[j] ... P[i-1] P[i] ... P[i+k-1] P[i+k] ...
                    #   New: ... P[j-1] P[i] ... P[i+k-1] P[j] ... P[i-1] P[i+k] ...
                    #   The element after P[i+k-1] is P[j]. Correct.
                    # If j > i+k:
                    #   Old: ... P[i-1] P[i] ... P[i+k-1] P[i+k] ... P[j-1] P[j] ...
                    #   New: ... P[i-1] P[i+k] ... P[j-1] P[i] ... P[i+k-1] P[j] ...
                    #   The element after P[i+k-1] is P[j]. Correct.
                    
                    new_left = overlap_matrix[best_path[j-1]][best_path[i]] if j > 0 else 0
                    # The element after the segment is best_path[j] in the old path?
                    # Yes, as derived above.
                    # But wait, if j+k == n_perms, there is no right neighbor.
                    # The condition for existence of right neighbor is j < n_perms - k?
                    # No, the segment occupies k slots.
                    # If we insert at j, the segment ends at j+k-1.
                    # The next element is at j+k.
                    # In the old path, the element at j+k is...
                    # If j < i:
                    #   The element at j+k in the new path is the element that was at j in the old path.
                    #   So we use best_path[j].
                    #   This is valid if j < n_perms.
                    #   But we need to check if there IS a right neighbor.
                    #   The right neighbor exists if j < n_perms - k? No.
                    #   The right neighbor is the element at index j+k in the NEW path.
                    #   This element exists if j+k < n_perms.
                    #   In the old path, this element is at index j (if j < i) or j (if j > i+k)?
                    #   Actually, the element at index j in the OLD path is always the one that ends up at index j+k in the NEW path (if j < i) or index j-k in the NEW path (if j > i+k)?
                    #   No.
                    #   Let's trace indices.
                    #   Remove i..i+k-1.
                    #   If j < i:
                    #     New path: [0..j-1] + Segment + [j..i-1] + [i+k..N-1]
                    #     The element at index j+k in New Path is the element at index j in Old Path.
                    #     This element exists if j < N.
                    #     So Right Neighbor is best_path[j].
                    #     It exists if j < N.
                    #     But we only care about the overlap if the neighbor exists.
                    #     The neighbor exists if j+k < N?
                    #     The segment ends at j+k-1. The next index is j+k.
                    #     If j+k == N, there is no next element.
                    #     So condition: j < N - k.
                    #     If j < N - k, then best_path[j] is a valid index in the old path?
                    #     Yes, j < N.
                    
                    if j < n_perms - k:
                        new_right = overlap_matrix[best_path[i+k-1]][best_path[j]]
                    else:
                        new_right = 0
                        
                    new_cost = new_left + new_right
                    delta = new_cost - old_cost
                    
                    if delta > 0:
                        # Apply move
                        segment = best_path[i:i+k]
                        del best_path[i:i+k]
                        if j > i:
                            j -= k
                        best_path.insert(j, *segment)
                        current_score += delta
                        improved = True
                        break # Restart loop for this k
                if improved:
                    break
            if improved:
                break
        
        # If no Or-Opt improvement, try 2-Opt
        if not improved:
            for i in range(n_perms - 1):
                if time.time() > search_deadline:
                    break
                for j in range(i + 1, n_perms):
                    if time.time() > search_deadline:
                        break
                    
                    # Reverse path[i:j+1]
                    # Old edges: (i-1, i) and (j, j+1)
                    # New edges: (i-1, j) and (i, j+1)
                    
                    old_left = overlap_matrix[best_path[i-1]][best_path[i]] if i > 0 else 0
                    old_right = overlap_matrix[best_path[j]][best_path[j+1]] if j+1 < n_perms else 0
                    old_cost = old_left + old_right
                    
                    new_left = overlap_matrix[best_path[i-1]][best_path[j]] if i > 0 else 0
                    new_right = overlap_matrix[best_path[i]][best_path[j+1]] if j+1 < n_perms else 0
                    new_cost = new_left + new_right
                    
                    delta = new_cost - old_cost
                    if delta > 0:
                        best_path[i:j+1] = reversed(best_path[i:j+1])
                        current_score += delta
                        improved = True
                        break
                if improved:
                    break
                    
    # Construct word from best_path
    order = [perms_6[idx] for idx in best_path]
    return _construct_from_order(order, 7)

def construct(n):
    if n <= 0:
        return ""
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1312131"
    if n == 4:
        return "1413121413124131241"
    if n == 5:
        return "".join(map(str, _construct_from_lex_order(5)))
    if n == 6:
        return "".join(map(str, _construct_from_lex_order(6)))
    
    if n == 7:
        try:
            word = _construct_n7_optimized()
            if _verify_superpermutation(word, 7):
                return "".join(map(str, word))
        except Exception:
            pass
        # Fallback to standard
        return "".join(map(str, _construct_from_lex_order(7)))
    
    return "".join(map(str, _construct_from_lex_order(n)))

# Test locally if needed
# if __name__ == "__main__":
#     w = construct(7)
#     print(len(w))
#     print(_verify_superpermutation(list(map(int, w)), 7))
