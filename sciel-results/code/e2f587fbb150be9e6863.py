"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius
    'radius'. Uses a greedy set-cover approach followed by redundancy removal.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # For the specific case of ternary (q=3), length 7, radius 1
    # We use a sophisticated greedy approach with vectorized operations
    # and then prune redundant codewords.
    
    # Generate all possible words
    # For q=3, n=7, radius=1: total words = 3^7 = 2187
    
    # Strategy:
    # 1. Generate a good initial set using the tetracode product construction
    # 2. Greedily add codewords to cover uncovered words
    # 3. Remove redundant codewords
    
    if q == 3 and n == 7 and radius == 1:
        # Use the known structure: direct sum of tetracode with all words on remaining coords
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        
        # Generate all words in the space
        all_words = list(itertools.product(range(3), repeat=7))
        
        # Precompute coverage: for each codeword, which words does it cover?
        # A codeword covers itself and all words at Hamming distance 1
        def get_coverage(codeword):
            covered = set()
            covered.add(codeword)
            for i in range(7):
                for val in range(3):
                    if val != codeword[i]:
                        new_word = list(codeword)
                        new_word[i] = val
                        covered.add(tuple(new_word))
            return covered
        
        # Start with tetracode product construction
        initial_codewords = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=3):
                initial_codewords.append(tuple(h + list(tail)))
        
        # Compute initial coverage
        covered = set()
        for cw in initial_codewords:
            covered.update(get_coverage(cw))
        
        # Greedily add codewords to cover remaining words
        uncovered = set(all_words) - covered
        codewords = list(initial_codewords)
        
        while uncovered and time.time() < deadline - 30:
            # Find the codeword that covers the most uncovered words
            best_cw = None
            best_count = -1
            best_new_covered = set()
            
            for candidate in all_words:
                new_covered = get_coverage(candidate) & uncovered
                count = len(new_covered)
                if count > best_count:
                    best_count = count
                    best_cw = candidate
                    best_new_covered = new_covered
            
            if best_count <= 0:
                break
            
            codewords.append(best_cw)
            covered.update(best_new_covered)
            uncovered -= best_new_covered
        
        # Redundancy removal: try to remove codewords that don't lose coverage
        i = 0
        while i < len(codewords) and time.time() < deadline - 10:
            cw_to_remove = codewords[i]
            coverage_of_cw = get_coverage(cw_to_remove)
            
            # Check if all words covered by cw_to_remove are also covered by other codewords
            can_remove = True
            for w in coverage_of_cw:
                is_covered_by_other = False
                for j, other_cw in enumerate(codewords):
                    if i == j:
                        continue
                    if w in get_coverage(other_cw):
                        is_covered_by_other = True
                        break
                if not is_covered_by_other:
                    can_remove = False
                    break
            
            if can_remove:
                codewords.pop(i)
            else:
                i += 1
        
        return [list(cw) for cw in codewords]
    
    # General fallback: greedy set cover
    all_words = list(itertools.product(range(q), repeat=n))
    
    def get_coverage_general(codeword):
        covered = set()
        covered.add(codeword)
        for i in range(n):
            for val in range(q):
                if val != codeword[i]:
                    new_word = list(codeword)
                    new_word[i] = val
                    covered.add(tuple(new_word))
        return covered
    
    codewords = []
    covered = set()
    
    while time.time() < deadline - 10:
        best_cw = None
        best_count = -1
        best_new_covered = set()
        
        for candidate in all_words:
            new_covered = get_coverage_general(candidate) - covered
            count = len(new_covered)
            if count > best_count:
                best_count = count
                best_cw = candidate
                best_new_covered = new_covered
        
        if best_count <= 0:
            break
        
        codewords.append(best_cw)
        covered.update(best_new_covered)
        
        if len(covered) == len(all_words):
            break
    
    # Redundancy removal
    i = 0
    while i < len(codewords) and time.time() < deadline - 5:
        cw_to_remove = codewords[i]
        coverage_of_cw = get_coverage_general(cw_to_remove)
        
        can_remove = True
        for w in coverage_of_cw:
            is_covered_by_other = False
            for j, other_cw in enumerate(codewords):
                if i == j:
                    continue
                if w in get_coverage_general(other_cw):
                    is_covered_by_other = True
                    break
            if not is_covered_by_other:
                can_remove = False
                break
        
        if can_remove:
            codewords.pop(i)
        else:
            i += 1
    
    return [list(cw) for cw in codewords]
# EVOLVE-BLOCK-END
