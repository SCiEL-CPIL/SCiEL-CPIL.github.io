import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Modular inverses of 1..10 mod 101
    invs = [pow(i, -1, 101) for i in range(1, 11)]
    
    # Precompute all subsets of denominators 1..10 with their residues
    # 2^10 - 1 = 1023 subsets, very fast
    all_subsets = []
    for mask in range(1, 1 << 10):
        terms = tuple(i + 1 for i in range(10) if (mask >> i) & 1)
        residue = sum(invs[i - 1] for i in terms) % 101
        all_subsets.append((residue, len(terms), terms))
    
    # Group subsets by residue, tracking minimum size per residue
    min_size_by_residue = [11] * 101  # 11 = infinity sentinel
    best_terms_by_residue = [None] * 101
    
    for residue, size, terms in all_subsets:
        if size < min_size_by_residue[residue]:
            min_size_by_residue[residue] = size
            best_terms_by_residue[residue] = terms
    
    # The global max row length is the max of min_size_by_residue over all residues
    # Since denominators can recur across rows, each row is independent
    max_needed = max(min_size_by_residue)
    
    # Verify: all residues must be covered (min_size < 11)
    # With 1023 subsets and 101 residues, every residue is covered
    assert all(s < 11 for s in min_size_by_residue), "Some residue not covered"
    
    # Build result: for each residue, use the minimum-size subset
    result = []
    for r in range(101):
        result.append(list(best_terms_by_residue[r]))
    
    return result
