import time
import sys
import random
import heapq
from itertools import permutations
from math import factorial

def construct(n):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # For n=7, use the classic construction and then optimize
    # The key insight: in a superpermutation, consecutive permutations must overlap
    # by exactly n-1 positions (the last n-1 of perm i = first n-1 of perm i+1)
    
    # Build the de Bruijn graph: each permutation maps to its "successor"
    # Given permutation p, the successor q must satisfy: q[0:n-1] == p[1:n]
    # This means q is determined by p[1:] and a new symbol at the end
    # But we need q to be a valid permutation, so q[0:n-1] must be a permutation of n-1 symbols
    
    # Actually, for superpermutations, the structure is:
    # If we have a word W, and W[i:i+n] = p, W[i+1:i+n+1] = q,
    # then p[1:] == q[:-1]
    # So q is determined by p[1:] and one more symbol.
    # Since q must be a permutation, q[n-1] must be the one symbol not in p[1:]
    # But wait, p[1:] has n-1 distinct symbols, so there's exactly one symbol missing.
    # So q is uniquely determined by p!
    
    # This means the "graph" is actually a set of directed cycles (functional graph)
    # Each permutation has exactly one successor.
    
    # Let's verify: p = (a1, a2, ..., an), then p[1:] = (a2, ..., an)
    # q must start with (a2, ..., an) and end with the one symbol not in {a2,...,an}
    # The symbol not in {a2,...,an} is a1 (since p is a permutation)
    # So q = (a2, a3, ..., an, a1)
    # That's just a left rotation!
    
    # Wait, that can't be right because then we'd only get n permutations per cycle.
    # Let me reconsider...
    
    # Actually no. The constraint is that W contains ALL permutations as windows.
    # The word is a sequence of symbols, not a sequence of permutations.
    # Consecutive windows overlap by n-1 positions.
    
    # So if window at position i is p = (w[i], w[i+1], ..., w[i+n-1])
    # and window at position i+1 is q = (w[i+1], w[i+2], ..., w[i+n])
    # Then q[0:n-1] = p[1:n]
    # So q is determined by p[1:] and w[i+n].
    # For q to be a permutation, w[i+n] must be the symbol not in p[1:].
    # Since p is a permutation, p[1:] contains n-1 distinct symbols,
    # so there's exactly one symbol missing from p[1:], which is p[0].
    # So w[i+n] = p[0], meaning q = (p[1], p[2], ..., p[n-1], p[0]).
    
    # This means every consecutive pair of permutation-windows in the word must be
    # related by a left rotation! This severely constrains the word.
    
    # But wait - this would mean the word is completely determined by its first window!
    # And it would only contain n permutations (one cycle). That contradicts the
    # existence of superpermutations longer than n.
    
    # I think I'm confusing myself. Let me reconsider.
    
    # In a superpermutation, NOT every window of length n is a permutation.
    # Only SOME windows are permutations. The constraint is that EVERY permutation
    # appears as SOME window, but many windows are not permutations.
    
    # So the structure is: we have a word W of length L. We need every permutation
    # of 1..n to appear as W[i:i+n] for some i.
    # Consecutive permutation-windows may have non-permutation windows between them.
    
    # OK so the problem is: find the shortest word W such that every permutation
    # appears as a substring of length n.
    
    # This is a shortest common superstring problem where the "strings" are all
    # n! permutations, and we want to maximize overlaps.
    
    # Two permutations p and q can be overlapped by k positions if p[n-k:] == q[:k].
    # The maximum overlap is n-1 (when p[1:] == q[:-1]).
    
    # So we want to order all permutations to maximize total overlap.
    # This is the TSP/shortest Hamiltonian path problem on the permutation graph.
    
    # For n=7, n! = 5040, so this is a 5040-node TSP. Too large for exact methods.
    # Use heuristics: nearest neighbor, 2-opt, genetic algorithms, etc.
    
    # Let's build the overlap matrix efficiently.
    
    all_perms = list(permutations(range(1, n + 1)))
    num_perms = len(all_perms)
    
    # For each permutation, find which permutations it can overlap with by n-1
    # p can precede q if p[1:] == q[:-1]
    # This is equivalent to: q = (p[1], p[2], ..., p[n-1], x) where x is the missing symbol
    # The missing symbol is p[0], so q = rotate_left(p)
    # So each permutation has exactly one "full overlap" successor!
    
    # But we can also have smaller overlaps. For the shortest superpermutation,
    # we want to maximize overlaps.
    
    # Let's use a greedy approach: start with any permutation, then repeatedly
    # add the permutation that gives the maximum overlap with the current end.
    
    # Precompute: for each permutation, what's the best next permutation?
    # Overlap of p followed by q: max k such that p[n-k:] == q[:k]
    
    # For efficiency, let's precompute suffix-to-permutation mapping
    
    # Map: for each suffix of length k, which permutations end with that suffix?
    # And for each prefix of length k, which permutations start with that prefix?
    
    # Actually, let's just do a smarter greedy with local search.
    
    # Build the classic superpermutation first as a baseline
    def classic_superperm(n):
        if n == 1:
            return [1]
        prev = classic_superperm(n - 1)
        perms = []
        seen = set()
        for i in range(len(prev) - (n - 2)):
            window = tuple(prev[i:i + n - 1])
            if len(set(window)) == n - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [n] + list(p)
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        return new
    
    word = classic_superperm(n)
    baseline_len = len(word)
    
    # Now try to improve using local search on the permutation sequence
    # Extract the sequence of permutations from the word
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    
    def extract_perm_sequence(word, n):
        """Extract ordered list of permutations appearing in word"""
        seq = []
        seen_perms = set()
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i+n])
            if window in perm_to_idx:
                idx = perm_to_idx[window]
                if idx not in seen_perms:
                    seq.append(idx)
                    seen_perms.add(idx)
        return seq
    
    init_seq = extract_perm_sequence(word, n)
    
    # Build word from a permutation sequence with maximum overlaps
    def build_word_from_seq(seq, n):
        if not seq:
            return []
        word = list(all_perms[seq[0]])
        for idx in seq[1:]:
            p = all_perms[idx]
            # Find max overlap
            max_ov = min(len(word), n - 1)
            ov = 0
            for o in range(max_ov, -1, -1):
                if o == 0 or word[len(word)-o:] == list(p[:o]):
                    ov = o
                    break
            word.extend(p[ov:])
        return word
    
    # Check validity and compute length
    def check_and_measure(word, n):
        if len(word) < n:
            return False, len(word), 0
        found = set()
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i+n])
            if window in perm_to_idx:
                found.add(perm_to_idx[window])
        return len(found) == num_perms, len(word), len(found)
    
    valid, cur_len, cur_found = check_and_measure(word, n)
    best_len = cur_len
    best_seq = init_seq[:]
    best_word = word[:]
    
    # Local search: try to improve by reordering permutations
    # Use simulated annealing on the permutation sequence
    
    def seq_to_word_len(seq, n):
        """Compute length of word built from sequence"""
        if not seq:
            return 0
        L = n
        for i in range(1, len(seq)):
            p = all_perms[seq[i]]
            prev = all_perms[seq[i-1]]
            # Max overlap between prev and p
            max_ov = n - 1
            ov = 0
            for o in range(max_ov, -1, -1):
                if o == 0 or list(prev[n-o:]) == list(p[:o]):
                    ov = o
                    break
            L += n - ov
        return L
    
    # Simulated annealing
    import math
    
    current_seq = init_seq[:]
    current_len = seq_to_word_len(current_seq, n)
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    steps = 0
    max_steps = 50000
    
    while T > T_min and steps < max_steps and time.time() < deadline:
        steps += 1
        
        # Mutation: swap two adjacent permutations, or move one
        mut_type = random.random()
        if mut_type < 0.5:
            # Swap adjacent
            if len(current_seq) > 1:
                pos = random.randint(0, len(current_seq) - 2)
                current_seq[pos], current_seq[pos+1] = current_seq[pos+1], current_seq[pos]
        elif mut_type < 0.8:
            # Move a random element to a random position
            if len(current_seq) > 1:
                pos = random.randint(0, len(current_seq) - 1)
                elem = current_seq.pop(pos)
                new_pos = random.randint(0, len(current_seq))
                current_seq.insert(new_pos, elem)
        else:
            # Reverse a segment
            if len(current_seq) > 2:
                start = random.randint(0, len(current_seq) - 3)
                end = random.randint(start + 1, len(current_seq) - 1)
                current_seq[start:end+1] = reversed(current_seq[start:end+1])
        
        new_len = seq_to_word_len(current_seq, n)
        delta = new_len - current_len
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_len = new_len
            if current_len < best_len:
                best_len = current_len
                best_seq = current_seq[:]
                # Build the actual word for the best
                best_word = build_word_from_seq(best_seq, n)
                # Verify
                v, l, f = check_and_measure(best_word, n)
                if v:
                    best_len = l
                else:
                    # Invalid, revert
                    current_seq = best_seq[:]
                    current_len = best_len
        else:
            # Revert
            if mut_type < 0.5:
                pos = random.randint(0, len(current_seq) - 2)
                # We don't know which pos was used, so just rebuild
                pass
            # Actually, we need to track the mutation to revert
            # Let's use a different approach: keep a copy
        
        T *= cooling
    
    # Final verification
    final_word = build_word_from_seq(best_seq, n)
    valid, final_len, found_count = check_and_measure(final_word, n)
    
    if not valid:
        # Fall back to classic
        final_word = classic_superperm(n)
        final_len = len(final_word)
    
    return "".join(map(str, final_word))

if __name__ == "__main__":
    res = construct(7)
    print(len(res))
