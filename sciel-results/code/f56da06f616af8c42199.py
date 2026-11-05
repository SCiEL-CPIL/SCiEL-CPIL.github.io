# EVOLVE-BLOCK-START
import time
import sys
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    if n <= 3:
        # Standard construction is optimal for n<=3
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
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # Standard recursive construction to get initial sequence
    def get_standard_sequence(n):
        word = [1]
        seq = [(1,)]
        for m in range(2, n + 1):
            new_seq = []
            # Extract (m-1)-permutations from current word
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            for p in perms:
                block = list(p) + [m] + list(p)
                # Determine overlap with current word end
                # But we are building the sequence of m-perms
                # The standard construction inserts m into the middle of p
                # The new m-perm is p + [m] + p? No.
                # The block added is p + [m] + p.
                # The new permutation introduced is the window ending at the new m?
                # Actually, the standard construction generates a specific list of permutations.
                # Let's just generate the word and extract the sequence of permutations in order of first appearance?
                # Or better, just use the word and extract the sequence of windows that are permutations.
                pass
            
            # Simpler: Just generate the word, then extract the sequence of permutations
            # that appear in the order they are "introduced" or just in order of occurrence?
            # The superpermutation property is that every permutation appears.
            # The "sequence" for the overlap optimization is the order in which we append them.
            # If we have a valid superpermutation, we can extract the sequence of permutations
            # by scanning left to right and taking the first time we see a new permutation?
            # No, that doesn't guarantee the "append" logic holds.
            # The append logic holds if we define the sequence as the order of the "new" permutations
            # added in the standard construction.
            
            # Let's stick to the standard construction word, and then extract the sequence
            # of permutations by finding the unique permutation that ends at each position
            # where a new symbol is appended?
            
            # Actually, for n=7, the standard construction yields a specific sequence.
            # We can just generate the word, and then find the sequence of permutations
            # that corresponds to the standard construction's order.
            # The standard construction adds one new permutation at a time.
            # The new permutation is the one that contains the new symbol m in the "center".
            
            # Alternative: Just use the word, and for the local search, we need a sequence of 5040 permutations.
            # We can define the sequence as follows:
            # Start with the first n symbols.
            # Then, for each subsequent symbol, it extends the previous permutation.
            # This doesn't directly give a sequence of permutations, but a path in the De Bruijn graph.
            # The nodes of the De Bruijn graph are (n-1)-permutations.
            # The path visits each (n-1)-permutation multiple times?
            # No, the superpermutation corresponds to a path in the graph of (n-1)-permutations
            # that visits every node exactly once? No, that would be a Hamiltonian path on (n-1)! nodes.
            # The number of (n-1)-permutations is (n-1)!.
            # For n=7, 720 nodes.
            # A Hamiltonian path on 720 nodes has 719 edges.
            # Each edge corresponds to adding 1 symbol.
            # Total length = n + 719 * 1 = 726? No.
            # The standard superpermutation length is much larger.
            # This means the path in the De Bruijn graph is NOT a simple Hamiltonian path.
            # It visits nodes multiple times.
            
            # So the "sequence of permutations" model is:
            # We have a list of N! permutations.
            # We concatenate them with overlaps.
            # This is equivalent to a walk in the permutation graph (nodes are N! perms, edges if overlap > 0).
            # We want a Hamiltonian path in this graph?
            # If we find a Hamiltonian path in the graph where edge weight is overlap,
            # and we maximize sum of weights, we get the shortest superpermutation.
            # The graph is complete (any two perms can be connected, overlap >= 0).
            # So we are looking for a TSP tour (or path) maximizing overlap.
            
            # So, generate the list of all 5040 permutations.
            # Initialize order with the standard construction order.
            # How to get standard construction order?
            # The standard construction word can be parsed to extract the order.
            # The first permutation is the first n symbols.
            # The next permutation is the next new permutation that appears.
            # In the standard construction, each step adds exactly one new permutation.
            # So we can just scan the word and collect the permutations in the order they first appear.
            
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
                    overlap = 0
                    for o in range(min(len(new), len(block)), -1, -1):
                        if o == 0 or new[len(new) - o:] == block[:o]:
                            overlap = o
                            break
                    new += block[overlap:]
                word = new
            
            # Extract sequence
            seq = []
            seen_perms = set()
            for i in range(len(word) - n + 1):
                p = tuple(word[i:i+n])
                if p not in seen_perms:
                    seen_perms.add(p)
                    seq.append(p)
            return seq

    seq = get_standard_sequence(n)
    
    # Local search to maximize overlap
    def get_overlap(u, v):
        # Max k such that u[-k:] == v[:k]
        # k in 1..n-1
        # Since u != v, k < n
        for k in range(n - 1, 0, -1):
            if u[n-k:] == v[:k]:
                return k
        return 0

    def total_overlap(seq):
        s = 0
        for i in range(len(seq) - 1):
            s += get_overlap(seq[i], seq[i+1])
        return s

    current_seq = seq[:]
    current_score = total_overlap(current_seq)
    
    # 2-opt search
    # Try reversing segments
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Randomize start point to avoid local minima?
        # Or just iterate all pairs? 5040^2 / 2 is 12.5 million pairs.
        # Each check is O(n). 12.5M * 7 ops = 87M ops. Might be slow in Python.
        # Let's try a subset of moves per iteration or random moves.
        
        # Strategy: Random 2-opt moves
        for _ in range(10000): # Fixed number of random moves per "pass"
            if time.time() > deadline - 10:
                break
            i = random.randint(0, len(current_seq) - 2)
            j = random.randint(i + 1, len(current_seq) - 1)
            
            # Calculate delta
            # Edges affected: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
            # Actually, reversing i..j affects:
            # Left boundary: (i-1, i) becomes (i-1, j)
            # Right boundary: (j, j+1) becomes (i, j+1)
            # Internal edges: (k, k+1) becomes (k, k-1) which is same overlap?
            # Overlap(u, v) is NOT symmetric. Overlap(u, v) != Overlap(v, u) generally.
            # Wait, overlap is defined as suffix of u == prefix of v.
            # So Overlap(u, v) is not symmetric.
            # So we must recalculate all internal edges?
            # No, the segment is reversed.
            # Original: ... A, u_i, u_{i+1}, ..., u_j, B ...
            # New: ... A, u_j, u_{j-1}, ..., u_i, B ...
            # Edges:
            # Old: (A, u_i), (u_i, u_{i+1}), ..., (u_{j-1}, u_j), (u_j, B)
            # New: (A, u_j), (u_j, u_{j-1}), ..., (u_{i+1}, u_i), (u_i, B)
            
            # We need to sum overlaps for old and new.
            # If segment is small, we can just compute.
            # If segment is large, it's expensive.
            # Let's restrict to small segments first (e.g., length < 50) or random sampling.
            
            if j - i > 50:
                continue
                
            old_score = 0
            new_score = 0
            
            # Left boundary
            if i > 0:
                old_score += get_overlap(current_seq[i-1], current_seq[i])
                new_score += get_overlap(current_seq[i-1], current_seq[j])
            else:
                # No left boundary cost
                pass
            
            # Right boundary
            if j < len(current_seq) - 1:
                old_score += get_overlap(current_seq[j], current_seq[j+1])
                new_score += get_overlap(current_seq[i], current_seq[j+1])
            else:
                pass
            
            # Internal
            for k in range(i, j):
                old_score += get_overlap(current_seq[k], current_seq[k+1])
                new_score += get_overlap(current_seq[k+1], current_seq[k])
            
            delta = new_score - old_score
            if delta > 0:
                current_seq[i:j+1] = current_seq[i:j+1][::-1]
                current_score += delta
                improved = True
    
    # Reconstruct word from sequence
    word = list(current_seq[0])
    for i in range(1, len(current_seq)):
        prev = current_seq[i-1]
        curr = current_seq[i]
        k = get_overlap(prev, curr)
        word.extend(curr[k:])
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
