"""Superpermutation construction via shortest Hamiltonian path on the (n-1)!-perm graph."""
import time
import random
import math

# EVOLVE-BLOCK-START

def construct(n):
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412134121431213421"

    # --- Build the transition graph on permutations of 1..(n-1) ---
    m = n - 1
    new_sym = n
    base_syms = list(range(1, m + 1))

    def perm_to_index(p):
        # Factorial number system -> index
        idx = 0
        mult = 1
        avail = list(range(m))
        for v in p:
            pos = avail.index(v - 1)
            idx += pos * mult
            avail.pop(pos)
            mult *= (m - 1 - (m - len(avail) - 1))
            mult = 1
            # recompute mult properly
        # simpler: recompute
        idx = 0
        avail = list(range(m))
        for i, v in enumerate(p):
            pos = avail.index(v - 1)
            idx += pos * math.factorial(m - 1 - i)
            avail.pop(pos)
        return idx

    def index_to_perm(idx):
        avail = list(range(1, m + 1))
        p = []
        for i in range(m):
            f = math.factorial(m - 1 - i)
            pos = idx // f
            idx %= f
            p.append(avail.pop(pos))
        return p

    # Precompute adjacency: for each permutation (as a tuple of symbols),
    # the list of (next_perm_index, overlap_length)
    # overlap between A (len m) and B (len m) via inserting new_sym:
    #   longest suffix of A that is a subsequence of B, then new_sym, then
    #   the remaining prefix of B. overlap = len(suffix) + 1 + len(prefix_remaining)
    # Actually the merged block is suffix + [new_sym] + (B minus the prefix that equals the suffix's continuation)
    # Let me define: find longest k such that A[m-k:] (as a sequence) appears as a subsequence in B
    # starting at position 0. Then the merged word for the transition is:
    #   A[m-k:] + [new_sym] + B[k'] where k' is the number of leading elements of B
    #   that are "covered". Hmm, let me think more carefully.

    # Standard approach: the word built is a concatenation where consecutive permutations
    # A, B (both length m) are joined by: (suffix of A) + [new_sym] + (prefix of B)
    # where the suffix of A is the longest suffix of A that is a subsequence of B,
    # and the prefix of B is the part of B after the longest prefix of B that is a
    # subsequence of A. Actually this is getting complex.

    # Let me use a cleaner formulation. The superpermutation word, when we look at
    # the positions of the new symbol, splits the word into segments. Between
    # consecutive occurrences of the new symbol, we have a permutation of the m old
    # symbols (possibly with some repetition at boundaries).

    # Alternative cleaner approach: use the known "greedy + local search" on the
    # permutation sequence. The word length = m + sum over transitions of (m - overlap_i)
    # where overlap_i is the overlap between consecutive perm-blocks.

    # Let me just compute overlaps directly with a simple method:
    # Given consecutive permutations A and B (both permutations of 1..m),
    # the contribution to the word when going A -> B is:
    #   We need the word to contain A (as the m symbols before new_sym) and B
    #   (as the m symbols after new_sym). The overlap is: the longest suffix of A
    #   that matches a prefix of B in the "interleaved" sense.

    # Actually, the cleanest: the word is built as:
    #   [A_1] new_sym [A_2] new_sym [A_3] ... new_sym [A_k]
    # where A_i are permutations of 1..m, and the word must contain every permutation
    # of 1..n. The total length = k*m + k (for new_sym) + overlaps...

    # I'm overcomplicating. Let me use the DIRECT construction and then optimize
    # the ORDER of permutations via local search, measuring actual word length.

    # --- Direct approach: generate a permutation order, build the word, measure length ---
    # The word construction given an order of (n-1)! permutations:
    #   word starts with perm_1 (length m)
    #   for each subsequent perm_i: append new_sym, then append perm_i
    #     but with maximum overlap: find longest suffix of current word
    #     that is a prefix of (new_sym + perm_i)
    # Actually the standard: the word is a sequence where between each pair of
    # consecutive (n-1)-permutations we insert the new symbol. The overlap
    # between consecutive blocks A and B (via new_sym) is:
    #   longest k such that the last k symbols of A equal the first k symbols of B? No.

    # Let me just do it empirically. Build the word for a given permutation order:
    #   word = list(perm_1)
    #   for each next perm B:
    #       candidate = [new_sym] + list(B)
    #       find max overlap: largest o such that word[-o:] == candidate[:o]
    #       word += candidate[o:]

    # The total length depends on the ORDER of permutations. We want to minimize
    # the total length = len(perm_1) + sum of (len(candidate) - overlap_i).

    # This is equivalent to finding a shortest Hamiltonian path in a directed graph
    # where edge weight (A->B) = len([new_sym]+B) - overlap(word ending in A, [new_sym]+B).
    # But the overlap depends on the full word, not just A. However, if the overlap
    # only depends on A (the last m symbols), then it's a proper graph.

    # Claim: the overlap between the current word (ending with permutation A of length m)
    # and the next block [new_sym] + B depends only on A and B, because:
    #   - The overlap can be at most m+1 (the whole new_sym+B block)
    #   - For overlap o <= m: word[-o:] must equal [new_sym, B_1, ..., B_{o-1}]
    #     This requires word[-o:] to start with new_sym, which means the last o symbols
    #     of A end with new_sym... but A only contains 1..m, not new_sym. So overlap
    #     involving new_sym in the middle is impossible unless o accounts for it.
    #   - Actually word[-o:] for o <= m is a suffix of A (which has no new_sym).
    #     candidate[:o] for o <= m+1: if o <= 1, candidate[:o] = [] or [new_sym].
    #     For o=1: candidate[:1]=[new_sym], word[-1:] is a symbol from A (1..m), no match.
    #     For o >= 2: candidate[:o] = [new_sym, B_1, ..., B_{o-2}], which contains new_sym,
    #     but word[-o:] (for o <= m) is a suffix of A with no new_sym. No match.
    #   - So overlap can only be 0, or it must be such that candidate[:o] has no new_sym,
    #     which means o=0. OR the overlap spans the boundary differently.

    # Hmm, I think the overlap is always 0 in this naive construction! That means the
    # word length is just (n-1)! * (m+1) + m, which is way too long.

    # I clearly misunderstand the construction. Let me look at the seed code again.

    # The seed code: for each permutation p of (m-1) symbols, it appends p + [m] + p.
    # So the block is p, then m, then p. The overlap is between consecutive blocks.
    # Block_i = p_i + [m] + p_i, and we overlap the end of block_i with the start of block_{i+1}.

    # So the word is: p_1 [m] p_1 p_2 [m] p_2 p_3 [m] p_3 ... p_k [m] p_k
    # with overlaps between consecutive (p_i [m] p_i) and (p_{i+1} [m] p_{i+1}).

    # The overlap between p_i + [m] + p_i and p_{i+1} + [m] + p_{i+1}:
    #   We need the longest suffix of (p_i + [m] + p_i) that equals a prefix of (p_{i+1} + [m] + p_{i+1}).

    # This overlap CAN be positive! For example, if p_i ends with some symbols that
    # match the start of p_{i+1}.

    # So the graph has nodes = permutations of (m-1) symbols, and edge weight
    # (A -> B) = len(block_B) - overlap(block_A, block_B), where block_X = X + [m] + X.

    # Wait, in the seed, m is the new symbol being added, and p is a permutation of
    # 1..(m-1). So for the final n, the last iteration has m=n, and p is a permutation
    # of 1..(n-1). The block is p + [n] + p, of length 2(n-1)+1.

    # Let me re-examine: the seed builds up from n=1 to n. At step m (adding symbol m),
    # it takes the current word (a superpermutation of 1..(m-1)), extracts all
    # (m-1)-length windows that are permutations of 1..(m-1), in order of appearance,
    # and for each such permutation p, appends p + [m] + p with overlap.

    # So the "nodes" at step m are the permutations of 1..(m-1) that appear in the
    # (m-1)-superpermutation word, in the order they appear. The number of such
    # permutations is (m-1)! (every permutation appears at least once).

    # For n=7: at the final step, we have 720 nodes (permutations of 1..6), and we
    # need a Hamiltonian path through all 720 nodes minimizing total length.

    # Edge weight from A to B (both permutations of 1..(m-1), so length m-1):
    #   block_A = A + [m] + A, length 2(m-1)+1
    #   block_B = B + [m] + B, length 2(m-1)+1
    #   overlap = longest o such that block_A[-o:] == block_B[:o]
    #   edge_cost = len(block_B) - overlap

    # Total word length = len(block_first) + sum of edge_costs for the path.

    # For m=7 (n=7): block length = 2*6+1 = 13. 720 nodes.
    # If all overlaps were 0: total = 13 + 719*13 = 9350. Too long.
    # The classic construction achieves 1!+2!+...+7! = 1+2+6+24+120+720+5040 = 5913.
    # So average overlap must be about (9350-5913)/719 ≈ 4.78 per edge.

    # OK so the problem is: find a Hamiltonian path through 720 nodes (permutations of 1..6)
    # minimizing total cost, where edge cost depends on the overlap of the 13-length blocks.

    # This is a TSP-like problem on 720 nodes. Too large for exact solution, but we can
    # use heuristics: nearest neighbor, 2-opt, or a greedy construction + local search.

    # Let me implement:
    # 1. Compute all 720 permutations of 1..6, assign indices.
    # 2. Precompute edge costs (overlap-based) between all pairs. 720^2 = 518400 edges.
    # 3. Use a greedy nearest-neighbor path, then improve with 2-opt / Or-opt.
    # 4. Build the word from the path.

    # But 720^2 edge precomputation might be slow in Python. Let me optimize.

    # Actually, let me reconsider the time budget. 150s. Precomputing 518400 edges
    # where each edge requires comparing up to 13-length strings... that's maybe
    # 518400 * 13 * a few ops = ~10M ops, should be fine in a few seconds with NumPy.

    # Let me implement this.

    m = n - 1  # number of old symbols
    new_sym = n
    num_perms = math.factorial(m)

    # Generate all permutations of 1..m
    from itertools import permutations
    all_perms = list(permutations(range(1, m + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}

    # Precompute blocks: block[i] = tuple of length 2*m+1
    blocks = []
    for p in all_perms:
        blocks.append(p + (new_sym,) + p)

    # Precompute edge costs: cost[i][j] = len(block[j]) - overlap(block[i], block[j])
    # overlap = longest o such that block[i][-o:] == block[j][:o]
    # Use a 2D array. 720x720 = 518400 entries.
    # For speed, use NumPy or just nested loops with early termination.

    # Let me compute overlaps efficiently. For each pair (i, j), the max possible
    # overlap is 2*m+1 = 13. We check o from 13 down to 0.
    # To speed up, we can use string matching.

    # Convert blocks to strings for fast comparison
    block_strs = ["".join(map(str, b)) for b in blocks]

    # Precompute overlap matrix
    # overlap[i][j] = max o such that block_strs[i][-o:] == block_strs[j][:o]
    # For efficiency, note that overlap[i][j] <= 2*m+1 = 13.

    # Use a list of lists
    overlap = [[0] * num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        s_i = block_strs[i]
        for j in range(num_perms):
            s_j = block_strs[j]
            # Max overlap is min(len(s_i), len(s_j)) = 2*m+1
            max_o = 2 * m + 1
            # Check from max_o down to 1
            for o in range(max_o, 0, -1):
                if s_i[-o:] == s_j[:o]:
                    overlap[i][j] = o
                    break

    # Edge cost: cost[i][j] = (2*m+1) - overlap[i][j]
    # Total path cost = (2*m+1) + sum_{consecutive} cost = (2*m+1) + (num_perms-1)*(2*m+1) - sum(overlaps)
    # = num_perms*(2*m+1) - sum(overlaps along path)
    # So minimizing total length = maximizing sum of overlaps along the path.

    # This is a maximum-weight Hamiltonian path problem.

    # Greedy: start from a random node, repeatedly go to the unvisited neighbor with max overlap.
    # Then improve with 2-opt (reverse segments to increase total overlap).

    # Let me implement greedy + 2-opt.

    def greedy_path(start=None):
        if start is None:
            start = random.randint(0, num_perms - 1)
        visited = [False] * num_perms
        path = [start]
        visited[start] = True
        for _ in range(num_perms - 1):
            last = path[-1]
            best_next = -1
            best_ov = -1
            for j in range(num_perms):
                if not visited[j]:
                    ov = overlap[last][j]
                    if ov > best_ov:
                        best_ov = ov
                        best_next = j
            if best_next == -1:
                break
            path.append(best_next)
            visited[best_next] = True
        return path

    def path_total_overlap(path):
        total = 0
        for i in range(len(path) - 1):
            total += overlap[path[i]][path[i + 1]]
        return total

    def path_length(path):
        return num_perms * (2 * m + 1) - path_total_overlap(path)

    # 2-opt: for a path, try reversing segments [i+1, j] and check if it improves.
    # The change in total overlap when reversing path[i+1..j]:
    #   Old: ... A -> path[i+1] -> ... -> path[j] -> B ...
    #   New: ... A -> path[j] -> ... -> path[i+1] -> B ...
    #   Change = overlap[A][path[j]] + overlap[path[i+1]][B] - overlap[A][path[i+1]] - overlap[path[j]][B]
    #   (internal edges are reversed but their overlaps are the same since overlap is
    #    not symmetric... wait, overlap[i][j] != overlap[j][i] in general)
    #   Actually, when we reverse the segment, the internal edges go from
    #   overlap[path[k]][path[k+1]] to overlap[path[k+1]][path[k]], which are different.
    #   So the 2-opt delta is more complex. Let me just compute the total before and after.

    # For efficiency, I'll do a limited number of 2-opt passes.

    def two_opt(path, max_passes=5):
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            n_path = len(path)
            for i in range(n_path - 1):
                if time.time() > deadline - 20:
                    return path
                for j in range(i + 1, n_path):
                    if time.time() > deadline - 20:
                        return path
                    # Try reversing path[i+1..j]
                    old_total = 0
                    # Compute old contribution of edges affected
                    # Edges affected: (path[i], path[i+1]), (path[j], path[j+1]) if j+1 exists,
                    # and all internal edges in [i+1, j]
                    # This is expensive. Let me just compute total before and after.
                    # For speed, skip 2-opt if too slow.
                    pass
        return path

    # 2-opt is too expensive for 720 nodes. Let me use a simpler local search:
    # Or-opt: move a single node to a better position.

    def or_opt(path, max_passes=3):
        for _ in range(max_passes):
            improved = False
            for i in range(len(path)):
                if time.time() > deadline - 20:
                    return path
                node = path[i]
                # Try inserting node at each position j
                for j in range(len(path)):
                    if j == i or j == i + 1:
                        continue
                    # Remove node from i, insert at j
                    new_path = path[:i] + path[i + 1:]
                    insert_at = j if j < i else j - 1
                    # Actually, let me just do a simple swap-based local search
                    pass
            # Skip or-opt, too complex. Use a different approach.
        return path

    # Let me use a simpler approach: greedy from multiple random starts, keep the best.
    # Then do a few rounds of "edge swap" local search.

    best_path = None
    best_len = float('inf')

    # Run greedy from several random starts
    num_starts = 10
    for s in range(num_starts):
        if time.time() > deadline - 30:
            break
        start = random.randint(0, num_perms - 1)
        path = greedy_path(start)
        plen = path_length(path)
        if plen < best_len:
            best_len = plen
            best_path = path[:]

    # Local search: try swapping consecutive pairs
    def local_search(path, max_iter=500):
        total_ov = path_total_overlap(path)
        for _ in range(max_iter):
            if time.time() > deadline - 20:
                break
            improved = False
            # Try reversing a random segment
            i = random.randint(0, len(path) - 2)
            j = random.randint(i + 1, len(path) - 1)
            new_path = path[:i] + path[i:j + 1][::-1] + path[j + 1:]
            new_ov = path_total_overlap(new_path)
            if new_ov > total_ov:
                path = new_path
                total_ov = new_ov
                improved = True
            if not improved and random.random() < 0.1:
                break
        return path

    if best_path is not None:
        best_path = local_search(best_path)
        best_len = path_length(best_path)

    # Build the word from the best path
    word = []
    for idx in range(len(best_path)):
        p = best_path[idx]
        block = blocks[p]
        if idx == 0:
            word.extend(block)
        else:
            # Find overlap with current word
            block_str = "".join(map(str, block))
            max_o = len(block_str)
            ov = 0
            for o in range(max_o, 0, -1):
                if "".join(map(str, word[-o:])) == block_str[:o]:
                    ov = o
                    break
            word.extend(block[ov:])

    if time.time() > deadline - 5:
        # Return what we have, even if not optimal
        return "".join(map(str, word))

    return "".join(map(str, word))

# EVOLVE-BLOCK-END
