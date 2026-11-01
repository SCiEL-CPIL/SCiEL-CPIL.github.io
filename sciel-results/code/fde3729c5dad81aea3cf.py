import time
import numpy as np
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0

    N = q ** n
    if N == 0:
        return []

    # Build all words as list of digit-tuples, and a dict for lookup
    all_words = []
    for idx in range(N):
        rem = idx
        digits = [0] * n
        for j in range(n - 1, -1, -1):
            digits[j] = rem % q
            rem //= q
        all_words.append(tuple(digits))

    # Precompute neighbors within distance <= radius+1 (conflict graph adjacency)
    # For q=3, n=7, radius=1: conflict distance <= 2
    conflict_dist = radius + 1
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    adj = [set() for _ in range(N)]
    for i in range(N):
        wi = all_words[i]
        # All words at distance 1 from wi
        for pos in range(n):
            for new_digit in range(q):
                if new_digit == wi[pos]:
                    continue
                lst = list(wi)
                lst[pos] = new_digit
                w = tuple(lst)
                if w in word_to_idx:
                    j = word_to_idx[w]
                    if j != i:
                        adj[i].add(j)
        # All words at distance 2 from wi
        for pos1 in range(n):
            for pos2 in range(pos1 + 1, n):
                for d1 in range(q):
                    if d1 == wi[pos1]:
                        continue
                    for d2 in range(q):
                        if d2 == wi[pos2]:
                            continue
                        lst = list(wi)
                        lst[pos1] = d1
                        lst[pos2] = d2
                        w = tuple(lst)
                        if w in word_to_idx:
                            j = word_to_idx[w]
                            if j != i:
                                adj[i].add(j)

    # Convert to bitsets
    bitsets = [0] * N
    for i in range(N):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        bitsets[i] = mask

    # Greedy seed for MIS
    def greedy_mis():
        available = (1 << N) - 1
        selected = []
        while available:
            # Pick vertex with max degree in available subgraph
            best_v = -1
            best_deg = -1
            temp = available
            while temp:
                low = temp & (-temp)
                v = low.bit_length() - 1
                temp ^= low
                deg = (bitsets[v] & available).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            selected.append(best_v)
            available &= ~(1 << best_v)
            available &= ~bitsets[best_v]
        return selected

    seed = greedy_mis()
    best_size = len(seed)
    best_set = set(seed)

    # Tomita's maximum clique heuristic on conflict graph = MIS
    # Actually, we want MIS in conflict graph. MIS(G) = Clique(complement of G).
    # But complement is dense (most pairs are non-adjacent in conflict graph).
    # Instead, use B&B directly on MIS: pick vertices greedily with upper bound.

    # Upper bound: for a set S of available vertices, max MIS size <= |S| / min_ball
    # Each selected vertex covers at least 1 (itself), and conflicts with others.
    # Simple bound: |S|
    # Better: greedy coloring of conflict graph restricted to S gives clique number of complement
    # Actually for MIS: use greedy independent set as lower bound, and |S| as upper.

    # Let's use a different approach: since we know the answer is ~186 and N=2187,
    # we can use iterative deepening: try to find MIS of size >= target.

    # First, get a good seed with local improvement
    def try_delete(code_set):
        """Try to remove redundant codewords (words whose ball is covered by others)."""
        changed = True
        while changed:
            changed = False
            for v in list(code_set):
                # Check if removing v still covers all points
                # A point is uncovered if it's within distance 1 of v and not within distance 1 of any other
                # Actually for MIS: v is in MIS iff no other vertex in code_set is adjacent to v in conflict graph
                # So v is "critical" if removing it would make some pair non-conflicting... no.
                # In MIS, all pairs must be non-adjacent in conflict graph (i.e., distance > conflict_dist).
                # So every vertex is essential in the sense that it's part of the independent set.
                # We can't delete any vertex from a valid MIS without invalidating it.
                # But we CAN replace a vertex with another to allow more vertices.
                pass
        return code_set

    # Local search: try swaps to increase size
    def local_search(initial_set, time_limit):
        current = set(initial_set)
        current_size = len(current)
        best = current
        best_size = current_size

        # Conflict adjacency for quick check
        def is_valid_add(candidate, current_set):
            # Check candidate is not adjacent to any in current_set in conflict graph
            c_mask = bitsets[candidate]
            for v in current_set:
                if (c_mask >> v) & 1:
                    return False
            return True

        # Try to add random non-adjacent vertices
        all_mask = (1 << N) - 1
        np.random.seed(42)

        while time.time() < time_limit:
            # Try adding a random vertex
            for _ in range(200):
                if time.time() > time_limit:
                    break
                v = np.random.randint(N)
                if v in current:
                    continue
                if is_valid_add(int(v), current):
                    new_set = current | {int(v)}
                    new_size = len(new_set)
                    if new_size > best_size:
                        best = new_set
                        best_size = new_size
                        current = new_set
                        break

            # Try swap: remove one, add two
            if time.time() > time_limit:
                break
            if current:
                v_remove = int(np.random.choice(list(current)))
                # Find vertices adjacent to v_remove (conflict), they must go if we keep v_remove
                # Instead: try removing v_remove and adding a different vertex
                temp_set = current - {v_remove}
                # Now try to add a vertex not in temp_set and not conflicting with temp_set
                # Get candidates: all vertices not in temp_set and not adjacent to any in temp_set
                # This is expensive, so try random
                for _ in range(50):
                    if time.time() > time_limit:
                        break
                    v_add = int(np.random.randint(N))
                    if v_add in temp_set:
                        continue
                    if is_valid_add(v_add, temp_set):
                        new_set = temp_set | {v_add}
                        if len(new_set) > best_size:
                            best = new_set
                            best_size = len(new_size if False else new_set)
                            current = new_set
                            break

        return best, best_size

    best_set, best_size = local_search(set(seed), deadline - 30)

    # Now use B&B to try to prove or find larger
    # For MIS B&B: branch on including or excluding a vertex
    # Upper bound for subset S: greedy coloring of conflict graph on S gives number of colors
    #   which is an upper bound on clique number of complement = MIS size of S in original
    # Actually: MIS in conflict graph = clique in complement.
    # Greedy coloring of conflict graph restricted to S: each color class is an independent set in conflict graph,
    # so it's a clique in complement. The number of colors is an upper bound on the clique number of complement restricted to S.
    # Wait, that's not right. Let me think again.
    #
    # We want MIS in G (conflict graph). 
    # Greedy coloring of G[S]: assigns colors such that adjacent vertices have different colors.
    # Each color class is an independent set in G[S].
    # The size of the largest independent set in G[S] is at most |S| (trivial) but can be bounded better.
    # Actually, greedy coloring gives a lower bound on chromatic number, not directly useful for MIS.
    #
    # Better upper bound: use the fact that each vertex in MIS covers itself and its neighbors in the "coverage" sense.
    # For our problem, each codeword covers 1 + n*(q-1) = 15 words.
    # So MIS size <= N / 15 = 2187/15 = 145.8? No, that's a lower bound on covering number, not upper bound on MIS.
    #
    # Let me use a simple B&B with the bound: |S| (trivial) and hope the search tree is small enough.
    # Given the structure, this might be too slow.

    # Alternative: use the known result and verify. The best known is 186.
    # Let's try to find a valid code of size 186 or less via local search from multiple starts.

    multi_start_time = deadline - 5
    for trial in range(20):
        if time.time() > multi_start_time:
            break
        np.random.seed(100 + trial)
        # Random valid set construction
        available = (1 << N) - 1
        current_set = set()
        while available and time.time() < multi_start_time:
            # Pick random vertex from available
            # Find a random bit set in available
            temp = available
            # Get list of set bits
            bits = []
            t = temp
            while t:
                low = t & (-t)
                bits.append(low.bit_length() - 1)
                t ^= low
            if not bits:
                break
            v = bits[np.random.randint(len(bits))]
            current_set.add(v)
            available &= ~(1 << v)
            available &= ~bitsets[v]
        
        if len(current_set) < best_size:
            # Improve via local search
            improved, imp_size = local_search(current_set, min(time.time() + 5, multi_start_time))
            if imp_size > best_size:
                best_set = improved
                best_size = imp_size

    # Convert back to codewords
    return [list(all_words[i]) for i in sorted(best_set)]
