import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    # ---------- subset-sum maintenance ----------
    def subset_sums(S):
        sums = set()
        for x in S:
            new = {x}
            for s in sums:
                new.add(x + s)
            sums |= new
        return sums

    # ---------- divisibility check (full, for validation) ----------
    def is_divisibility_free(sums):
        sl = sorted(sums)
        k = len(sl)
        for i in range(k):
            a = sl[i]
            if a <= 1:
                continue
            for j in range(i + 1, k):
                b = sl[j]
                if b >= 2 * a and b % a == 0:
                    return False
        return True

    # ---------- incremental add check (fast, incremental) ----------
    def try_add(current_sums, x):
        new_sums = {x}
        for s in current_sums:
            new_sums.add(x + s)
        all_sums = current_sums | new_sums
        # duplicate sums?
        if len(all_sums) != len(current_sums) + len(new_sums):
            return False, None
        # divisibility: only need to check pairs where at least one is new
        # new vs old
        for sn in new_sums:
            for so in current_sums:
                if sn > so:
                    if sn % so == 0:
                        return False, None
                elif so > sn:
                    if so % sn == 0:
                        return False, None
        # new vs new
        nl = list(new_sums)
        for i in range(len(nl)):
            a = nl[i]
            for j in range(i + 1, len(nl)):
                b = nl[j]
                if a > b:
                    if a % b == 0:
                        return False, None
                else:
                    if b % a == 0:
                        return False, None
        return True, all_sums

    # ---------- candidate pool: large numbers reduce divisibility ----------
    all_candidates = list(range(128, 257))  # 128 numbers
    n = len(all_candidates)
    val = [x for x in all_candidates]
    idx = {x: i for i, x in enumerate(val)}

    # ---------- precompute directed compatibility ----------
    # comp[i] = bitset of j such that x_j can be added AFTER x_i (i.e., {x_i,x_j} valid with x_i first)
    # We build a directed graph; edge i->j means adding x_j to a set containing x_i is valid.
    # For clique in this directed graph: a clique is a set where there exists an ordering
    # such that each element is compatible with all previous. We use the directed graph where
    # edge i->j means x_j can be added after x_i.
    # Actually for the clique approach: we need a set S such that there exists an ordering
    # x_{p1}, x_{p2}, ... where each x_{pk} is compatible with {x_{p1},...,x_{p_{k-1}}}.
    # This is equivalent to: in the directed graph where edge i->j means "j can be added after i",
    # we need a directed clique (tournament-like). But our compatibility is not symmetric.
    #
    # Better approach: use the undirected "pairwise coexist" graph where edge i-j means
    # {x_i, x_j} is a valid set (both orderings work, or at least one). Then a clique in this
    # graph is a set where every pair coexists, which is necessary but not sufficient for
    # the full set to be valid. We'll use it as a heuristic for ordering in B&B.
    #
    # For the actual search, I'll use a backtracking that builds the set incrementally
    # with the try_add check, guided by the compatibility graph for ordering.

    # Precompute: for each pair (i,j), can x_j be added to a set {x_i}?
    # pair_ok[i][j] = True if {x_i, x_j} is valid
    pair_ok = [[False] * n for _ in range(n)]
    for i in range(n):
        sums_i = {val[i]}
        for j in range(n):
            if i == j:
                continue
            ok, _ = try_add(sums_i, val[j])
            pair_ok[i][j] = ok
            if ok:
                pair_ok[j][i] = True  # symmetric for the pair

    # Degree for ordering
    degree = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and pair_ok[i][j]:
                degree[i] += 1

    # ---------- Tomita-style maximum clique on the pairwise compatibility graph ----------
    # This gives a lower bound / good initial solution. Then we verify and extend.
    # Vertices: 0..n-1, edge i-j if pair_ok[i][j]
    # Use bitsets for speed.

    # adjacency bitsets
    adj = [0] * n
    for i in range(n):
        bits = 0
        for j in range(n):
            if i != j and pair_ok[i][j]:
                bits |= (1 << j)
        adj[i] = bits

    best_clique = []
    best_clique_size = 0
    global_deadline = deadline

    def max_clique_branch(candidates_bits, current_clique, depth):
        nonlocal best_clique, best_clique_size
        if time.time() > global_deadline - 3.0:
            return
        if candidates_bits == 0:
            if len(current_clique) > best_clique_size:
                best_clique_size = len(current_clique)
                best_clique = list(current_clique)
            return
        # Upper bound: current size + popcount(candidates)
        bound = len(current_clique) + bin(candidates_bits).count('1')
        if bound <= best_clique_size:
            return
        # Order vertices by degree within candidates (descending) for better pruning
        # Extract vertices from bitset
        verts = []
        cb = candidates_bits
        while cb:
            lsb = cb & (-cb)
            v = lsb.bit_length() - 1
            verts.append(v)
            cb ^= lsb
        # Sort by degree in candidates (descending)
        verts.sort(key=lambda v: bin(adj[v] & candidates_bits).count('1'), reverse=True)

        for v in verts:
            if time.time() > global_deadline - 3.0:
                return
            if not (candidates_bits & (1 << v)):
                continue
            # Branch: include v
            new_cand = candidates_bits & adj[v]
            # Remove v and all subsequent (already processed) vertices
            # In Tomita, we remove v from candidates and recurse on candidates & adj[v]
            new_cand = new_cand & ~((1 << (v + 1)) - 1)  # remove v and lower (already tried)
            # Actually standard Tomita: new_cand = candidates & adj[v], then remove v
            new_cand = (candidates_bits & adj[v]) & ~(1 << v)
            current_clique.append(v)
            max_clique_branch(new_cand, current_clique, depth + 1)
            current_clique.pop()
            # Remove v from candidates for subsequent branches
            candidates_bits &= ~(1 << v)
            # Update bound
            bound = len(current_clique) + bin(candidates_bits).count('1')
            if bound <= best_clique_size:
                return

    # Run max clique search
    max_clique_branch((1 << n) - 1, [], 0)

    # Convert clique to set and verify
    if best_clique:
        clique_set = [val[v] for v in best_clique]
        clique_sums = subset_sums(clique_set)
        if is_divisibility_free(clique_sums):
            best_size = len(clique_set)
            best = list(clique_set)
        else:
            # Clique in pairwise graph isn't necessarily a valid full set
            # Fall back to greedy on the clique vertices
            best_size = 0
            best = []
    else:
        best_size = 0
        best = []

    # ---------- Greedy extension / refinement ----------
    # Use the clique as a starting point, or find a good starting set via greedy
    def greedy_extend(start_set, start_sums):
        """Try to extend start_set with remaining candidates."""
        S = list(start_set)
        current_sums = set(start_sums)
        used = set(S)
        remaining = [x for x in all_candidates if x not in used]
        # Order remaining by degree (descending) for better results
        remaining.sort(key=lambda x: degree[idx[x]], reverse=True)
        for x in remaining:
            if time.time() > deadline - 2.0:
                break
            ok, new_sums = try_add(current_sums, x)
            if ok:
                S.append(x)
                current_sums = new_sums
        return S

    # If clique gave us a valid set, extend it
    if best_size > 0:
        ext = greedy_extend(best, subset_sums(best))
        if len(ext) > best_size:
            sums = subset_sums(ext)
            if is_divisibility_free(sums):
                best = ext
                best_size = len(ext)

    # ---------- Randomized greedy with multiple restarts ----------
    random.seed(42)
    for trial in range(3000):
        if time.time() > deadline - 2.0:
            break
        S = []
        current_sums = set()
        # Random order with slight bias toward high-degree candidates
        candidates = list(all_candidates)
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 2.0:
                break
            ok, new_sums = try_add(current_sums, x)
            if ok:
                S.append(x)
                current_sums = new_sums
        if len(S) > best_size:
            sums = subset_sums(S)
            if is_divisibility_free(sums):
                best = S
                best_size = len(S)

    # ---------- Local search: try swaps to improve ----------
    if time.time() < deadline - 5.0 and best_size > 0:
        best_copy = list(best)
        for _ in range(200):
            if time.time() > deadline - 3.0:
                break
            improved = False
            for i in range(len(best_copy)):
                if time.time() > deadline - 3.0:
                    break
                # Try removing best_copy[i] and adding a different candidate
                trial_set = best_copy[:i] + best_copy[i + 1:]
                trial_sums = subset_sums(trial_set)
                used = set(trial_set)
                # Find best replacement
                for x in all_candidates:
                    if x in used:
                        continue
                    ok, _ = try_add(trial_sums, x)
                    if ok:
                        new_set = trial_set + [x]
                        new_sums = subset_sums(new_set)
                        if is_divisibility_free(new_sums):
                            # Same size, but might allow further extension
                            best_copy = new_set
                            improved = True
                            break
                if improved:
                    break
            # Try to extend the current best
            ext = greedy_extend(best_copy, subset_sums(best_copy))
            if len(ext) > best_size:
                sums = subset_sums(ext)
                if is_divisibility_free(sums):
                    best = ext
                    best_size = len(ext)
                    best_copy = ext

    # ---------- Final validation ----------
    if best_size >= 1:
        final_sums = subset_sums(best)
        if is_divisibility_free(final_sums):
            return [int(x) for x in best]

    # Fallback
    fallback = [127, 129, 131]
    fs = subset_sums(fallback)
    if is_divisibility_free(fs):
        return [int(x) for x in fallback]
    return [127, 129]
