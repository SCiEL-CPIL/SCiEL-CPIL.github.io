import time
import random
from collections import defaultdict
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Precompute all forbidden triples (a,b,c) with a<b<c, 1/a = 1/b + 1/c
    # Equivalent to: bc = a(b+c) => c = ab/(b-a)
    triples = []
    triple_map = defaultdict(set)  # element -> set of triples containing it

    for a in range(1, 181):
        for b in range(a + 1, 181):
            den = b - a
            if den == 0:
                continue
            num = a * b
            if num % den == 0:
                c = num // den
                if c > b and c <= 180:
                    triple = (a, b, c)
                    triples.append(triple)
                    triple_map[a].add(triple)
                    triple_map[b].add(triple)
                    triple_map[c].add(triple)

    # For each element, precompute the pairs that would conflict
    conflict_pairs = defaultdict(list)
    for (a, b, c) in triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))

    # conflicts[x] = set of elements that share a triple with x
    conflicts = defaultdict(set)
    for (a, b, c) in triples:
        conflicts[a].add(b)
        conflicts[a].add(c)
        conflicts[b].add(a)
        conflicts[b].add(c)
        conflicts[c].add(a)
        conflicts[c].add(b)

    def check_conflict(x, S):
        """Check if adding x to set S creates any forbidden triple."""
        for y, z in conflict_pairs[x]:
            if y in S and z in S:
                return True
        return False

    def can_add(x, S):
        """Check if x can be added to S without creating a forbidden triple."""
        return not check_conflict(x, S)

    def greedy_build(order, S_init=None):
        """Build a set greedily following the given order."""
        S = set(S_init) if S_init is not None else set()
        for x in order:
            if x in S:
                continue
            if not check_conflict(x, S):
                S.add(x)
        return S

    best_S = set()
    best_size = 0

    # Strategy 1: Start with large numbers, add small ones
    S1 = set(range(91, 181))
    candidates = list(range(1, 91))
    random.shuffle(candidates)
    for x in candidates:
        if not check_conflict(x, S1):
            S1.add(x)
    if len(S1) > best_size:
        best_size = len(S1)
        best_S = set(S1)

    # Strategy 1b: Start with large numbers, add small ones in descending order (deterministic)
    S1b = set(range(91, 181))
    for x in range(90, 0, -1):
        if not check_conflict(x, S1b):
            S1b.add(x)
    if len(S1b) > best_size:
        best_size = len(S1b)
        best_S = set(S1b)

    # Strategy 2: Pure greedy from all numbers in random order
    all_nums = list(range(1, 181))
    random.shuffle(all_nums)
    S2 = greedy_build(all_nums)
    if len(S2) > best_size:
        best_size = len(S2)
        best_S = set(S2)

    # Strategy 3: Greedy with small numbers first
    S3 = greedy_build(list(range(1, 181)))
    if len(S3) > best_size:
        best_size = len(S3)
        best_S = set(S3)

    # Strategy 4: Greedy with large numbers first
    S4 = greedy_build(list(range(180, 0, -1)))
    if len(S4) > best_size:
        best_size = len(S4)
        best_S = set(S4)

    # Strategy 5: Multiple random restarts
    for _ in range(10):
        if time.time() > deadline - 40:
            break
        nums = list(range(1, 181))
        random.shuffle(nums)
        S5 = greedy_build(nums)
        if len(S5) > best_size:
            best_size = len(S5)
            best_S = set(S5)

    # Strategy 6: Weighted greedy - prioritize elements with fewer conflicts
    if time.time() < deadline - 35:
        degree = {x: len(conflict_pairs[x]) for x in range(1, 181)}
        order = sorted(range(1, 181), key=lambda x: degree[x])
        S6 = greedy_build(order)
        if len(S6) > best_size:
            best_size = len(S6)
            best_S = set(S6)

    # Local search on best_S
    S = set(best_S)

    # Phase 1: Try to add any remaining elements
    remaining = [x for x in range(1, 181) if x not in S]
    for x in remaining:
        if not check_conflict(x, S):
            S.add(x)
    best_size = len(S)
    best_S = set(S)

    # Phase 2: Local search - remove one, add multiple
    while time.time() < deadline - 15:
        improved = False
        S_list = list(S)
        random.shuffle(S_list)

        for remove_elem in S_list:
            if time.time() > deadline - 15:
                break
            S.remove(remove_elem)

            added = []
            for x in remaining:
                if x in S:
                    continue
                if not check_conflict(x, S):
                    S.add(x)
                    added.append(x)

            if len(added) >= 1 and len(S) > len(best_S):
                best_size = len(S)
                best_S = set(S)
                improved = True
                remaining = [x for x in range(1, 181) if x not in S]
            else:
                for x in added:
                    S.discard(x)
                S.add(remove_elem)

        if not improved:
            break

    # Phase 3: Pair removal - remove two, try to add three
    S = set(best_S)
    remaining = [x for x in range(1, 181) if x not in S]

    while time.time() < deadline - 8:
        improved = False
        S_list = list(S)
        random.shuffle(S_list)

        for i in range(len(S_list)):
            if time.time() > deadline - 8:
                break
            for j in range(i + 1, len(S_list)):
                if time.time() > deadline - 8:
                    break
                r1, r2 = S_list[i], S_list[j]
                S.remove(r1)
                S.remove(r2)

                added = []
                for x in remaining:
                    if x in S:
                        continue
                    if not check_conflict(x, S):
                        S.add(x)
                        added.append(x)

                if len(added) >= 3 and len(S) > len(best_S):
                    best_size = len(S)
                    best_S = set(S)
                    improved = True
                    remaining = [x for x in range(1, 181) if x not in S]
                else:
                    for x in added:
                        S.discard(x)
                    S.add(r1)
                    S.add(r2)

        if not improved:
            break

    # Phase 4: Branch and Bound with unit propagation (3-SAT style)
    # Variables: x_i for i in 1..180 (1 = include in set)
    # Clauses: for each triple (a,b,c): (¬x_a ∨ ¬x_b ∨ ¬x_c)
    # Maximize sum(x_i)
    if time.time() < deadline - 5:
        # Precompute for each variable, the list of clauses it appears in
        var_to_clauses = defaultdict(list)
        for idx, (a, b, c) in enumerate(triples):
            var_to_clauses[a].append(idx)
            var_to_clauses[b].append(idx)
            var_to_clauses[c].append(idx)

        # State: assignment dict var -> 0 or 1, -1 for unassigned
        # We'll use a more efficient approach: maintain sets of assigned vars
        # and track which clauses are satisfied/violated

        # For BnB: we track included set and candidate set
        # A variable is "forced out" if two of its triples already have their other two vars included

        def is_forced_out(x, included_set):
            """Check if x must be excluded because some triple has the other two in included_set."""
            for y, z in conflict_pairs[x]:
                if y in included_set and z in included_set:
                    return True
            return False

        def get_candidates(included_set, all_vars):
            """Get all vars that can still be included."""
            return [v for v in all_vars if v not in included_set and not is_forced_out(v, included_set)]

        def upper_bound(included_set, candidates):
            """Upper bound on final set size.
            Use: |included| + |candidates| - (number of forced exclusions among candidates)
            Better: count how many triples have exactly 1 var in included and 2 in candidates;
            each such triple can contribute at most 1 more from candidates.
            Simple bound: |included| + |candidates|
            Improved: for each candidate x, if it's in k active triples (with 1 in included, 1 in candidates),
            at most k-1 of those candidates can be added... complex.
            Use a greedy coloring bound on the conflict graph restricted to candidates.
            """
            if not candidates:
                return len(included_set)

            # Greedy coloring: assign colors so that no triple is monochromatic
            # This gives an upper bound via sum over colors of min(color_class_size, 2)
            # Actually for 3-uniform hypergraph, a proper coloring means no triple is monochromatic,
            # so each color class is an independent set. Upper bound = sum of color class sizes.
            # But that's just |candidates|. We need: max independent set <= max color class size * something.
            # Simpler: use the fact that each triple among candidates can block at most 1 element.
            # Count triples entirely within candidates; each can block 1, so upper bound =
            # |candidates| - (number of disjoint triples among candidates) ... not quite.
            # Use: |included| + |candidates| as trivial bound, with a simple improvement.
            # For each candidate, count how many "conflict partners" are also candidates.
            # If a candidate has 2+ partners in candidates that form a triple, it can't all be included.

            # Practical bound: |included| + |candidates|
            # With a small discount for dense conflict regions
            return len(included_set) + len(candidates)

        bnb_best_size = best_size
        bnb_best_set = set(best_S)
        nodes_visited = 0
        max_nodes = 1000000

        def branch_and_bound_3sat(candidates, included_set, current_size, depth=0):
            """BnB with unit propagation for 3-SAT MaxSAT."""
            nonlocal bnb_best_size, bnb_best_set, nodes_visited

            if time.time() > deadline - 3:
                return
            if nodes_visited >= max_nodes:
                return

            nodes_visited += 1

            # Update best
            if current_size > bnb_best_size:
                bnb_best_size = current_size
                bnb_best_set = set(included_set)

            if not candidates:
                return

            # Upper bound
            ub = upper_bound(included_set, candidates)
            if ub <= bnb_best_size:
                return

            # Unit propagation: check if any candidate is forced out
            # (already handled in get_candidates, but re-check after additions)

            # Choose pivot: element with most conflicts in candidates
            pivot = None
            max_conf = -1
            for x in candidates:
                conf_count = len(conflicts[x] & set(candidates))
                if conf_count > max_conf:
                    max_conf = conf_count
                    pivot = x

            if pivot is None:
                return

            pivot_set = set(candidates)

            # Branch 1: Include pivot
            new_included = set(included_set)
            new_included.add(pivot)
            # Update candidates: remove pivot, remove any that are now forced out
            new_candidates = []
            for x in candidates:
                if x == pivot:
                    continue
                if not is_forced_out(x, new_included):
                    new_candidates.append(x)

            branch_and_bound_3sat(new_candidates, new_included, current_size + 1, depth + 1)

            # Branch 2: Exclude pivot
            # When we exclude pivot, some triples may become "satisfied" (can't be violated),
            # which might free up other variables. But for simplicity, just remove pivot.
            remaining_candidates = [x for x in candidates if x != pivot]
            branch_and_bound_3sat(remaining_candidates, included_set, current_size, depth + 1)

        # Initialize: start with empty included, all vars as candidates
        # But use the greedy solution to guide: pre-include some elements
        # Better: start fresh but with good variable ordering

        # Variable ordering: by degree (fewer conflicts first for inclusion branch)
        degree = {x: len(conflicts[x]) for x in range(1, 181)}
        initial_candidates = sorted(range(1, 181), key=lambda x: degree[x])

        branch_and_bound_3sat(initial_candidates, set(), 0)

        if bnb_best_size > best_size:
            best_size = bnb_best_size
            best_S = set(bnb_best_set)

    # Final verification: ensure no forbidden triples
    def verify_set(S):
        for (a, b, c) in triples:
            if a in S and b in S and c in S:
                return False
        return True

    if not verify_set(best_S):
        # Fallback to a known valid set
        result = list(range(91, 181))
        if verify_set(set(result)):
            return result
        else:
            return [180]

    result = list(best_S)
    if not result:
        result = list(range(91, 181))

    return result

