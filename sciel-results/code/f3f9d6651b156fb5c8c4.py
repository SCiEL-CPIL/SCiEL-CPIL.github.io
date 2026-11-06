import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill

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
    # conflict_pairs[x] = list of (y, z) such that {x,y,z} is a forbidden triple
    conflict_pairs = defaultdict(list)
    for (a, b, c) in triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))

    # Build adjacency: for each element, which other elements appear in its conflict triples
    adj = defaultdict(set)
    for (a, b, c) in triples:
        adj[a].update([b, c])
        adj[b].update([a, c])
        adj[c].update([a, b])

    # For each pair (y,z), which elements x would conflict with both y and z?
    # i.e., x such that {x,y,z} is a forbidden triple
    pair_to_elements = defaultdict(set)
    for (a, b, c) in triples:
        pair_to_elements[(a, b)].add(c)
        pair_to_elements[(a, c)].add(b)
        pair_to_elements[(b, c)].add(a)
        # Also add sorted versions for consistent lookup
        if a > b:
            pair_to_elements[(b, a)].add(c)
        if a > c:
            pair_to_elements[(c, a)].add(b)
        if b > c:
            pair_to_elements[(c, b)].add(a)

    def check_conflict(x, S):
        """Check if adding x to set S creates any forbidden triple."""
        for y, z in conflict_pairs[x]:
            if y in S and z in S:
                return True
        return False

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
    for _ in range(8):
        if time.time() > deadline - 35:
            break
        nums = list(range(1, 181))
        random.shuffle(nums)
        S5 = greedy_build(nums)
        if len(S5) > best_size:
            best_size = len(S5)
            best_S = set(S5)

    # Strategy 6: Weighted greedy - prioritize elements with fewer conflicts
    if time.time() < deadline - 30:
        # Count how many triples each element participates in
        degree = {x: len(conflict_pairs[x]) for x in range(1, 181)}
        # Sort by degree ascending (fewer conflicts first)
        order = sorted(range(1, 181), key=lambda x: degree[x])
        S6 = greedy_build(order)
        if len(S6) > best_size:
            best_size = len(S6)
            best_S = set(S6)

    # Strategy 7: Structure-aware greedy
    # Elements that don't appear in any triple can always be included
    # Elements that appear in few triples are "safer"
    if time.time() < deadline - 25:
        # Find elements with zero conflicts
        zero_conflict = [x for x in range(1, 181) if len(conflict_pairs[x]) == 0]
        # Find elements with minimal conflicts
        low_conflict = sorted(range(1, 181), key=lambda x: (len(conflict_pairs[x]), x))
        
        S7 = set(zero_conflict)
        for x in low_conflict:
            if x not in S7 and not check_conflict(x, S7):
                S7.add(x)
        if len(S7) > best_size:
            best_size = len(S7)
            best_S = set(S7)

    # Local search on best_S
    S = set(best_S)

    # Phase 1: Try to add any remaining elements
    remaining = [x for x in range(1, 181) if x not in S]
    for x in remaining:
        if not check_conflict(x, S):
            S.add(x)
    best_size = len(S)
    best_S = set(S)

    # Phase 2: Enhanced local search - remove one, add multiple
    # Also try: for each element in S, see if removing it allows adding more
    while time.time() < deadline - 10:
        improved = False
        S_list = list(S)
        random.shuffle(S_list)

        for remove_elem in S_list:
            if time.time() > deadline - 10:
                break
            S.remove(remove_elem)

            # Try to add elements, prioritizing those with fewer conflicts
            added = []
            for x in sorted(remaining, key=lambda x: len(conflict_pairs[x])):
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
                # Revert
                for x in added:
                    S.discard(x)
                S.add(remove_elem)

        if not improved:
            break

    # Phase 3: Pair removal - remove two, try to add three
    S = set(best_S)
    remaining = [x for x in range(1, 181) if x not in S]

    while time.time() < deadline - 5:
        improved = False
        S_list = list(S)
        random.shuffle(S_list)

        for i in range(len(S_list)):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, len(S_list)):
                if time.time() > deadline - 5:
                    break
                r1, r2 = S_list[i], S_list[j]
                S.remove(r1)
                S.remove(r2)

                added = []
                for x in sorted(remaining, key=lambda x: len(conflict_pairs[x])):
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

    # Phase 4: Simulated annealing-style refinement with better moves
    S = set(best_S)
    remaining = [x for x in range(1, 181) if x not in S]
    current_size = len(S)
    best_SA_size = current_size
    best_SA_S = set(S)

    temp = 5.0
    min_temp = 0.1
    cooling = 0.98

    while temp > min_temp and time.time() < deadline - 3:
        S_list = list(S)
        random.shuffle(S_list)
        remaining_list = list(remaining)
        random.shuffle(remaining_list)

        for _ in range(min(50, len(S_list))):
            if time.time() > deadline - 3:
                break
            # Random move: remove one from S, add one from remaining
            if not S_list or not remaining_list:
                break
            rem_idx = random.randint(0, len(S_list) - 1)
            add_idx = random.randint(0, len(remaining_list) - 1)
            r = S_list[rem_idx]
            a = remaining_list[add_idx]

            if r in S and a not in S:
                S.remove(r)
                remaining_list.remove(a)
                S_list.remove(r)
                remaining_list.append(r)

                # Try to add more, prioritizing low-conflict elements
                added_extra = []
                for x in sorted(remaining_list[:], key=lambda x: len(conflict_pairs[x])):
                    if x not in S and not check_conflict(x, S):
                        S.add(x)
                        added_extra.append(x)

                new_size = len(S)
                delta = new_size - current_size

                if delta >= 0 or random.random() < pow(2.718281828, delta / temp):
                    current_size = new_size
                    if current_size > best_SA_size:
                        best_SA_size = current_size
                        best_SA_S = set(S)
                        remaining = [x for x in range(1, 181) if x not in S]
                        remaining_list = list(remaining)
                        S_list = list(S)
                else:
                    # Revert
                    S.add(r)
                    for x in added_extra:
                        S.discard(x)
                    S_list.append(r)
                    remaining_list.remove(r)
                    remaining_list.append(a)

        temp *= cooling

    if best_SA_size > best_size:
        best_size = best_SA_size
        best_S = best_SA_S

    # Phase 5: Final polish - try all single additions
    S = set(best_S)
    for x in range(1, 181):
        if x not in S and not check_conflict(x, S):
            S.add(x)
    if len(S) > best_size:
        best_size = len(S)
        best_S = S

    # Final verification
    result = list(best_S)
    if not result:
        result = list(range(91, 181))

    return result
