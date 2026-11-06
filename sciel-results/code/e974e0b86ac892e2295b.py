import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)  # 560

    # Build the forbidden transversals as lists of triple indices.
    transversals = []
    seen = set()
    for six_pts in itertools.combinations(points, 6):
        sp = list(six_pts)
        matchings = set()
        for perm in itertools.permutations(range(6)):
            p1 = tuple(sorted((sp[perm[0]], sp[perm[1]])))
            p2 = tuple(sorted((sp[perm[2]], sp[perm[3]])))
            p3 = tuple(sorted((sp[perm[4]], sp[perm[5]])))
            matchings.add(tuple(sorted((p1, p2, p3))))
        for m in matchings:
            (p1, p2, p3) = m
            a, b = p1
            c, d = p2
            e, f = p3
            triple_indices = []
            for (x, y, z) in ((a, c, e), (a, c, f), (a, d, e), (a, d, f),
                              (b, c, e), (b, c, f), (b, d, e), (b, d, f)):
                triple_indices.append(triple_to_idx[tuple(sorted((x, y, z)))])
            key = tuple(sorted(triple_indices))
            if key not in seen:
                seen.add(key)
                transversals.append(triple_indices)
    num_trans = len(transversals)

    # For each triple, the list of transversals it participates in.
    triple_trans = [[] for _ in range(num_triples)]
    for ti, tlist in enumerate(transversals):
        for idx in tlist:
            triple_trans[idx].append(ti)

    # Precompute neighbors: triples that share at least one transversal.
    # neighbors[i] is a set of triple indices j != i such that they share a transversal.
    neighbors = [set() for _ in range(num_triples)]
    for ti, tlist in enumerate(transversals):
        for i in range(len(tlist)):
            for j in range(i + 1, len(tlist)):
                a = tlist[i]
                b = tlist[j]
                neighbors[a].add(b)
                neighbors[b].add(a)
    
    # Convert sets to lists for faster iteration? Or keep as set for deduplication in candidate list?
    # We will use a list for candidate_list and check if sol[c] is False before processing.
    # So duplicates in candidate_list are okay, we just skip them if already selected.
    neighbor_lists = [list(s) for s in neighbors]

    def is_safe(idx, counts, sol):
        if sol[idx]:
            return False
        for ti in triple_trans[idx]:
            if counts[ti] == 7:
                return False
        return True

    def greedy_init():
        sol = [False] * num_triples
        counts = [0] * num_trans
        candidates = list(range(num_triples))
        random.shuffle(candidates)
        
        for idx in candidates:
            if is_safe(idx, counts, sol):
                sol[idx] = True
                for ti in triple_trans[idx]:
                    counts[ti] += 1
        return sol, counts, sum(sol)

    def local_search(sol, counts, current_count):
        # candidate_list contains unselected triples that might be safe
        # Initialize with all unselected
        candidate_list = [i for i in range(num_triples) if not sol[i]]
        random.shuffle(candidate_list)
        
        while time.time() < deadline - 0.5:
            added = 0
            new_candidate_list = []
            
            # Process current candidate list
            for idx in candidate_list:
                if sol[idx]:
                    continue
                if is_safe(idx, counts, sol):
                    sol[idx] = True
                    for ti in triple_trans[idx]:
                        counts[ti] += 1
                    added += 1
                else:
                    # Keep it? No, discard. It will be re-added if a neighbor is removed.
                    pass 

            current_count += added
            
            if added > 0:
                # Update best? Handled in outer loop or here?
                # Let's return the new count and let outer loop handle best tracking.
                # But we need to continue the search if we added things.
                # The candidate_list is now empty (we processed all).
                # We need to find new candidates?
                # No, adding triples only blocks others. It doesn't unblock.
                # So if we added some, we should check if there are *other* candidates?
                # We discarded the unsafe ones.
                # Are there any unselected triples NOT in candidate_list?
                # Initially, candidate_list had all unselected.
                # We discarded unsafe ones.
                # So any unselected triple is either in sol (True) or was discarded.
                # If we added some, the counts changed.
                # The discarded ones might now be safe?
                # NO. Adding triples increases counts.
                # If a triple was unsafe (count 7), it is still unsafe (count >= 7).
                # So we don't need to re-check discarded ones.
                # So if we added > 0, we have reached a local optimum (no more safe candidates).
                # We must kick.
                pass
            else:
                # No added, so we are at a local optimum.
                pass
            
            # Need to kick to escape local optimum
            # Find a tight transversal (count 7)
            tight_trans = []
            for ti, cnt in enumerate(counts):
                if cnt == 7:
                    tight_trans.append(ti)
            
            if not tight_trans:
                # No tight transversals? Then we can add more?
                # But we checked all candidates.
                # This implies we are at a global optimum or stuck?
                # If no count is 7, then all candidates should have been safe?
                # Unless is_safe failed for another reason?
                # is_safe only fails if count == 7.
                # So if no count is 7, all unselected should be safe.
                # But we added 0.
                # This means candidate_list was empty or all were already selected.
                # If candidate_list was empty, we are done.
                break

            # Pick a random tight transversal
            ti = random.choice(tight_trans)
            # Pick a random triple from this transversal that is selected
            selected_in_trans = [idx for idx in transversals[ti] if sol[idx]]
            if not selected_in_trans:
                continue # Should not happen
            kick_idx = random.choice(selected_in_trans)
            
            # Remove kick_idx
            sol[kick_idx] = False
            for t in triple_trans[kick_idx]:
                counts[t] -= 1
            current_count -= 1
            
            # Add neighbors of kick_idx to candidate_list
            # Only those that are not selected
            for n in neighbor_lists[kick_idx]:
                if not sol[n]:
                    candidate_list.append(n)
            
            random.shuffle(candidate_list)

        return current_count, sol, counts

    best_solution = None
    best_counts = None
    best_count = -1

    # Run multiple restarts
    num_restarts = 100 # Will be limited by time
    for restart in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
        
        sol, counts, current_count = greedy_init()
        
        # Run local search
        current_count, sol, counts = local_search(sol, counts, current_count)
        
        if current_count > best_count:
            best_count = current_count
            best_solution = sol[:]
            best_counts = counts[:]

    if best_solution is None:
        best_solution, best_counts, _ = greedy_init()

    # Final validation
    final_sol = best_solution[:]
    final_counts = best_counts[:]
    for ci in range(num_trans):
        while final_counts[ci] == 8:
            for t_idx in transversals[ci]:
                if final_sol[t_idx]:
                    final_sol[t_idx] = False
                    for ci2 in triple_trans[t_idx]:
                        final_counts[ci2] -= 1
                    break

    result = [list(all_triples[i]) for i in range(num_triples) if final_sol[i]]
    return result
