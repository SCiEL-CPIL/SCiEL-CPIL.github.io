import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def get_used_pairs(sys):
        used = set()
        for a, b, c in sys:
            used.add((a, b) if a < b else (b, a))
            used.add((b, c) if b < c else (c, b))
            used.add((a, c) if a < c else (c, a))
        return used

    def calculate_independence_number(system):
        """
        Exact calculation of independence number for n=16 using bitset branch-and-bound.
        """
        if not system:
            return 16

        # Pre-compute bitmask for each triple
        triple_masks = []
        for a, b, c in system:
            mask = (1 << a) | (1 << b) | (1 << c)
            triple_masks.append(mask)

        # Sort triples by mask value? Or just keep them.
        # For pruning, we check if (candidate_mask & triple_mask) == triple_mask

        max_size = 0
        # We want to find max k.
        # Branch and bound:
        # State: current_mask, size, next_point_to_consider
        # We iterate points from 0 to 15.
        # At each point, we can include it or exclude it.
        # If we include it, we check if it creates a triple.
        # If it does, we cannot include it.

        # Optimization:
        # If we include point i, we must ensure no triple is formed.
        # A triple is formed if (current_mask | (1<<i)) contains any triple_mask.
        # We can precompute for each point i, the set of triples it participates in?
        # Or just check all triple_masks. 16 points * ~80 triples * 16 steps is small.

        # Even better:
        # Let's use a recursive search that tries to build the largest independent set.
        # We can prune if size + (16 - current_point) <= max_size.

        def backtrack(mask, size, start_idx):
            nonlocal max_size
            if size > max_size:
                max_size = size

            # Pruning
            if size + (16 - start_idx) <= max_size:
                return

            for i in range(start_idx, 16):
                bit = 1 << i
                if mask & bit:
                    continue

                new_mask = mask | bit

                # Check independence
                valid = True
                for tm in triple_masks:
                    if (new_mask & tm) == tm:
                        valid = False
                        break

                if valid:
                    backtrack(new_mask, size + 1, i + 1)
                # If invalid, we just skip adding i.
                # Note: We do not prune the entire branch if i is invalid,
                # because adding a later point j might be valid even if i is not.
                # Wait, if adding i is invalid, it means new_mask contains a triple.
                # That triple must contain i, because mask was independent.
                # So any superset of new_mask is also dependent.
                # But we are building the set incrementally.
                # If we skip i, we can still add j.
                # So we just continue the loop.

        backtrack(0, 0, 0)
        return max_size

    def greedy_packing(seed):
        random.seed(seed)
        points = list(range(16))
        all_triples = list(combinations(points, 3))
        random.shuffle(all_triples)

        triples = []
        used_pairs = set()

        for t in all_triples:
            a, b, c = t
            p1 = (a, b) if a < b else (b, a)
            p2 = (b, c) if b < c else (c, b)
            p3 = (a, c) if a < c else (c, a)

            if p1 not in used_pairs and p2 not in used_pairs and p3 not in used_pairs:
                triples.append([a, b, c])
                used_pairs.add(p1)
                used_pairs.add(p2)
                used_pairs.add(p3)

        return triples

    def local_search(system, time_limit):
        """
        Deterministic local search with exhaustive single-point swaps.
        """
        current_system = [list(t) for t in system]
        current_score = calculate_independence_number(current_system)
        used_pairs = get_used_pairs(current_system)

        # To ensure we explore different parts of the space,
        # we can shuffle the order of triples and points, but keep the search exhaustive per step.
        # However, since we are doing a "structural change" to exact evaluation,
        # let's make the search deterministic but robust.
        # We will iterate through all possible single-point swaps in a fixed order,
        # and apply the first one that improves the score.
        # If no improvement is found in a full pass, we stop.
        # This is a steepest descent method.

        # To avoid getting stuck in a local minimum that is not the global one,
        # we could try multiple restarts or randomization, but the directive says
        # "deterministic branch-and-bound solver... allowing a genetic algorithm or local search".
        # The key is the EXACT evaluation.
        # Let's use a randomized local search but with exact evaluation,
        # running for the remaining time.
        # Actually, the current program's weakness was the heuristic evaluation.
        # If we use exact evaluation, even a simple random walk or steepest descent
        # should be much more effective.

        # Let's do a mix:
        # 1. Try all single-point swaps in a random order.
        # 2. If an improvement is found, apply it and restart the scan.
        # 3. If no improvement is found in a full scan, stop.

        random.seed(42) # Fixed seed for determinism within the run, but different from greedy seed
        no_improvement_count = 0

        while time.time() < time_limit:
            improved = False
            # Generate a random permutation of all possible moves
            # Moves are (triple_idx, point_idx_in_triple, new_point)
            moves = []
            for ti in range(len(current_system)):
                t = current_system[ti]
                for ki in range(3):
                    old_point = t[ki]
                    for new_point in range(16):
                        if new_point not in t:
                            moves.append((ti, ki, new_point))

            random.shuffle(moves)

            for ti, ki, new_point in moves:
                if time.time() > time_limit:
                    break

                t = current_system[ti]
                if new_point in t:
                    continue

                # Construct new triple
                new_t = t.copy()
                new_t[ki] = new_point

                # Check linearity
                a, b, c = t
                old_pairs = set()
                old_pairs.add((a, b) if a < b else (b, a))
                old_pairs.add((b, c) if b < c else (c, b))
                old_pairs.add((a, c) if a < c else (c, a))

                a, b, c = new_t
                new_pairs = set()
                new_pairs.add((a, b) if a < b else (b, a))
                new_pairs.add((b, c) if b < c else (c, b))
                new_pairs.add((a, c) if a < c else (c, a))

                # Check if any new pair is already in used_pairs (excluding the ones we just removed)
                # We need to be careful: if a pair is in both old and new, it's fine.
                # The conflict is if a pair in new_pairs is in used_pairs AND not in old_pairs.
                # Or simpler: temporarily remove old_pairs, check new_pairs, then restore.

                # Optimization:
                # conflict = False
                # for p in new_pairs:
                #     if p in used_pairs and p not in old_pairs:
                #         conflict = True
                #         break

                conflict = False
                for p in new_pairs:
                    if p in used_pairs and p not in old_pairs:
                        conflict = True
                        break

                if not conflict:
                    # Apply move
                    for p in old_pairs:
                        if p not in new_pairs:
                            used_pairs.remove(p)
                    for p in new_pairs:
                        if p not in old_pairs:
                            used_pairs.add(p)

                    current_system[ti] = new_t

                    new_score = calculate_independence_number(current_system)

                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        # Revert
                        for p in new_pairs:
                            if p not in old_pairs:
                                used_pairs.remove(p)
                        for p in old_pairs:
                            if p not in new_pairs:
                                used_pairs.add(p)
                        current_system[ti] = t

            if not improved:
                no_improvement_count += 1
                if no_improvement_count > 5:
                    break
            else:
                no_improvement_count = 0

        return current_system, current_score

    best_system = None
    best_score = 16

    # Run multiple instances
    # Since local search is now more effective due to exact evaluation,
    # we can afford to run fewer, longer searches.
    num_runs = 20

    for i in range(num_runs):
        if time.time() > deadline - 10.0:
            break

        init_system = greedy_packing(i)
        optimized_system, score = local_search(init_system, deadline - 5.0)

        if score < best_score:
            best_score = score
            best_system = optimized_system

            if best_score <= 5:
                break

    if best_system is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

    return best_system
