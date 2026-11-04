"""Stochastic local search (simulated annealing) for covering design C(24,7,3)."""
import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    rng = random.Random(42)

    # Precompute all t-subsets and map to indices for fast coverage counting
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # For each possible block (k-subset), precompute which t-subsets it covers
    # This is expensive for large v,k,t but for v=24,k=7,t=3 it's C(24,7)=346104
    # Each block covers C(7,3)=35 triples. We'll compute on the fly instead.

    def block_covers_tsubs(block):
        """Return set of t-subset indices covered by this block."""
        return set(tsub_index[ts] for ts in itertools.combinations(block, t))

    # --- Phase 1: Random greedy construction to get initial blocks ---
    def greedy_initial(n_blocks_target=None):
        """Build initial set of blocks using randomized greedy."""
        covered = set()
        blocks = []
        uncovered_count = n_tsubs
        while uncovered_count > 0:
            # Find all uncovered t-subsets
            # Pick a random uncovered t-subset
            uncov_list = [ts for i, ts in enumerate(all_tsubs) if i not in covered]
            if not uncov_list:
                break
            seed_ts = rng.choice(uncov_list)
            # Build a block containing seed_ts that maximizes new coverage
            best_block = None
            best_gain = -1
            # Try random blocks containing the seed triple
            seed_set = set(seed_ts)
            remaining_points = [p for p in range(v) if p not in seed_set]
            # Generate several candidate blocks
            for _ in range(200):
                candidates = list(remaining_points)
                rng.shuffle(candidates)
                block = sorted(seed_set | set(candidates[:k - t]))
                # Count new coverage
                new_cov = 0
                for ts in itertools.combinations(block, t):
                    idx = tsub_index[ts]
                    if idx not in covered:
                        new_cov += 1
                if new_cov > best_gain:
                    best_gain = new_cov
                    best_block = block
            if best_block is None:
                best_block = sorted(seed_set | set(remaining_points[:k - t]))
            blocks.append(best_block)
            # Update covered
            for ts in itertools.combinations(best_block, t):
                covered.add(tsub_index[ts])
            uncovered_count = n_tsubs - len(covered)
            if n_blocks_target and len(blocks) >= n_blocks_target:
                break
        return blocks, covered

    # --- Phase 2: Simulated Annealing ---
    def coverage_count(blocks, covered_set):
        """Return the number of covered t-subsets."""
        return len(covered_set)

    def try_improve(blocks, covered_set, time_budget):
        """Run simulated annealing to improve the covering."""
        current_blocks = list(blocks)
        current_covered = set(covered_set)
        current_score = len(current_covered)
        best_blocks = list(current_blocks)
        best_covered = set(current_covered)
        best_score = current_score

        # Temperature schedule
        T = 10.0
        T_min = 0.1
        cooling = 0.9995

        step = 0
        while time.time() < time_budget and T > T_min:
            step += 1
            # Choose a move: swap one block, replace one block, or add a block
            move_type = rng.random()
            old_block_idx = rng.randrange(len(current_blocks))
            old_block = current_blocks[old_block_idx]
            old_block_set = set(old_block)

            # Generate a new block
            if move_type < 0.5:
                # Swap: replace old_block with a random new block
                new_block = sorted(rng.sample(range(v), k))
            elif move_type < 0.8:
                # Perturb: change 1-2 points in old_block
                new_block_list = list(old_block)
                n_changes = rng.randint(1, 2)
                for _ in range(n_changes):
                    pos = rng.randrange(k)
                    new_point = rng.randrange(v)
                    if new_point not in new_block_list:
                        new_block_list[pos] = new_point
                if len(set(new_block_list)) < k:
                    new_block_list = list(old_block)
                new_block = sorted(new_block_list)
            else:
                # Add a new block (if under limit)
                if len(current_blocks) >= 200:
                    new_block = sorted(rng.sample(range(v), k))
                else:
                    new_block = sorted(rng.sample(range(v), k))

            if new_block == old_block:
                continue

            # Compute delta in coverage
            # t-subsets only covered by old_block (lost)
            lost = 0
            gained = 0
            # Count how many times each t-subset is covered by current blocks
            # For efficiency, track coverage counts
            # But recomputing from scratch is O(n_blocks * C(k,t))
            # Let's do a delta computation:
            # A t-subset is "lost" if it was covered only by old_block
            # A t-subset is "gained" if it's covered by new_block but not by any other block

            # Build coverage count map for current blocks (expensive but needed for correctness)
            # Actually, let's just recompute covered set after the move for simplicity
            # This is O(n_blocks * C(k,t)) which for 100 blocks * 35 = 3500 ops, acceptable

            # Remove old_block, add new_block
            test_blocks = current_blocks[:old_block_idx] + current_blocks[old_block_idx+1:]
            if move_type < 0.8:
                test_blocks.append(new_block)
            else:
                test_blocks.append(new_block)

            # Compute new covered set
            new_covered = set()
            for blk in test_blocks:
                for ts in itertools.combinations(blk, t):
                    new_covered.add(tsub_index[ts])

            new_score = len(new_covered)
            delta = new_score - current_score

            if delta >= 0 or rng.random() < pow(2.718281828, delta / T):
                # Accept move
                current_blocks = test_blocks
                current_covered = new_covered
                current_score = new_score

                if current_score > best_score:
                    best_score = current_score
                    best_blocks = list(current_blocks)
                    best_covered = set(current_covered)

            T *= cooling

        return best_blocks, best_covered

    # --- Phase 3: Delete redundant blocks ---
    def delete_redundant(blocks, covered_set):
        """Iteratively remove blocks whose removal doesn't uncover any t-subset."""
        blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                # Check if block i is redundant
                remaining = blocks[:i] + blocks[i+1:]
                # Compute coverage of remaining
                new_covered = set()
                for blk in remaining:
                    for ts in itertools.combinations(blk, t):
                        new_covered.add(tsub_index[ts])
                if len(new_covered) == len(covered_set):
                    # Redundant
                    blocks = remaining
                    covered_set = new_covered
                    changed = True
                    break
        return blocks, covered_set

    # --- Main ---
    # Initial construction
    initial_blocks, initial_covered = greedy_initial()
    best_blocks = list(initial_blocks)
    best_covered = set(initial_covered)
    best_score = len(best_covered)

    # Run multiple SA restarts
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        # Fresh random initial for diversity
        if r == 0:
            cur_blocks, cur_covered = list(initial_blocks), set(initial_covered)
        else:
            cur_blocks, cur_covered = greedy_initial()

        # SA improvement
        sa_budget = min(deadline - 10.0, time.time() + 20.0)
        improved_blocks, improved_covered = try_improve(cur_blocks, cur_covered, sa_budget)

        if len(improved_covered) > best_score:
            best_score = len(improved_covered)
            best_blocks = list(improved_blocks)
            best_covered = set(improved_covered)

    # Final redundant block deletion
    if time.time() < deadline - 5.0:
        best_blocks, best_covered = delete_redundant(best_blocks, best_covered)

    # Verify coverage (should be complete if best_score == n_tsubs)
    # Return as list of sorted lists
    return [sorted(b) for b in best_blocks]
