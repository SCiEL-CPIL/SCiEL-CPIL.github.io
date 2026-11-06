import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit

    def time_left():
        return deadline - time.time()

    def compute_all_products(seq):
        """Compute all non-empty consecutive block products."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products

    def check_valid_addition(seq, x, products_set):
        """
        Check if x can be appended to seq.
        Returns (is_valid, new_tails)
        """
        new_tails = [x]
        prod = x
        for a in reversed(seq):
            prod *= a
            new_tails.append(prod)

        if products_set.intersection(new_tails):
            return False, None

        return True, new_tails

    def greedy_construct(start_val=2, end_val=128, skip_set=None, seed=None,
                         future_look=0, weight_future=0.0):
        """
        Greedy construction with optional future collision awareness.
        """
        if seed is not None:
            random.seed(seed)

        seq = []
        products_set = set()

        for x in range(start_val, end_val + 1):
            if time_left() < 3.0:
                break

            if skip_set and x in skip_set:
                continue

            valid, new_tails = check_valid_addition(seq, x, products_set)

            if valid:
                # Optional: check future collision risk
                if future_look > 0 and weight_future > 0:
                    future_collisions = 0
                    for y in range(x + 1, min(x + 1 + future_look, end_val + 1)):
                        temp_tails = [y]
                        prod = y
                        for a in reversed(seq + [x]):
                            prod *= a
                            temp_tails.append(prod)
                        if (products_set | set(new_tails)).intersection(temp_tails):
                            future_collisions += 1

                    if future_collisions > 2 and random.random() < weight_future:
                        continue

                seq.append(x)
                products_set.update(new_tails)

        return seq

    def local_search_polish(seq, max_iterations=200):
        """
        Try single-element swaps and insertions to improve the sequence.
        """
        if not seq or len(seq) < 2:
            return seq

        best_seq = list(seq)
        best_len = len(best_seq)

        for iteration in range(max_iterations):
            if time_left() < 5.0:
                break

            improved = False

            for pos in range(len(best_seq)):
                if time_left() < 5.0:
                    break

                temp_seq = best_seq[:pos] + best_seq[pos + 1:]
                temp_products = compute_all_products(temp_seq)

                prev_val = temp_seq[pos - 1] if pos > 0 else 1
                next_val = temp_seq[pos] if pos < len(temp_seq) else 129

                for x in range(prev_val + 1, next_val):
                    if x == best_seq[pos]:
                        continue

                    valid, new_tails = check_valid_addition(temp_seq[:pos], x, temp_products)
                    if valid:
                        full_seq = temp_seq[:pos] + [x] + temp_seq[pos:]
                        full_products = compute_all_products(full_seq)

                        if len(full_products) == len(full_seq) * (len(full_seq) + 1) // 2:
                            if len(full_seq) > best_len:
                                best_seq = full_seq
                                best_len = len(best_seq)
                                improved = True
                                break
                            elif len(full_seq) == best_len:
                                if sum(full_seq) < sum(best_seq):
                                    best_seq = full_seq
                                    improved = True
                                    break

            if not improved:
                break

        return best_seq

    def backtracking_improve(seq, max_backtracks=5):
        """
        Try to improve by backtracking: remove last k elements and try different choices.
        """
        if not seq or len(seq) < 3:
            return seq

        best = list(seq)

        for back_depth in range(1, min(max_backtracks, len(seq)) + 1):
            if time_left() < 5.0:
                break

            truncated = seq[:-back_depth] if back_depth < len(seq) else []

            # Recompute products for truncated
            truncated_products = compute_all_products(truncated)

            # Try greedy fill from truncated point
            new_seq = list(truncated)
            new_products = set(truncated_products)

            start_x = truncated[-1] + 1 if truncated else 2
            for x in range(start_x, 129):
                if time_left() < 5.0:
                    break
                valid, new_tails = check_valid_addition(new_seq, x, new_products)
                if valid:
                    new_seq.append(x)
                    new_products.update(new_tails)

            if len(new_seq) > len(best):
                best = new_seq

        return best

    def branch_and_bound_search(seq, products_set, start_idx, max_depth, best_len):
        """
        Branch and bound: try to extend the sequence by choosing from remaining candidates.
        """
        if time_left() < 5.0:
            return seq, products_set

        if max_depth <= 0 or len(seq) + (128 - start_idx + 1) <= best_len:
            return seq, products_set

        best = list(seq)
        best_products = set(products_set)
        best_len_found = len(best)

        for x in range(start_idx, 129):
            if time_left() < 5.0:
                break

            valid, new_tails = check_valid_addition(best, x, best_products)
            if valid:
                candidate_seq = best + [x]
                candidate_products = best_products | set(new_tails)

                if len(candidate_seq) + (128 - x) > best_len_found:
                    improved_seq, improved_products = branch_and_bound_search(
                        candidate_seq, candidate_products, x + 1, max_depth - 1, best_len_found
                    )
                    if len(improved_seq) > best_len_found:
                        best = improved_seq
                        best_products = improved_products
                        best_len_found = len(best)

        return best, best_products

    def iterative_deepening_search(seq, products_set, max_iterations=50):
        """
        Iterative deepening: gradually increase the search depth.
        """
        best_seq = list(seq)
        best_products = set(products_set)

        for depth in range(1, min(6, 128 - seq[-1] + 1) if seq else 6):
            if time_left() < 5.0:
                break

            improved_seq, improved_products = branch_and_bound_search(
                best_seq, best_products, best_seq[-1] + 1 if best_seq else 2,
                depth, len(best_seq)
            )

            if len(improved_seq) > len(best_seq):
                best_seq = improved_seq
                best_products = improved_products

        return best_seq

    # Phase 1: Standard greedy (baseline)
    best_seq = greedy_construct(2)

    # Phase 2: Sweep start values
    for sv in range(3, 16):
        if time_left() < 15.0:
            break
        seq = greedy_construct(sv)
        if len(seq) > len(best_seq):
            best_seq = seq

    # Phase 3: Heavy randomized trials (main search engine)
    random.seed(42)
    trials = 0
    while time_left() > 25.0:  # Leave 25s for refinement
        trials += 1
        start_val = random.randint(2, 12)
        base_skip = random.uniform(0.1, 0.5)

        seq = []
        products_set = set()

        for x in range(start_val, 129):
            if time_left() < 25.0:
                break

            if x < 15:
                p_skip = base_skip * (15 - x) / 15.0
                if random.random() < p_skip:
                    continue

            valid, new_tails = check_valid_addition(seq, x, products_set)
            if valid:
                seq.append(x)
                products_set.update(new_tails)

        if len(seq) > len(best_seq):
            best_seq = seq

    # Phase 4: Backtracking improvement
    if len(best_seq) > 3 and time_left() > 15.0:
        improved = backtracking_improve(best_seq, max_backtracks=4)
        if len(improved) > len(best_seq):
            best_seq = improved

    # Phase 5: Local search polish
    if len(best_seq) > 3 and time_left() > 10.0:
        polished = local_search_polish(best_seq, max_iterations=100)
        if len(polished) > len(best_seq):
            best_seq = polished

    # Phase 6: Branch and bound from current best
    if len(best_seq) > 3 and time_left() > 8.0:
        products_set = compute_all_products(best_seq)
        improved = iterative_deepening_search(best_seq, products_set, max_iterations=20)
        if len(improved) > len(best_seq):
            best_seq = improved

    # Final validation
    if best_seq:
        products = compute_all_products(best_seq)
        expected_count = len(best_seq) * (len(best_seq) + 1) // 2
        if len(products) != expected_count:
            for cut in range(len(best_seq), 0, -1):
                prefix = best_seq[:cut]
                p = compute_all_products(prefix)
                if len(p) == cut * (cut + 1) // 2:
                    best_seq = prefix
                    break

    return best_seq
