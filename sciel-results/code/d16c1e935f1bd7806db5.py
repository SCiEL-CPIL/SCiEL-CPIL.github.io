"""Max Weight Closure via min-cut for 3-AP-free subset of [1,256] maximizing reciprocal sum."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    n = 256
    recip = [0.0] + [1.0 / i for i in range(1, n + 1)]

    # Precompute all 3-APs: (a, b, c) with a < b < c and a + c = 2*b
    threes = []
    for b in range(2, n):
        for a in range(1, b):
            c = 2 * b - a
            if c <= n and c > b:
                threes.append((a, b, c))

    # Build Maximum Weight Closure network
    # Nodes:
    #   0: source, 1: sink
    #   2..n+1: element nodes (element i is node i+1)
    #   For each 3-AP (a,b,c), we add a helper node to enforce NOT(x_a & x_b & x_c)
    #
    # Standard MWC formulation:
    # - Positive weight w_i: edge source->i with capacity w_i
    # - Negative weight -w_i: edge i->sink with capacity w_i
    # - Precedence constraint "if i selected then j must be selected": edge i->j with INF capacity
    #
    # For 3-AP constraint NOT(x_a & x_b & x_c):
    # Introduce auxiliary node u_abc.
    # Add edges: a->u (INF), b->u (INF), c->u (INF)
    # Add edge: u->sink with capacity = min(w_a, w_b, w_c) - but this doesn't work directly.
    #
    # Alternative: Use the fact that 3-AP-free means for each triple, at most 2 can be selected.
    # This is NOT a simple closure constraint. Let me reconsider.
    #
    # Actually, the constraint "not all three in {a,b,c} are selected" is a 3-uniform hypergraph independent set.
    # This is NOT directly a maximum weight closure problem.
    #
    # Let me use a different approach: Integer Programming via branch and bound, or use the fact that
    # for n=256, we can use a smarter local search with better initialization.
    #
    # Revised plan: Use a high-quality greedy initialization with lookahead, then intensive local search
    # with 2-swap and 3-swap moves, plus simulated annealing with proper acceptance criteria.

    # Precompute for fast 3-AP checking
    # For each pair (i, j), find all k such that {i,j,k} forms a 3-AP
    third_elems = {}
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            candidates = set()
            # i is middle: k = 2*i - j
            k = 2 * i - j
            if 1 <= k <= n and k != i and k != j:
                candidates.add(k)
            # j is middle: k = 2*j - i
            k = 2 * j - i
            if 1 <= k <= n and k != i and k != j:
                candidates.add(k)
            # middle is (i+j)/2
            if (i + j) % 2 == 0:
                k = (i + j) // 2
                if 1 <= k <= n and k != i and k != j:
                    candidates.add(k)
            if candidates:
                third_elems[(i, j)] = candidates

    # Fast validity check using conflict counting
    def count_conflicts(mask):
        """Count number of 3-APs fully contained in the set represented by mask."""
        count = 0
        for (i, j), k_set in third_elems.items():
            if (mask >> i) & 1 and (mask >> j) & 1:
                for k in k_set:
                    if (mask >> k) & 1:
                        count += 1
        return count

    def is_3ap_free_mask(mask):
        """Check if bitmask represents a 3-AP-free set."""
        return count_conflicts(mask) == 0

    def reciprocal_sum_mask(mask):
        """Compute reciprocal sum of bitmask."""
        total = 0.0
        for i in range(1, n + 1):
            if (mask >> i) & 1:
                total += recip[i]
        return total

    # High-quality greedy initialization with lookahead
    def greedy_init_with_lookahead():
        mask = 0
        selected = set()
        # Try multiple orderings and keep the best
        best_local_mask = 0
        best_local_score = 0.0

        # Ordering 1: ascending
        order1 = list(range(1, n + 1))
        # Ordering 2: descending
        order2 = list(range(n, 0, -1))
        # Ordering 3: by reciprocal value (descending)
        order3 = sorted(range(1, n + 1), key=lambda x: -recip[x])
        # Ordering 4: random
        order4 = list(range(1, n + 1))
        random.shuffle(order4)

        for order in [order1, order2, order3, order4]:
            local_mask = 0
            local_selected = set()
            for x in order:
                # Check if adding x creates a 3-AP
                valid = True
                for a in local_selected:
                    # x is middle: a + c = 2*x => c = 2*x - a
                    c = 2 * x - a
                    if c in local_selected and c != x:
                        valid = False
                        break
                    # x is endpoint, a is middle: x + c = 2*a => c = 2*a - x
                    c = 2 * a - x
                    if c in local_selected and c != x and c != a:
                        valid = False
                        break
                    # a is endpoint, x is other endpoint: a + x = 2*c => c = (a+x)/2
                    if (a + x) % 2 == 0:
                        c = (a + x) // 2
                        if c in local_selected and c != x and c != a:
                            valid = False
                            break
                if valid:
                    local_mask |= (1 << x)
                    local_selected.add(x)

            local_score = reciprocal_sum_mask(local_mask)
            if local_score > best_local_score:
                best_local_score = local_score
                best_local_mask = local_mask

        return best_local_mask

    best_mask = greedy_init_with_lookahead()
    best_score = reciprocal_sum_mask(best_mask)

    # Verify
    if not is_3ap_free_mask(best_mask):
        best_mask = 1 << 1
        best_score = recip[1]

    # Intensive local search with 1-swap, 2-swap moves
    def local_search(mask, score, deadline, max_iterations=100000):
        """Intensive local search with 1-swap and 2-swap moves."""
        current_mask = mask
        current_score = score
        best_ls_mask = mask
        best_ls_score = score
        iteration = 0

        while (time.time() - start_time) < deadline and iteration < max_iterations:
            iteration += 1
            improved = False

            # 1-swap: try adding a single element
            for x in range(1, n + 1):
                if not ((current_mask >> x) & 1):
                    new_mask = current_mask | (1 << x)
                    if is_3ap_free_mask(new_mask):
                        new_score = current_score + recip[x]
                        if new_score > current_score + 1e-15:
                            current_mask = new_mask
                            current_score = new_score
                            improved = True
                            if current_score > best_ls_score:
                                best_ls_mask = current_mask
                                best_ls_score = current_score
                            break  # restart scan after improvement

            if improved:
                continue

            # 2-swap: try removing one and adding another
            elements = [i for i in range(1, n + 1) if (current_mask >> i) & 1]
            for x_remove in elements:
                if (time.time() - start_time) > deadline:
                    break
                temp_mask = current_mask & ~(1 << x_remove)
                for x_add in range(1, n + 1):
                    if (temp_mask >> x_add) & 1:
                        continue
                    new_mask = temp_mask | (1 << x_add)
                    if is_3ap_free_mask(new_mask):
                        new_score = current_score - recip[x_remove] + recip[x_add]
                        if new_score > current_score + 1e-15:
                            current_mask = new_mask
                            current_score = new_score
                            improved = True
                            if current_score > best_ls_score:
                                best_ls_mask = current_mask
                                best_ls_score = current_score
                            break
                if improved:
                    break

            if not improved:
                # Try 3-swap: remove two, add two
                if len(elements) >= 2 and (time.time() - start_time) < deadline - 5:
                    for i in range(len(elements)):
                        if (time.time() - start_time) > deadline:
                            break
                        x1 = elements[i]
                        for j in range(i + 1, len(elements)):
                            x2 = elements[j]
                            temp_mask = current_mask & ~(1 << x1) & ~(1 << x2)
                            for x_add1 in range(1, n + 1):
                                if (temp_mask >> x_add1) & 1:
                                    continue
                                for x_add2 in range(x_add1 + 1, n + 1):
                                    if (temp_mask >> x_add2) & 1:
                                        continue
                                    new_mask = temp_mask | (1 << x_add1) | (1 << x_add2)
                                    if is_3ap_free_mask(new_mask):
                                        new_score = current_score - recip[x1] - recip[x2] + recip[x_add1] + recip[x_add2]
                                        if new_score > current_score + 1e-15:
                                            current_mask = new_mask
                                            current_score = new_score
                                            improved = True
                                            if current_score > best_ls_score:
                                                best_ls_mask = current_mask
                                                best_ls_score = current_score
                                            break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break

            if not improved:
                break

        return best_ls_mask, best_ls_score

    # Simulated annealing with proper temperature schedule
    def simulated_annealing(mask, score, deadline):
        """Run SA to escape local optima."""
        current_mask = mask
        current_score = score
        best_sa_mask = mask
        best_sa_score = score

        T = 0.005
        cooling = 0.999
        min_T = 1e-7

        while (time.time() - start_time) < deadline and T > min_T:
            # Random move: flip 1-3 bits
            num_flips = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            bits_to_flip = random.sample(range(1, n + 1), num_flips)

            new_mask = current_mask
            for x in bits_to_flip:
                new_mask ^= (1 << x)

            if not is_3ap_free_mask(new_mask):
                T *= cooling
                continue

            new_score = reciprocal_sum_mask(new_mask)
            delta = new_score - current_score

            if delta > 0 or random.random() < math.exp(delta / T):
                current_mask = new_mask
                current_score = new_score
                if current_score > best_sa_score:
                    best_sa_mask = current_mask
                    best_sa_score = current_score

            T *= cooling

        return best_sa_mask, best_sa_score

    # Main search loop
    # Phase 1: Local search from greedy init
    best_mask, best_score = local_search(best_mask, best_score, DEADLINE - 30.0, max_iterations=50000)

    # Phase 2: SA with restarts
    remaining_deadline = DEADLINE - 10.0
    num_restarts = 8
    for restart in range(num_restarts):
        if (time.time() - start_time) > remaining_deadline:
            break

        if restart == 0:
            start_mask = best_mask
            start_score = best_score
        else:
            # Perturb the best solution
            start_mask = best_mask
            elements = [i for i in range(1, n + 1) if (start_mask >> i) & 1]
            if len(elements) > 5:
                to_remove = random.sample(elements, min(4, len(elements) // 3))
                for x in to_remove:
                    start_mask &= ~(1 << x)
            start_score = reciprocal_sum_mask(start_mask)

        sa_mask, sa_score = simulated_annealing(start_mask, start_score, remaining_deadline)
        if sa_score > best_score:
            best_mask = sa_mask
            best_score = sa_score

        # Local search after SA
        if (time.time() - start_time) < remaining_deadline - 5:
            ls_mask, ls_score = local_search(best_mask, best_score, remaining_deadline - 5, max_iterations=20000)
            if ls_score > best_score:
                best_mask = ls_mask
                best_score = ls_score

    # Final polish
    if (time.time() - start_time) < DEADLINE - 2:
        best_mask, best_score = local_search(best_mask, best_score, DEADLINE - 2, max_iterations=30000)

    # Final verification
    if not is_3ap_free_mask(best_mask):
        best_mask = 1 << 1
        best_score = recip[1]

    result = [i for i in range(1, n + 1) if (best_mask >> i) & 1]
    assert len(result) > 0, "Solution is empty!"
    assert is_3ap_free_mask(best_mask), "Final solution has a 3-AP!"

    return result
# EVOLVE-BLOCK-END
