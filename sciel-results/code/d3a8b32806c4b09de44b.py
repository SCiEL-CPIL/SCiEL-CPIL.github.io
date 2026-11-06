"""B2[2] set construction with fast sum-count validation and randomized restarts."""
import time
import random

DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    best = None
    best_size = 0
    rng = random.Random(42)

    def greedy(order, seed_time):
        """Greedy: try each x in `order`, add if it keeps all sums <= 2."""
        sum_counts = [0] * 2003  # index by sum (2..2000)
        chosen = []
        for x in order:
            # Check: for each y in chosen with y <= x, sum x+y must have count < 2
            # Also 2*x must have count < 2
            ok = True
            # 2*x
            if sum_counts[2 * x] >= 2:
                ok = False
            else:
                for y in chosen:
                    if y > x:
                        break
                    s = x + y
                    if sum_counts[s] >= 2:
                        ok = False
                        break
            if ok:
                # Commit: update counts
                sum_counts[2 * x] += 1
                for y in chosen:
                    if y > x:
                        break
                    sum_counts[x + y] += 1
                chosen.append(x)
            if time.time() > seed_time:
                break
        return chosen

    # First pass: deterministic ascending greedy (baseline)
    order = list(range(1, 1001))
    result = greedy(order, DEADLINE)
    if len(result) > best_size:
        best = result
        best_size = len(result)

    # Randomized restarts with different orderings
    attempts = 0
    while time.time() < DEADLINE - 5.0:
        # Different ordering strategies to escape local optima
        strategy = rng.random()
        if strategy < 0.3:
            # Random shuffle
            order = list(range(1, 1001))
            rng.shuffle(order)
        elif strategy < 0.5:
            # Descending
            order = list(range(1000, 0, -1))
        elif strategy < 0.7:
            # Random with slight bias toward smaller numbers
            order = list(range(1, 1001))
            for i in range(len(order) - 1, 0, -1):
                j = rng.randint(0, i)
                order[i], order[j] = order[j], order[i]
            # Partial reverse to mix
            half = len(order) // 2
            order = order[:half] + list(reversed(order[half:]))
        elif strategy < 0.85:
            # Start with some small numbers, then random
            small = list(range(1, 50))
            rest = list(range(50, 1001))
            rng.shuffle(rest)
            order = small + rest
        else:
            # Random with blocks
            order = list(range(1, 1001))
            rng.shuffle(order)
            # Group into blocks and shuffle block order
            block_size = 20
            blocks = [order[i:i+block_size] for i in range(0, len(order), block_size)]
            rng.shuffle(blocks)
            order = [x for block in blocks for x in block]

        seed_deadline = min(time.time() + 8.0, DEADLINE - 5.0)
        result = greedy(order, seed_deadline)
        if len(result) > best_size:
            best = result
            best_size = len(result)
        attempts += 1

    return best if best is not None else [1]

# EVOLVE-BLOCK-END
