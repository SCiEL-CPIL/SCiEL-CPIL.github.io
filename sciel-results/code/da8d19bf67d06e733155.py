"""Rewritten program: fast incremental greedy + random restarts with time guard."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill

    n = 4096
    max_k = 80

    def can_add(x, used_sums, elements):
        """Check if adding x creates any triple-sum collision.
        used_sums: set of all triple sums from current elements.
        elements: list of current elements (sorted).
        New triple sums involving x:
          - x+x+x = 3x
          - x+x+y for y in elements
          - x+y+z for y<=z in elements (y,z from existing)
        Must not collide with used_sums or with each other.
        """
        new_sums = set()
        # x+x+x
        s = 3 * x
        if s in used_sums:
            return False
        new_sums.add(s)
        # x+x+y
        for y in elements:
            s = 2 * x + y
            if s in used_sums:
                return False
            if s in new_sums:
                return False
            new_sums.add(s)
        # x+y+z for y<=z in elements
        for i in range(len(elements)):
            for j in range(i, len(elements)):
                s = x + elements[i] + elements[j]
                if s in used_sums:
                    return False
                if s in new_sums:
                    return False
                new_sums.add(s)
        return True

    def greedy_build(seed=None, max_elems=80, use_random_order=False):
        """Build a set greedily. If use_random_order, try candidates in random order."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        elements = []
        used_sums = set()

        if use_random_order:
            # Generate a random permutation of [1, n]
            candidates = list(range(1, n + 1))
            rng.shuffle(candidates)
        else:
            candidates = list(range(1, n + 1))

        for x in candidates:
            if time.time() > deadline:
                break
            if len(elements) >= max_elems:
                break
            if can_add(x, used_sums, elements):
                elements.append(x)
                # Add new sums
                used_sums.add(3 * x)
                for y in elements[:-1]:
                    used_sums.add(2 * x + y)
                for i in range(len(elements) - 1):
                    for j in range(i, len(elements) - 1):
                        used_sums.add(x + elements[i] + elements[j])
        return elements

    # First: pure greedy (deterministic, fast)
    best = greedy_build(seed=0, use_random_order=False)
    best_len = len(best)

    # Now do random restarts with different seeds and strategies
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        # Vary the approach: sometimes bias toward smaller numbers, sometimes random
        if attempt % 3 == 0:
            # Random order
            result = greedy_build(seed=attempt * 7919, use_random_order=True)
        elif attempt % 3 == 1:
            # Greedy but with a small random perturbation: skip first few small numbers
            rng = random.Random(attempt * 104729)
            skip = set(rng.sample(range(1, 50), min(5, attempt % 5 + 1)))
            elements = []
            used_sums = set()
            for x in range(1, n + 1):
                if time.time() > deadline:
                    break
                if len(elements) >= max_elems:
                    break
                if x in skip:
                    continue
                if can_add(x, used_sums, elements):
                    elements.append(x)
                    used_sums.add(3 * x)
                    for y in elements[:-1]:
                        used_sums.add(2 * x + y)
                    for i in range(len(elements) - 1):
                        for j in range(i, len(elements) - 1):
                            used_sums.add(x + elements[i] + elements[j])
            result = elements
        else:
            # Greedy with random start offset
            rng = random.Random(attempt * 15485863)
            offset = rng.randint(1, 20)
            elements = []
            used_sums = set()
            for x in range(1, n + 1):
                if time.time() > deadline:
                    break
                if len(elements) >= max_elems:
                    break
                if x <= offset:
                    continue
                if can_add(x, used_sums, elements):
                    elements.append(x)
                    used_sums.add(3 * x)
                    for y in elements[:-1]:
                        used_sums.add(2 * x + y)
                    for i in range(len(elements) - 1):
                        for j in range(i, len(elements) - 1):
                            used_sums.add(x + elements[i] + elements[j])
            result = elements

        if len(result) > best_len:
            best_len = len(result)
            best = result

    return best
