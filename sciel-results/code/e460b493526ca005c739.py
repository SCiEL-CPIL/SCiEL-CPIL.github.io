import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 10.0

    # Beam Search parameters
    beam_width = 80
    N = 512
    max_a = N - 1

    # Pre-compute weights and coprime relationships
    weights = [0.0] * (max_a + 1)
    for a in range(1, max_a + 1):
        weights[a] = 1.0 / (N - a)

    # Precompute coprime matrix for speed
    coprime = [[False] * (max_a + 1) for _ in range(max_a + 1)]
    for a in range(1, max_a + 1):
        for b in range(a + 1, max_a + 1):
            coprime[a][b] = (gcd(a, b) == 1)

    # Initialize beam with one empty state
    beams = [(0.0, [])]

    # Iterate a from max_a down to 1
    for a in range(max_a, 0, -1):
        w = weights[a]
        new_beams = []

        for score, chosen in beams:
            # Option 1: Skip a
            new_beams.append((score, chosen))

            # Option 2: Include a (if coprime to all in chosen)
            can_include = True
            for b in chosen:
                if not coprime[a][b]:
                    can_include = False
                    break
            if can_include:
                new_beams.append((score + w, chosen + [a]))

        # Prune to beam_width
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]

        # Time check with adaptive beam width
        elapsed = time.time() - start_time
        remaining = deadline - safety_margin - elapsed
        if remaining < 0:
            break
        # Reduce beam width if running out of time
        if remaining < 30:
            beam_width = max(20, int(beam_width * 0.7))

    # Take the best beam
    best_beam = max(beams, key=lambda x: x[0])
    best_score = best_beam[0]
    best_chosen = list(best_beam[1])
    chosen_set = set(best_chosen)

    # Greedy fill: try to add any remaining coprime numbers
    changed = True
    while changed:
        changed = False
        for a in range(max_a, 0, -1):
            if a in chosen_set:
                continue
            can_add = True
            for b in best_chosen:
                if not coprime[a][b]:
                    can_add = False
                    break
            if can_add:
                best_chosen.append(a)
                chosen_set.add(a)
                best_score += weights[a]
                changed = True
        # Check time
        if time.time() - start_time > deadline - safety_margin:
            break

    # Local search polish: try swapping one element for another
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin - 10:
        improved = True
        while improved:
            improved = False
            for i, a in enumerate(best_chosen):
                # Try removing a and adding a better b
                for b in range(max_a, 0, -1):
                    if b == a or b in chosen_set:
                        continue
                    # Check if b is coprime to all others (excluding a)
                    can_add_b = True
                    for j, c in enumerate(best_chosen):
                        if j == i:
                            continue
                        if not coprime[b][c]:
                            can_add_b = False
                            break
                    if not can_add_b:
                        continue
                    # Check if b is coprime to a (not needed since we're replacing)
                    new_score = best_score - weights[a] + weights[b]
                    if new_score > best_score + 1e-15:
                        best_chosen[i] = b
                        chosen_set.remove(a)
                        chosen_set.add(b)
                        best_score = new_score
                        improved = True
                        break
                if time.time() - start_time > deadline - safety_margin:
                    break
            if time.time() - start_time > deadline - safety_margin:
                break

    return best_chosen
