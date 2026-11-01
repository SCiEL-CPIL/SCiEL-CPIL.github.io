"""Rewritten program: randomized greedy with best-of-many and deadline guard."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 140  # safety margin before 150s kill
    best_rows = None
    best_scope = None

    # First, always run the deterministic greedy as a baseline
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    best_rows = rows
    best_scope = max(max(r) for r in rows)

    # Now try randomized variants with different orderings and tie-breaking
    rng = random.Random(12345)
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Randomize the order in which we try candidate next marks
        # by shuffling a window of candidates
        used, rows = set(), []
        ok = True
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                # Try candidates in a randomized window
                found = False
                # Deterministic scan with occasional randomization
                if rng.random() < 0.3:
                    # Randomize: pick a random offset to start scanning
                    start = x + rng.randint(0, 5)
                    for cand in range(start, x + 20):
                        new = {cand - a for a in row}
                        if not new & used:
                            x = cand
                            row.append(x)
                            used |= new
                            found = True
                            break
                    if not found:
                        # Fallback to sequential
                        x = start - 1
                        while len(row) < k + 1:
                            x += 1
                            new = {x - a for a in row}
                            if not new & used:
                                row.append(x)
                                used |= new
                else:
                    # Sequential scan
                    while len(row) < k + 1:
                        x += 1
                        new = {x - a for a in row}
                        if not new & used:
                            row.append(x)
                            used |= new
                            break
            if len(row) != k + 1:
                ok = False
                break
            rows.append(row)
        if ok:
            scope = max(max(r) for r in rows)
            if best_scope is None or scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]

    return best_rows
# EVOLVE-BLOCK-END
