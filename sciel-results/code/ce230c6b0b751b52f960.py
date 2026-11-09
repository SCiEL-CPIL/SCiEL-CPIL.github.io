"""B2[2] set in 1..1000: maximize size with every sum having <= 2 representations a+b, a<=b."""
import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - t0

    def is_valid_insertion(x, chosen, counts):
        """Check if inserting x keeps every sum <= 2."""
        for y in chosen:
            s = x + y
            if counts.get(s, 0) + 1 > 2:
                return False
        s = 2 * x
        if counts.get(s, 0) + 1 > 2:
            return False
        return True

    def insertion_slack(x, chosen, counts):
        """Return number of new sums that would have count < 2 after insertion, or -1 if invalid."""
        slack = 0
        for y in chosen:
            s = x + y
            c = counts.get(s, 0) + 1
            if c > 2:
                return -1
            if c < 2:
                slack += 1
        s = 2 * x
        c = counts.get(s, 0) + 1
        if c > 2:
            return -1
        if c < 2:
            slack += 1
        return slack

    def apply_insertion(x, chosen, counts, chosen_set):
        """Insert x, update counts. Assumes valid."""
        chosen.append(x)
        chosen_set.add(x)
        for y in chosen[:-1]:
            s = x + y
            counts[s] = counts.get(s, 0) + 1
        s = 2 * x
        counts[s] = counts.get(s, 0) + 1
        return chosen, counts, chosen_set

    def remove_element(x, chosen, counts, chosen_set):
        """Remove x, update counts."""
        chosen.remove(x)
        chosen_set.discard(x)
        for y in chosen:
            s = x + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
        s = 2 * x
        counts[s] -= 1
        if counts[s] == 0:
            del counts[s]
        return chosen, counts, chosen_set

    # ---- Phase 1: greedy with slack-based acceptance ----
    random.seed(42)
    chosen = []
    counts = {}
    chosen_set = set()
    for x in range(1, 1001):
        if elapsed() > DEADLINE - 40:
            break
        slack = insertion_slack(x, chosen, counts)
        if slack < 0:
            continue
        # Accept with probability based on slack: higher slack = more room = more likely to accept
        # Use a soft threshold: accept if slack >= 1 with high prob, slack == 0 with lower prob
        if slack >= 2:
            accept = True
        elif slack == 1:
            accept = random.random() < 0.7
        else:
            accept = random.random() < 0.3
        if accept:
            apply_insertion(x, chosen, counts, chosen_set)

    # ---- Phase 2: iterative add-only local search ----
    improved = True
    while improved and elapsed() < DEADLINE - 20:
        improved = False
        best_x = None
        best_slack = -1
        for x in range(1, 1001):
            if x in chosen_set:
                continue
            slack = insertion_slack(x, chosen, counts)
            if slack > best_slack:
                best_slack = slack
                best_x = x
        if best_x is not None and best_slack >= 0:
            apply_insertion(best_x, chosen, counts, chosen_set)
            improved = True

    # ---- Phase 3: swap-based local search to break plateaus ----
    if elapsed() < DEADLINE - 10:
        # Try random swaps: remove one element, try to add two new ones
        for trial in range(50):
            if elapsed() > DEADLINE - 10:
                break
            if not chosen:
                break
            a = random.choice(chosen)
            # Save state
            chosen_copy = list(chosen)
            counts_copy = dict(counts)
            set_copy = set(chosen_set)
            
            # Remove a
            remove_element(a, chosen, counts, chosen_set)
            
            # Try to find two elements that can be added
            added = []
            for x in range(1, 1001):
                if x in chosen_set:
                    continue
                slack = insertion_slack(x, chosen, counts)
                if slack >= 0:
                    apply_insertion(x, chosen, counts, chosen_set)
                    added.append(x)
                    if len(added) == 2:
                        break
            
            if len(added) == 2:
                # We grew by 1 (removed 1, added 2)
                pass
            else:
                # Revert: restore original state
                chosen.clear()
                chosen.extend(chosen_copy)
                counts.clear()
                counts.update(counts_copy)
                chosen_set.clear()
                chosen_set.update(set_copy)

    # ---- Phase 4: final exhaustive add pass ----
    while elapsed() < DEADLINE - 3:
        best_x = None
        best_slack = -1
        for x in range(1, 1001):
            if x in chosen_set:
                continue
            slack = insertion_slack(x, chosen, counts)
            if slack > best_slack:
                best_slack = slack
                best_x = x
        if best_x is None or best_slack < 0:
            break
        apply_insertion(best_x, chosen, counts, chosen_set)

    return chosen
