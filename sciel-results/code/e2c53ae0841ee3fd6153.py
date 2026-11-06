import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    def greedy_construct():
        """Greedy: include x if it doesn't create duplicate products."""
        values = []
        products = set()
        for x in range(2, 129):
            new_tails = [x]
            p = x
            for a in reversed(values):
                p *= a
                new_tails.append(p)
            if not products.intersection(new_tails):
                values.append(x)
                products.update(new_tails)
        return values

    initial_seq = greedy_construct()
    best_seq = initial_seq[:]
    best_len = len(initial_seq)

    def compute_all_products(seq):
        """Return list of all consecutive block products."""
        n = len(seq)
        if n == 0:
            return []
        prods = []
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.append(p)
        return prods

    def count_conflicts(seq):
        """Count number of duplicate product pairs."""
        if len(seq) < 2:
            return 0
        prods = compute_all_products(seq)
        c = Counter(prods)
        conflicts = 0
        for v in c.values():
            if v > 1:
                conflicts += v * (v - 1) // 2
        return conflicts

    def get_score(seq):
        """Score: length if valid, else length - penalty * conflicts."""
        if not seq:
            return 0
        conf = count_conflicts(seq)
        if conf == 0:
            return len(seq)
        return len(seq) - 50 * conf

    # --- Fast incremental conflict check for a single insertion/removal ---
    def check_insert(seq, pos, x):
        """Check if inserting x at pos creates any new product conflicts.
        Returns True if valid (no new conflicts)."""
        n = len(seq)
        if n == 0:
            return True
        # Products that involve the new element x:
        # (i, pos) for i in [0..pos], and (pos, j) for j in [pos..n-1]
        # We need to check these against all existing products.
        # Build set of existing products quickly
        existing_prods = set(compute_all_products(seq))
        # New products involving x
        new_prods = set()
        # Left extensions: x * seq[pos-1] * ... * seq[i]
        p = x
        new_prods.add(p)
        for i in range(pos - 1, -1, -1):
            p *= seq[i]
            new_prods.add(p)
        # Right extensions: x * seq[pos] * ... * seq[j]  (note: pos is insertion point, so right starts at pos)
        p = x
        for j in range(pos, n):
            p *= seq[j]
            new_prods.add(p)
        # Check overlap
        if new_prods & existing_prods:
            return False
        # Also check within new_prods themselves (should be unique since x is new and seq is strictly increasing)
        return True

    def check_remove(seq, idx):
        """Check if removing seq[idx] keeps all remaining products distinct.
        Returns True if valid."""
        new_seq = seq[:idx] + seq[idx+1:]
        return count_conflicts(new_seq) == 0

    # Simulated annealing
    current_seq = initial_seq[:]
    current_score = get_score(current_seq)
    current_conflicts = count_conflicts(current_seq)

    T = 5.0
    cool_rate = 0.9995
    best_T = T
    no_improve = 0
    step = 0

    while time.time() < deadline - 3:
        step += 1
        r = random.random()

        if r < 0.45:
            # Try to add an element
            in_seq = set(current_seq)
            available = [x for x in range(2, 129) if x not in in_seq]
            if not available:
                continue
            x = random.choice(available)
            pos = 0
            for i, v in enumerate(current_seq):
                if v < x:
                    pos = i + 1
                else:
                    break
            # Fast check: only verify new products don't conflict
            if not check_insert(current_seq, pos, x):
                continue
            new_seq = current_seq[:pos] + [x] + current_seq[pos:]
        elif r < 0.75:
            # Try to remove an element
            if not current_seq:
                continue
            idx = random.randint(0, len(current_seq) - 1)
            new_seq = current_seq[:idx] + current_seq[idx+1:]
        else:
            # Try a swap: remove one element, add another
            if len(current_seq) < 2:
                continue
            in_seq = set(current_seq)
            available = [x for x in range(2, 129) if x not in in_seq]
            if not available:
                continue
            idx = random.randint(0, len(current_seq) - 1)
            x = random.choice(available)
            # Remove idx, then insert x in sorted position
            temp = current_seq[:idx] + current_seq[idx+1:]
            pos = 0
            for i, v in enumerate(temp):
                if v < x:
                    pos = i + 1
                else:
                    break
            new_seq = temp[:pos] + [x] + temp[pos:]

        new_score = get_score(new_seq)
        delta = new_score - current_score

        # Accept if better or with probability exp(-delta/T) for worse
        if delta >= 0 or (T > 0.01 and random.random() < np.exp(delta / T)):
            current_seq = new_seq
            current_score = new_score
            current_conflicts = count_conflicts(current_seq)

            if current_conflicts == 0:
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = current_seq[:]
                    no_improve = 0
                    T = max(T, 2.0)
            else:
                no_improve += 1

        # Cooling
        T *= cool_rate
        if T < 0.1:
            T = 0.1

        # Restart if stuck
        if no_improve > 400:
            no_improve = 0
            T = best_T * 0.5
            if current_seq:
                for _ in range(min(3, len(current_seq))):
                    if random.random() < 0.5:
                        idx = random.randint(0, len(current_seq) - 1)
                        current_seq = current_seq[:idx] + current_seq[idx+1:]
                    else:
                        in_seq = set(current_seq)
                        available = [x for x in range(2, 129) if x not in in_seq]
                        if available:
                            x = random.choice(available)
                            pos = 0
                            for i, v in enumerate(current_seq):
                                if v < x:
                                    pos = i + 1
                                else:
                                    break
                            current_seq = current_seq[:pos] + [x] + current_seq[pos:]
                current_score = get_score(current_seq)
                current_conflicts = count_conflicts(current_seq)

    # Final polish: try to add any missing elements that don't create conflicts
    in_best = set(best_seq)
    changed = True
    while changed:
        changed = False
        for x in range(2, 129):
            if x not in in_best:
                pos = 0
                for i, v in enumerate(best_seq):
                    if v < x:
                        pos = i + 1
                    else:
                        break
                if check_insert(best_seq, pos, x):
                    best_seq = best_seq[:pos] + [x] + best_seq[pos:]
                    in_best.add(x)
                    changed = True
                    best_len = len(best_seq)

    return best_seq
