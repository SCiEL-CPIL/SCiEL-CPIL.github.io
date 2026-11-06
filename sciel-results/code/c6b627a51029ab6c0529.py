import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin within 90s limit

    N = 256
    MAX_SIZE_A = 16
    MAX_SIZE_B = 16

    best_size = 0
    best_A = []
    best_B = []

    class SetState:
        __slots__ = ['elements', 'sums_mask', 'diffs_mask', 'element_set']

        def __init__(self):
            self.elements = []
            self.sums_mask = 0
            self.diffs_mask = 0
            self.element_set = set()

        def try_add(self, x, forbidden_diffs_mask):
            if x in self.element_set:
                return False
            # Check Sidon: 2x and x+a for all a in elements
            if (self.sums_mask >> (2 * x)) & 1:
                return False
            for a in self.elements:
                if (self.sums_mask >> (x + a)) & 1:
                    return False
            # Check diff conflicts with forbidden
            for a in self.elements:
                d = abs(x - a)
                if d > 0 and (forbidden_diffs_mask >> d) & 1:
                    return False
            # Commit
            self.elements.append(x)
            self.element_set.add(x)
            for a in self.elements[:-1]:
                self.sums_mask |= (1 << (x + a))
            self.sums_mask |= (1 << (2 * x))
            for a in self.elements[:-1]:
                d = abs(x - a)
                if d > 0:
                    self.diffs_mask |= (1 << d)
            return True

        def remove(self, x):
            idx = self.elements.index(x)
            self.elements.pop(idx)
            self.element_set.discard(x)
            # Recompute masks
            self.sums_mask = 0
            self.diffs_mask = 0
            for i in range(len(self.elements)):
                for j in range(i, len(self.elements)):
                    self.sums_mask |= (1 << (self.elements[i] + self.elements[j]))
                for j in range(i + 1, len(self.elements)):
                    d = abs(self.elements[i] - self.elements[j])
                    self.diffs_mask |= (1 << d)

    def build_interleaved(order_A, order_B, max_a, max_b):
        """
        Build A and B simultaneously, interleaving to avoid dead-ends.
        """
        A = SetState()
        B = SetState()

        ia, ib = 0, 0
        while ia < len(order_A) and ib < len(order_B):
            # Try to add to A
            if len(A.elements) < max_a:
                added = False
                for _ in range(3):  # Try a few candidates
                    if ia < len(order_A):
                        x = order_A[ia]
                        ia += 1
                        if A.try_add(x, B.diffs_mask):
                            added = True
                            break
                if added:
                    continue

            # Try to add to B
            if len(B.elements) < max_b:
                added = False
                for _ in range(3):
                    if ib < len(order_B):
                        x = order_B[ib]
                        ib += 1
                        if B.try_add(x, A.diffs_mask):
                            added = True
                            break
                if added:
                    continue

            # If neither could add, advance both
            if ia < len(order_A):
                ia += 1
            if ib < len(order_B):
                ib += 1

        # Finish remaining A
        while ia < len(order_A) and len(A.elements) < max_a:
            x = order_A[ia]
            ia += 1
            A.try_add(x, B.diffs_mask)

        # Finish remaining B
        while ib < len(order_B) and len(B.elements) < max_b:
            x = order_B[ib]
            ib += 1
            B.try_add(x, A.diffs_mask)

        return A, B

    def build_sequential(order_A, order_B, max_a, max_b):
        """
        Build A first, then B avoiding A's diffs.
        """
        A = SetState()
        for x in order_A:
            if len(A.elements) >= max_a:
                break
            A.try_add(x, 0)

        B = SetState()
        for x in order_B:
            if len(B.elements) >= max_b:
                break
            B.try_add(x, A.diffs_mask)

        return A, B

    def polish(A, B, max_time):
        """
        Aggressive local search: add, remove, swap elements to maximize |A|+|B|.
        """
        improved = True
        while improved and time.time() < max_time:
            improved = False

            # Try to add to A
            if len(A.elements) < MAX_SIZE_A:
                candidates = [x for x in range(1, N + 1) if x not in A.element_set]
                random.shuffle(candidates)
                for x in candidates:
                    if A.try_add(x, B.diffs_mask):
                        improved = True
                        break
                if improved:
                    continue

            # Try to add to B
            if len(B.elements) < MAX_SIZE_B:
                candidates = [x for x in range(1, N + 1) if x not in B.element_set]
                random.shuffle(candidates)
                for x in candidates:
                    if B.try_add(x, A.diffs_mask):
                        improved = True
                        break
                if improved:
                    continue

            # Try swap in A: remove one, add one different
            if len(A.elements) > 0:
                idx = random.randint(0, len(A.elements) - 1)
                x_removed = A.elements[idx]
                A.remove(x_removed)
                candidates = [x for x in range(1, N + 1) if x not in A.element_set and x != x_removed]
                random.shuffle(candidates)
                found = False
                for x in candidates:
                    if A.try_add(x, B.diffs_mask):
                        found = True
                        improved = True
                        break
                if not found:
                    # Revert
                    A.try_add(x_removed, B.diffs_mask)
                if improved:
                    continue

            # Try swap in B: remove one, add one different
            if len(B.elements) > 0:
                idx = random.randint(0, len(B.elements) - 1)
                x_removed = B.elements[idx]
                B.remove(x_removed)
                candidates = [x for x in range(1, N + 1) if x not in B.element_set and x != x_removed]
                random.shuffle(candidates)
                found = False
                for x in candidates:
                    if B.try_add(x, A.diffs_mask):
                        found = True
                        improved = True
                        break
                if not found:
                    # Revert
                    B.try_add(x_removed, A.diffs_mask)
                if improved:
                    continue

            # Try remove from A (if it helps B)
            if len(A.elements) > 1:
                idx = random.randint(0, len(A.elements) - 1)
                x_removed = A.elements[idx]
                A.remove(x_removed)
                # Try to add to B
                candidates = [x for x in range(1, N + 1) if x not in B.element_set]
                random.shuffle(candidates)
                found = False
                for x in candidates:
                    if B.try_add(x, A.diffs_mask):
                        found = True
                        improved = True
                        break
                if not found:
                    # Revert A
                    A.try_add(x_removed, B.diffs_mask)
                if improved:
                    continue

            # Try remove from B (if it helps A)
            if len(B.elements) > 1:
                idx = random.randint(0, len(B.elements) - 1)
                x_removed = B.elements[idx]
                B.remove(x_removed)
                # Try to add to A
                candidates = [x for x in range(1, N + 1) if x not in A.element_set]
                random.shuffle(candidates)
                found = False
                for x in candidates:
                    if A.try_add(x, B.diffs_mask):
                        found = True
                        improved = True
                        break
                if not found:
                    # Revert B
                    B.try_add(x_removed, A.diffs_mask)
                if improved:
                    continue

        return A, B

    # Main search loop
    while time.time() < deadline:
        # Random restart
        order_A = list(range(1, N + 1))
        random.shuffle(order_A)
        order_B = list(range(1, N + 1))
        random.shuffle(order_B)

        # Alternate between interleaved and sequential builds
        if random.random() < 0.5:
            A, B = build_interleaved(order_A, order_B, MAX_SIZE_A, MAX_SIZE_B)
        else:
            A, B = build_sequential(order_A, order_B, MAX_SIZE_A, MAX_SIZE_B)

        current_size = len(A.elements) + len(B.elements)

        # Polish
        if current_size > best_size or (current_size >= best_size - 2 and random.random() < 0.5):
            polish_time = min(5.0, deadline - time.time() - 0.5)
            if polish_time > 0.5:
                A, B = polish(A, B, start_time + polish_time)
                current_size = len(A.elements) + len(B.elements)

            if current_size > best_size:
                best_size = current_size
                best_A = A.elements[:]
                best_B = B.elements[:]

        if best_size >= 32:
            break

    # Fallback
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
