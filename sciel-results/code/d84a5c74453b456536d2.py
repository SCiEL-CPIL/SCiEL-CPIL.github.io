"""
Search for a 12-state binary automaton with a long shortest reset word.
Uses randomized local search over transition structures, evaluating each
candidate with exact BFS on the subset automaton.
"""
import time
import random
from collections import deque

N = 12
DEADLINE_FRACTION = 0.85  # safety margin

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 90 * DEADLINE_FRACTION

    def apply_letter(subset, trans, letter):
        """Apply a letter to a frozenset of states."""
        new_set = set()
        for s in subset:
            new_set.add(trans[letter][s])
        return frozenset(new_set)

    def compute_threshold(trans):
        """
        Compute exact shortest reset word length via BFS on subset automaton.
        Returns -1 if not synchronizing.
        """
        full_set = frozenset(range(N))
        visited = {full_set: 0}
        queue = deque([full_set])

        while queue:
            current = queue.popleft()
            dist = visited[current]

            if len(current) == 1:
                return dist

            for letter in range(2):
                nxt = apply_letter(current, trans, letter)
                if nxt not in visited:
                    visited[nxt] = dist + 1
                    queue.append(nxt)

        return -1

    def random_automaton():
        """Generate a random complete binary deterministic automaton."""
        trans = [[random.randint(0, N - 1) for _ in range(N)] for _ in range(2)]
        return trans

    def mutate(trans):
        """Randomly change a few transitions."""
        new_trans = [row[:] for row in trans]
        n_mutations = random.randint(1, 3)
        for _ in range(n_mutations):
            letter = random.randint(0, 1)
            state = random.randint(0, N - 1)
            new_trans[letter][state] = random.randint(0, N - 1)
        return new_trans

    def make_cycle_with_defect():
        """
        Create a cycle automaton with a controlled defect.
        Letter 0: cycle (i -> i+1 mod N)
        Letter 1: mostly maps to a fixed state, with some defects.
        """
        trans = [[(i + 1) % N for i in range(N)], [0] * N]
        # Introduce 1-3 defects in letter 1
        n_defects = random.randint(1, 3)
        for _ in range(n_defects):
            state = random.randint(0, N - 1)
            trans[1][state] = random.randint(0, N - 1)
        return trans

    def make_permutation_with_defects():
        """
        Letter 0: a random permutation (cycle structure).
        Letter 1: mostly maps to one state, with defects.
        """
        # Random permutation for letter 0
        perm = list(range(N))
        random.shuffle(perm)
        trans = [perm[:], [0] * N]
        # Add defects to letter 1
        n_defects = random.randint(1, 4)
        for _ in range(n_defects):
            state = random.randint(0, N - 1)
            trans[1][state] = random.randint(0, N - 1)
        return trans

    def make_two_structures():
        """
        Letter 0: cycle.
        Letter 1: another permutation with some non-permutation elements.
        """
        trans = [[(i + 1) % N for i in range(N)], []]
        # Letter 1: start as a permutation, then break it
        perm = list(range(N))
        random.shuffle(perm)
        trans[1] = perm[:]
        # Break permutation: make some states map to same target
        n_breaks = random.randint(1, 3)
        for _ in range(n_breaks):
            state = random.randint(0, N - 1)
            target = random.randint(0, N - 1)
            trans[1][state] = target
        return trans

    best_trans = None
    best_score = -1

    # Initial candidates from structured families
    candidates = [
        make_cycle_with_defect(),
        make_cycle_with_defect(),
        make_permutation_with_defects(),
        make_permutation_with_defects(),
        make_two_structures(),
        make_two_structures(),
        random_automaton(),
        random_automaton(),
    ]

    # Evaluate initial candidates
    for trans in candidates:
        score = compute_threshold(trans)
        if score > best_score:
            best_score = score
            best_trans = [row[:] for row in trans]

    # Local search: mutate the best and explore
    current = [row[:] for row in best_trans]
    no_improvement_count = 0
    max_no_improvement = 50

    while time.time() < deadline:
        # Check deadline with margin
        if time.time() > deadline - 2:
            break

        # Generate a candidate
        if random.random() < 0.3:
            # Occasional restart from structured family
            if random.random() < 0.5:
                candidate = make_cycle_with_defect()
            elif random.random() < 0.5:
                candidate = make_permutation_with_defects()
            else:
                candidate = make_two_structures()
        else:
            # Mutate current best
            candidate = mutate(current)

        score = compute_threshold(candidate)

        if score > best_score:
            best_score = score
            best_trans = [row[:] for row in candidate]
            current = [row[:] for row in candidate]
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Reset to a fresh structured candidate
                if random.random() < 0.5:
                    candidate = make_cycle_with_defect()
                else:
                    candidate = make_permutation_with_defects()
                current = [row[:] for row in candidate]
                no_improvement_count = 0
                max_no_improvement = random.randint(30, 80)

    if best_trans is None:
        # Fallback: simple synchronizing automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]

    return best_trans
# EVOLVE-BLOCK-END
