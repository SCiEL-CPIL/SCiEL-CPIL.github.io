import time
import random
import sys
import numpy as np

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0  # Safety margin for 150s limit


def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    
    Strategy: Multi-start with hybrid approach:
    1. Greedy bidirectional growth with Warnsdorff's rule
    2. Local search improvement: try to extend stuck snakes by backtracking
    3. Diversity through different random seeds and parameters
    """
    start_time = time.time()
    best_path = None
    best_len = 0

    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))

    def grow_snake(start_vertex, max_steps=500, randomness=0.1, use_backtrack=False):
        """
        Grow a snake from start_vertex using Warnsdorff's rule.
        Returns the path as a list.
        """
        path = [start_vertex]
        used = [False] * MAX_VERTS
        used[start_vertex] = True

        # deg_in_path[v] = number of neighbors of v that are in the current path
        deg_in_path = [0] * MAX_VERTS

        for w in neighbors[start_vertex]:
            deg_in_path[w] += 1

        steps = 0
        while steps < max_steps:
            # Check time
            if time.time() - start_time > DEADLINE:
                break

            left_end = path[0]
            right_end = path[-1]

            # Get valid moves for left end
            valid_left = []
            for v in neighbors[left_end]:
                if not used[v] and deg_in_path[v] == 1:
                    valid_left.append(v)

            # Get valid moves for right end
            valid_right = []
            for v in neighbors[right_end]:
                if not used[v] and deg_in_path[v] == 1:
                    valid_right.append(v)

            if not valid_left and not valid_right:
                break

            # Count onward moves for scoring
            def count_onward(candidate):
                count = 0
                for w in neighbors[candidate]:
                    if not used[w] and deg_in_path[w] == 0:
                        count += 1
                return count

            # Score candidates
            best_left = None
            min_left = float('inf')
            for v in valid_left:
                score = count_onward(v) + random.random() * randomness
                if score < min_left:
                    min_left = score
                    best_left = v

            best_right = None
            min_right = float('inf')
            for v in valid_right:
                score = count_onward(v) + random.random() * randomness
                if score < min_right:
                    min_right = score
                    best_right = v

            # Choose which side to extend
            if not valid_left:
                path.append(best_right)
                used[best_right] = True
                for w in neighbors[best_right]:
                    deg_in_path[w] += 1
            elif not valid_right:
                path.insert(0, best_left)
                used[best_left] = True
                for w in neighbors[best_left]:
                    deg_in_path[w] += 1
            else:
                # Choose side with tighter constraint (fewer onward moves)
                if min_left < min_right:
                    path.insert(0, best_left)
                    used[best_left] = True
                    for w in neighbors[best_left]:
                        deg_in_path[w] += 1
                elif min_right < min_left:
                    path.append(best_right)
                    used[best_right] = True
                    for w in neighbors[best_right]:
                        deg_in_path[w] += 1
                else:
                    if random.random() < 0.5:
                        path.insert(0, best_left)
                        used[best_left] = True
                        for w in neighbors[best_left]:
                            deg_in_path[w] += 1
                    else:
                        path.append(best_right)
                        used[best_right] = True
                        for w in neighbors[best_right]:
                            deg_in_path[w] += 1

            steps += 1

        return path

    def try_extend_with_backtrack(path, max_backtracks=20):
        """
        Try to extend a stuck snake by backtracking a few steps and trying different branches.
        Returns the extended path if successful, otherwise returns the original path.
        """
        if time.time() - start_time > DEADLINE:
            return path

        original_len = len(path)
        best = list(path)

        # Try backtracking from different positions
        for backtrack_depth in range(1, min(max_backtracks, len(path) - 2) + 1):
            if time.time() - start_time > DEADLINE:
                break

            # Try removing the last 'backtrack_depth' vertices and regrowing
            truncated = path[:-backtrack_depth] if backtrack_depth > 0 else list(path)

            if len(truncated) < 2:
                continue

            # Regrow from the truncated path
            new_path = list(truncated)
            used = [False] * MAX_VERTS
            for v in new_path:
                used[v] = True

            deg_in_path = [0] * MAX_VERTS
            for v in new_path:
                for w in neighbors[v]:
                    deg_in_path[w] += 1

            # Try to extend from both ends
            extended = False
            for _ in range(10):  # Try up to 10 extension steps
                left_end = new_path[0]
                right_end = new_path[-1]

                valid_left = [v for v in neighbors[left_end] if not used[v] and deg_in_path[v] == 1]
                valid_right = [v for v in neighbors[right_end] if not used[v] and deg_in_path[v] == 1]

                if not valid_left and not valid_right:
                    break

                # Choose a random valid move
                if valid_left and (not valid_right or random.random() < 0.5):
                    v = random.choice(valid_left)
                    new_path.insert(0, v)
                    used[v] = True
                    for w in neighbors[v]:
                        deg_in_path[w] += 1
                    extended = True
                elif valid_right:
                    v = random.choice(valid_right)
                    new_path.append(v)
                    used[v] = True
                    for w in neighbors[v]:
                        deg_in_path[w] += 1
                    extended = True
                else:
                    break

            if len(new_path) > len(best):
                best = new_path

        return best

    # Multi-start search
    attempts = 0
    while time.time() - start_time < DEADLINE:
        attempts += 1

        # Vary randomness for diversity
        if attempts % 5 == 0:
            randomness = random.choice([0.0, 0.05, 0.1, 0.2, 0.5])
        else:
            randomness = random.uniform(0.0, 0.15)

        # Random start vertex
        v0 = random.randint(0, MAX_VERTS - 1)

        # Grow snake
        p = grow_snake(v0, max_steps=600, randomness=randomness)

        # Try to extend with backtracking if we have time
        if time.time() - start_time < DEADLINE - 5.0:
            p = try_extend_with_backtrack(p, max_backtracks=15)

        if len(p) > best_len:
            best_len = len(p)
            best_path = p

        # Early termination if we've found a very long snake
        if best_len >= 200:
            break

    if best_path is None:
        return [0]

    return best_path


# EVOLVE-BLOCK-END
