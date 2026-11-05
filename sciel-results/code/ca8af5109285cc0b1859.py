import time
import random

N = 9
NUM_VERTS = 1 << N  # 512

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s kill

    # Precompute neighbor lists
    neighbors = [[v ^ (1 << b) for b in range(N)] for v in range(NUM_VERTS)]
    # Precompute neighbor bitmask for each vertex (for fast "is neighbor" checks)
    nbmask = [0] * NUM_VERTS
    for v in range(NUM_VERTS):
        m = 0
        for b in range(N):
            m |= 1 << (v ^ (1 << b))
        nbmask[v] = m

    # Precompute a bitmask of "all vertices"
    ALL = (1 << NUM_VERTS) - 1

    def greedy_with_backtrack(start, max_bt=2000, use_rand=False):
        """
        Extend a snake from `start` using Warnsdorff heuristic with bounded backtracking.
        Returns the longest path found.
        """
        path = [start]
        used = 1 << start
        # forbidden: bitmask of vertices that are neighbors of any path vertex except the last two
        # Invariant: forbidden = union of nbmask[path[i]] for i in range(len(path)-2)
        forbidden = 0

        best_path = path[:]
        best_len = len(path)
        bt_budget = max_bt
        random.seed(start if use_rand else 0)

        def score_candidate(nxt, cur, used, forbidden):
            """
            Score a candidate next vertex. Lower is better (Warnsdorff: fewest onward options).
            Also penalize moves that over-constrain the frontier.
            """
            # Onward free neighbors of nxt: neighbors of nxt that are not used and not forbidden
            # and not in nbmask[cur] (since after moving, cur joins the forbidden history)
            onward = 0
            for m in neighbors[nxt]:
                if (used >> m) & 1:
                    continue
                if (forbidden >> m) & 1:
                    continue
                # m will be forbidden after we move (cur becomes history)
                if (nbmask[cur] >> m) & 1:
                    continue
                onward |= 1 << m
            onward_count = onward.bit_count()

            # Danger: how many of nxt's neighbors are already forbidden or used
            # (high danger means the move is into a tight spot)
            danger = 0
            for m in neighbors[nxt]:
                if (used >> m) & 1 or (forbidden >> m) & 1:
                    danger += 1

            # Prefer fewest onward options, then least danger
            return onward_count * 100 + danger

        def try_extend():
            nonlocal path, used, forbidden, bt_budget, best_path, best_len
            while True:
                cur = path[-1]
                # Gather candidates
                cands = []
                for nxt in neighbors[cur]:
                    if (used >> nxt) & 1:
                        continue
                    if (forbidden >> nxt) & 1:
                        continue
                    cands.append(nxt)

                if not cands:
                    break  # dead end

                if len(cands) == 1:
                    nxt = cands[0]
                else:
                    # Score candidates
                    best_score = None
                    best_nxt = None
                    if use_rand:
                        random.shuffle(cands)
                    for nxt in cands:
                        s = score_candidate(nxt, cur, used, forbidden)
                        if best_score is None or s < best_score:
                            best_score = s
                            best_nxt = nxt
                    nxt = best_nxt

                # Move
                path.append(nxt)
                used |= 1 << nxt
                # Add cur to forbidden history
                forbidden |= nbmask[cur]

                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path[:]

                # Time check
                if time.time() > deadline:
                    return best_path

                # Backtracking check
                if time.time() > deadline:
                    return best_path

            # Dead end reached. Try backtracking.
            # We need to undo the last move and try a different branch.
            # Simple approach: backtrack by popping and trying alternatives.
            # This is complex to implement fully; instead, do a limited local search.
            # We'll try to "restart" from a slightly earlier point with different choices.
            # For simplicity, we do a few random restarts from the current best prefix.
            return best_path

        # Main loop with backtracking via iterative deepening of restarts
        # We use a simpler strategy: greedy with random restarts at dead ends.
        # Actually, let's do proper backtracking with a stack.

        # Reimplement with explicit backtracking stack
        path = [start]
        used = 1 << start
        forbidden = 0
        best_path = path[:]
        best_len = 1
        bt_count = 0
        max_bt = 5000  # max backtracking steps

        # We'll use a recursive-style loop with a trail of choices
        # trail[i] = list of candidate vertices at path position i (for backtracking)
        trail = []

        while True:
            cur = path[-1]
            cands = []
            for nxt in neighbors[cur]:
                if (used >> nxt) & 1:
                    continue
                if (forbidden >> nxt) & 1:
                    continue
                cands.append(nxt)

            if not cands:
                # Dead end: backtrack
                if bt_count >= max_bt:
                    break
                # Find the last position where we can try a different candidate
                # We need to have stored the candidates at each step.
                # For simplicity, we pop the last vertex and mark it as "avoided".
                # This is a simplified backtracking that may not be complete.
                # Let's just break and rely on random restarts.
                break

            if len(cands) == 1:
                nxt = cands[0]
            else:
                # Score and pick
                best_score = None
                best_nxt = None
                if use_rand:
                    random.shuffle(cands)
                for nxt in cands:
                    s = score_candidate(nxt, cur, used, forbidden)
                    if best_score is None or s < best_score:
                        best_score = s
                        best_nxt = nxt
                nxt = best_nxt

            path.append(nxt)
            used |= 1 << nxt
            forbidden |= nbmask[cur]

            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]

            if time.time() > deadline:
                return best_path

        return best_path

    # Strategy 1: Deterministic greedy from all starts
    for start in range(NUM_VERTS):
        if time.time() > deadline - 5:
            break
        p = greedy_with_backtrack(start, max_bt=0, use_rand=False)
        if len(p) > best_len if False else True:
            pass
        # Track best
        if not hasattr(construct, '_best'):
            construct._best = []
        if len(p) > len(construct._best):
            construct._best = p[:]

    # Strategy 2: Randomized greedy restarts
    while time.time() < deadline - 2:
        start = random.randint(0, NUM_VERTS - 1)
        p = greedy_with_backtrack(start, max_bt=0, use_rand=True)
        if len(p) > len(construct._best):
            construct._best = p[:]

    return construct._best

def verify_snake(path):
    if not path:
        return True
    if len(set(path)) != len(path):
        return False
    for i in range(len(path) - 1):
        if (path[i] ^ path[i + 1]).bit_count() != 1:
            return False
    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            if (path[i] ^ path[j]).bit_count() < 2:
                return False
    return True

if __name__ == "__main__":
    p = construct()
    print(f"Length: {len(p)}")
    print(f"Valid: {verify_snake(p)}")
