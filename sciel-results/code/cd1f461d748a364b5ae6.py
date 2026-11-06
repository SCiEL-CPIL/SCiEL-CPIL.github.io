import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1
DEADLINE_SECONDS = 82.0  # safety margin before 90s kill


def _build_subset_trans(trans):
    size = 1 << N
    next_subset = [0] * size
    for mask in range(1, size):
        lsb = mask & (-mask)
        i = lsb.bit_length() - 1
        rest = mask ^ lsb
        next_subset[mask] = next_subset[rest] | (1 << trans[i])
    return next_subset


def compute_threshold(trans_a, trans_b):
    """BFS on subset automaton; returns exact shortest reset word length, 0 if not synchronizing."""
    next_a = _build_subset_trans(trans_a)
    next_b = _build_subset_trans(trans_b)
    size = 1 << N
    dist = [-1] * size
    dist[FULL_MASK] = 0
    queue = deque([FULL_MASK])
    while queue:
        cur = queue.popleft()
        if cur & (cur - 1) == 0:
            return dist[cur]
        nxt_a = next_a[cur]
        if dist[nxt_a] == -1:
            dist[nxt_a] = dist[cur] + 1
            queue.append(nxt_a)
        nxt_b = next_b[cur]
        if dist[nxt_b] == -1:
            dist[nxt_b] = dist[cur] + 1
            queue.append(nxt_b)
    return 0


def _construct_ping_pong(n):
    """
    Construct a ping-pong type automaton:
    - a: cycle (0->1->2->...->n-1->0)
    - b: maps all states to 0 except state 0 maps to n-1
    This gives a known high threshold.
    """
    a = [(i + 1) % n for i in range(n)]
    b = [0] * n
    b[0] = n - 1
    return a, b


def _construct_cycle_with_escape(n, e, t):
    """
    Cycle a=(i+1)%n, b maps all to 0 except e->t.
    """
    a = [(i + 1) % n for i in range(n)]
    b = [0] * n
    b[e] = t
    return a, b


def _beam_search(a_init, b_init, deadline, best_holder, beam_width=20, max_iter=2000):
    """
    Beam search over transition modifications.
    State: (score, a_tuple, b_tuple)
    Mutations: change one entry in a or b to a random target.
    """
    n = len(a_init)
    rng = random.Random(int(time.time() * 1000) % 100000)
    
    # Initialize beam with the starting configuration
    start_score = compute_threshold(a_init, b_init)
    if start_score == 0:
        return
    
    beam = [(start_score, tuple(a_init), tuple(b_init))]
    
    if start_score > best_holder[0]:
        best_holder[0] = start_score
        best_holder[1] = list(a_init)
        best_holder[2] = list(b_init)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        candidates = []
        for score, a_tup, b_tup in beam:
            a_list = list(a_tup)
            b_list = list(b_tup)
            
            # Generate several mutations
            for _ in range(3):
                mut_a = rng.random() < 0.3
                if mut_a:
                    idx = rng.randint(0, n - 1)
                    old = a_list[idx]
                    new_val = rng.randint(0, n - 1)
                    if new_val == old:
                        continue
                    a_list[idx] = new_val
                    new_score = compute_threshold(a_list, b_list)
                    if new_score > 0:
                        candidates.append((new_score, tuple(a_list), tuple(b_list)))
                    a_list[idx] = old
                else:
                    idx = rng.randint(0, n - 1)
                    old = b_list[idx]
                    new_val = rng.randint(0, n - 1)
                    if new_val == old:
                        continue
                    b_list[idx] = new_val
                    new_score = compute_threshold(a_list, b_list)
                    if new_score > 0:
                        candidates.append((new_score, tuple(a_list), tuple(b_list)))
                    b_list[idx] = old
        
        if not candidates:
            break
        
        # Keep top beam_width candidates, deduplicate
        candidates.sort(key=lambda x: -x[0])
        seen = set()
        new_beam = []
        for c in candidates:
            key = (c[1], c[2])
            if key not in seen:
                seen.add(key)
                new_beam.append(c)
            if len(new_beam) >= beam_width:
                break
        
        beam = new_beam
        
        # Update best
        top = beam[0]
        if top[0] > best_holder[0]:
            best_holder[0] = top[0]
            best_holder[1] = list(top[1])
            best_holder[2] = list(top[2])


def _local_search_from(a_init, b_init, deadline, best_holder):
    """
    Local search: repeatedly modify one transition, keep if improves.
    """
    n = len(a_init)
    rng = random.Random(int(time.time() * 7777) % 100000)
    
    cur_a = list(a_init)
    cur_b = list(b_init)
    cur_score = compute_threshold(cur_a, cur_b)
    if cur_score == 0:
        return
    
    if cur_score > best_holder[0]:
        best_holder[0] = cur_score
        best_holder[1] = cur_a[:]
        best_holder[2] = cur_b[:]
    
    stagnation = 0
    max_stagnation = 50
    
    while time.time() < deadline and stagnation < max_stagnation:
        mut_a = rng.random() < 0.25
        idx = rng.randint(0, n - 1)
        
        if mut_a:
            old = cur_a[idx]
            new_val = rng.randint(0, n - 1)
            if new_val == old:
                continue
            cur_a[idx] = new_val
            new_score = compute_threshold(cur_a, cur_b)
            if new_score > cur_score:
                cur_score = new_score
                stagnation = 0
                if cur_score > best_holder[0]:
                    best_holder[0] = cur_score
                    best_holder[1] = cur_a[:]
                    best_holder[2] = cur_b[:]
            else:
                cur_a[idx] = old
                stagnation += 1
        else:
            old = cur_b[idx]
            new_val = rng.randint(0, n - 1)
            if new_val == old:
                continue
            cur_b[idx] = new_val
            new_score = compute_threshold(cur_a, cur_b)
            if new_score > cur_score:
                cur_score = new_score
                stagnation = 0
                if cur_score > best_holder[0]:
                    best_holder[0] = cur_score
                    best_holder[1] = cur_a[:]
                    best_holder[2] = cur_b[:]
            else:
                cur_b[idx] = old
                stagnation += 1


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best_holder = [0, None, None]  # [score, a, b]
    
    # Phase 1: Try several known high-threshold constructions
    constructions = []
    
    # Ping-pong
    a_pp, b_pp = _construct_ping_pong(N)
    constructions.append((a_pp, b_pp))
    
    # Cycle with various escapes
    for e in range(N):
        for t in range(N):
            if t == e:
                continue
            a_ce, b_ce = _construct_cycle_with_escape(N, e, t)
            constructions.append((a_ce, b_ce))
    
    # Evaluate all constructions
    for a, b in constructions:
        if time.time() > deadline - 30.0:
            break
        score = compute_threshold(a, b)
        if score > best_holder[0]:
            best_holder[0] = score
            best_holder[1] = a[:]
            best_holder[2] = b[:]
    
    # Phase 2: Beam search from the best construction
    if best_holder[1] is not None and time.time() < deadline - 15.0:
        _beam_search(best_holder[1], best_holder[2], deadline - 10.0, best_holder, beam_width=15, max_iter=3000)
    
    # Phase 3: Local search from best
    if best_holder[1] is not None and time.time() < deadline - 5.0:
        _local_search_from(best_holder[1], best_holder[2], deadline - 2.0, best_holder)
    
    # Fallback: guaranteed synchronizing automaton
    if best_holder[1] is None or best_holder[0] == 0:
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        b[0] = N - 1
        return [a, b]
    
    return [best_holder[1], best_holder[2]]


if __name__ == "__main__":
    a, b = construct()
    print("Best score:", compute_threshold(a, b))
    print("A:", a)
    print("B:", b)
