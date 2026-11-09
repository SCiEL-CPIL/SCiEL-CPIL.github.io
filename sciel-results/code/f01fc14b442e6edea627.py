"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import deque

N = 12

# EVOLVE-BLOCK-START
def _compute_threshold(trans):
    """BFS on subset automaton. Returns (threshold, word) or (inf, None)."""
    full = tuple(range(N))
    visited = {full: ""}
    queue = deque([full])
    while queue:
        s = queue.popleft()
        w = visited[s]
        if len(s) == 1:
            return len(w), w
        for a in range(2):
            img = tuple(sorted(set(trans[a][x] for x in s)))
            if img not in visited:
                visited[img] = w + str(a)
                queue.append(img)
    return float('inf'), None


def _pawlik_variant(offset):
    """Generate a Pawlik-type automaton with a parameter."""
    sigma0 = [(i + 1) % N for i in range(N)]
    # sigma1: fix 0, and for i>0 map to (i - 1 + offset) % N, but ensure bijective on 1..N-1
    sigma1 = [0]
    for i in range(1, N):
        sigma1.append((i - 1 + offset) % (N - 1) + 1)
    return [sigma0, sigma1]


def construct():
    deadline = time.time() + 85.0
    
    best_trans = None
    best_thresh = 0
    best_word = ""
    
    # Start with the standard Pawlik automaton
    sigma0 = [(i + 1) % N for i in range(N)]
    sigma1 = [0 if i == 0 else i - 1 for i in range(N)]
    trans = [sigma0, sigma1]
    thresh, word = _compute_threshold(trans)
    if thresh < float('inf'):
        best_trans = trans
        best_thresh = thresh
        best_word = word
    
    # Try a few other constructions
    for offset in range(1, 5):
        if time.time() > deadline:
            break
        trans = _pawlik_variant(offset)
        thresh, word = _compute_threshold(trans)
        if thresh < float('inf') and thresh > best_thresh:
            best_trans = trans
            best_thresh = thresh
            best_word = word
    
    # Try random perturbation of sigma1
    import random
    rng = random.Random(42)
    for _ in range(200):
        if time.time() > deadline:
            break
        trans = [sigma0[:], sigma1[:]]
        # Randomly permute the images of states 1..N-1 under sigma1
        # while keeping sigma1[0] = 0
        images = list(range(1, N))
        rng.shuffle(images)
        trans[1] = [0] + images
        thresh, word = _compute_threshold(trans)
        if thresh < float('inf') and thresh > best_thresh:
            best_trans = trans
            best_thresh = thresh
            best_word = word
    
    if best_trans is None:
        # Fallback: simple synchronizing automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    return best_trans
# EVOLVE-BLOCK-END
