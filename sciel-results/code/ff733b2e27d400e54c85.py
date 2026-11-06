"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

N = 12

# EVOLVE-BLOCK-START

def _reset_length(trans_a, trans_b):
    """Compute the exact shortest reset word length via BFS on subsets.
    Returns (length, word) or (None, None) if not synchronizing."""
    start = frozenset(range(N))
    if len(start) == 1:
        return 0, ""
    # Map subsets to index for quick lookup
    visited = {start: 0}
    parent = {}  # (subset, letter) -> prev_subset
    queue = deque([(start, "")])
    # BFS
    while queue:
        subset, word = queue.popleft()
        for letter, trans in zip('ab', (trans_a, trans_b)):
            new_subset = frozenset(trans[s] for s in subset)
            if len(new_subset) == 1:
                return len(word) + 1, word + letter
            if new_subset not in visited:
                visited[new_subset] = len(word) + 1
                parent[(new_subset, letter)] = (subset, word)
                queue.append((new_subset, word + letter))
    return None, None


def _pinwheel(n, k):
    """
    Pinwheel automaton: states 0..n-1.
    Letter a: i -> (i+1) % n (cycle)
    Letter b: 0 -> 0, i -> i-1 for i=1..k-1, i -> 0 for i=k..n-1
    This is a known family with reset length n-1.
    """
    a = [(i + 1) % n for i in range(n)]
    b = [0] * n
    for i in range(1, k):
        b[i] = i - 1
    # for i >= k, b[i] = 0 already
    return a, b


def _pinwheel_variant(n, k, offset=0):
    """
    Variant pinwheel with a shifted 'reset' region.
    a: cycle
    b: for states in a contiguous block of size k, they step towards one end;
       the rest map to a fixed point.
    """
    a = [(i + 1) % n for i in range(n)]
    b = [0] * n
    # Block from (n-k) to n-1 wraps around? Let's do: states k..n-1 map down to k
    # Actually let's try: b[i] = i-1 for i in [1, k], b[i] = 0 for i > k
    # That's the same as _pinwheel.
    # Let's try a different structure:
    # a: cycle
    # b: 0->0, 1->0, 2->1, ..., k->k-1, k+1->0, ..., n-1->0
    # This is just pinwheel with k.
    # Alternative: "rank-2" style
    # Let's try: a: cycle, b: splits the cycle into two parts that collapse
    pass
    return a, b


def _rank2_cycle(n, k):
    """
    Rank-2 cycle construction.
    a: 0->1->2->...->n-1->0 (cycle)
    b: 0->0, 1->0, 2->1, 3->1, 4->2, 5->2, ..., pattern: b[i] = i//2 for i>=1, b[0]=0
    Actually let's use the standard rank-2:
    b[i] = i//2 for all i. Then b(0)=0, b(1)=0, b(2)=1, b(3)=1, etc.
    This has reset length roughly 2*(n-1) - something? Let's just compute.
    """
    a = [(i + 1) % n for i in range(n)]
    b = [i // 2 for i in range(n)]
    return a, b


def _binary_tree_collapse(n):
    """
    a: cycle
    b: i -> i//2 (floor division), which collapses the state space.
    """
    a = [(i + 1) % n for i in range(n)]
    b = [i // 2 for i in range(n)]
    return a, b


def _improved_pinwheel(n):
    """
    Try pinwheel with k = n-1 (maximum).
    """
    return _pinwheel(n, n - 1)


def _construct_best():
    """
    Try several known constructions and return the one with the longest reset word.
    """
    candidates = []
    
    # Pinwheel with various k
    for k in range(1, n := N):
        a, b = _pinwheel(N, k)
        length, word = _reset_length(a, b)
        if length is not None:
            candidates.append((length, a, b, word))
    
    # Rank-2 / binary tree collapse
    a, b = _binary_tree_collapse(N)
    length, word = _reset_length(a, b)
    if length is not None:
        candidates.append((length, a, b, word))
    
    # Simple cycle + map to zero (original)
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    length, word = _reset_length(a, b)
    if length is not None:
        candidates.append((length, a, b, word))
    
    if not candidates:
        # Fallback: should not happen
        return [(i + 1) % N for i in range(N)], [0] * N
    
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_length, best_a, best_b, best_word = candidates[0]
    return best_a, best_b


def construct():
    deadline = time.time() + 80  # safety margin before 90s limit
    
    # Start with the best known construction
    best_a, best_b = _construct_best()
    best_length, _ = _reset_length(best_a, best_b)
    
    # Try to improve via local search if time permits
    start_time = time.time()
    rng = random.Random(42)
    
    # Local search: perturb the best construction
    current_a, current_b = list(best_a), list(best_b)
    current_length = best_length
    
    while time.time() < deadline:
        # Pick a random state and letter to perturb
        letter = rng.random() < 0.5
        state = rng.randint(0, N - 1)
        new_dest = rng.randint(0, N - 1)
        
        if letter:
            old = current_b[state]
            current_b[state] = new_dest
            a, b = current_a, current_b
        else:
            old = current_a[state]
            current_a[state] = new_dest
            a, b = current_a, current_b
        
        length, word = _reset_length(a, b)
        
        if length is not None and length > current_length:
            current_length = length
            best_a, best_b = list(a), list(b)
        else:
            # Revert
            if letter:
                current_b[state] = old
            else:
                current_a[state] = old
        
        # Check deadline
        if time.time() >= deadline:
            break
    
    return best_a, best_b

# EVOLVE-BLOCK-END
