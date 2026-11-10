import random
import time
import numpy as np

N = 256

def _build_aps():
    """Build list of all 4-term arithmetic progressions in 0..N-1."""
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

def _build_ap_membership(aps):
    """For each position, list of (ap_index, position_in_ap) for APs containing that position."""
    membership = [[] for _ in range(N)]
    for idx, (a, b, c, d) in enumerate(aps):
        membership[a].append((idx, 0))
        membership[b].append((idx, 1))
        membership[c].append((idx, 2))
        membership[d].append((idx, 3))
    return membership

def _count_mono(aps, colors):
    """Count monochromatic 4-APs."""
    count = 0
    for a, b, c, d in aps:
        if colors[a] == colors[b] == colors[c] == colors[d]:
            count += 1
    return count

def _local_search(aps, membership, colors, max_passes=200):
    """Greedy local search with incremental delta scoring."""
    colors = list(colors)
    n = len(colors)
    ap_count = len(aps)
    
    # Track which APs are currently monochromatic
    ap_mono = [False] * ap_count
    for idx, (a, b, c, d) in enumerate(aps):
        if colors[a] == colors[b] == colors[c] == colors[d]:
            ap_mono[idx] = True
    
    current_mono = sum(ap_mono)
    
    for _ in range(max_passes):
        improved = False
        for i in range(n):
            new_color = 1 - colors[i]
            delta = 0
            for (ap_idx, pos_in_ap) in membership[i]:
                a, b, c, d = aps[ap_idx]
                others = [aps[ap_idx][j] for j in range(4) if j != pos_in_ap]
                if ap_mono[ap_idx]:
                    delta -= 1
                else:
                    if all(colors[o] == new_color for o in others):
                        delta += 1
            if delta < 0:
                colors[i] = new_color
                current_mono += delta
                for (ap_idx, pos_in_ap) in membership[i]:
                    a, b, c, d = aps[ap_idx]
                    ap_mono[ap_idx] = (colors[a] == colors[b] == colors[c] == colors[d])
                improved = True
        if not improved:
            break
    
    return colors, current_mono

def _simulated_annealing(aps, membership, colors, start_temp=10.0, end_temp=0.1, max_iter=2000):
    """Simulated annealing with incremental delta scoring."""
    colors = list(colors)
    n = len(colors)
    ap_count = len(aps)
    
    ap_mono = [False] * ap_count
    for idx, (a, b, c, d) in enumerate(aps):
        if colors[a] == colors[b] == colors[c] == colors[d]:
            ap_mono[idx] = True
    
    current_mono = sum(ap_mono)
    best_mono = current_mono
    best_colors = list(colors)
    
    for it in range(max_iter):
        temp = start_temp * (end_temp / start_temp) ** (it / max_iter)
        i = random.randint(0, n - 1)
        new_color = 1 - colors[i]
        
        delta = 0
        for (ap_idx, pos_in_ap) in membership[i]:
            a, b, c, d = aps[ap_idx]
            others = [aps[ap_idx][j] for j in range(4) if j != pos_in_ap]
            if ap_mono[ap_idx]:
                delta -= 1
            else:
                if all(colors[o] == new_color for o in others):
                    delta += 1
        
        if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
            colors[i] = new_color
            current_mono += delta
            for (ap_idx, pos_in_ap) in membership[i]:
                a, b, c, d = aps[ap_idx]
                ap_mono[ap_idx] = (colors[a] == colors[b] == colors[c] == colors[d])
            if current_mono < best_mono:
                best_mono = current_mono
                best_colors = list(colors)
    
    return best_colors, best_mono

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    aps = _build_aps()
    membership = _build_ap_membership(aps)
    
    best_colors = None
    best_mono = float('inf')
    
    rng = random.Random(42)
    
    # Seed pattern
    seed_colors = [(i // 2) % 2 for i in range(N)]
    c, m = _local_search(aps, membership, seed_colors)
    if m < best_mono:
        best_mono = m
        best_colors = c
    
    # Multi-start: random + local search
    while time.time() < deadline - 30:  # Leave 30s for SA
        colors = [rng.randint(0, 1) for _ in range(N)]
        c, m = _local_search(aps, membership, colors)
        if m < best_mono:
            best_mono = m
            best_colors = c
    
    # Simulated annealing from best
    while time.time() < deadline:
        c, m = _simulated_annealing(aps, membership, best_colors, 
                                     start_temp=5.0, end_temp=0.05, max_iter=1000)
        if m < best_mono:
            best_mono = m
            best_colors = c
        # Polish with local search
        c, m = _local_search(aps, membership, best_colors, max_passes=50)
        if m < best_mono:
            best_mono = m
            best_colors = c
    
    if best_colors is None:
        best_colors = seed_colors
    
    return best_colors
# EVOLVE-BLOCK-END
