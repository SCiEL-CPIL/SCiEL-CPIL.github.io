import numpy as np
import time


def partition(adjacency):
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    
    deadline = time.time() + 110  # 10s safety margin before 120s limit
    
    def _cut_size(labels):
        l = labels.astype(np.int8)
        cross_mask = (l[:, None] != l[None, :])
        return int((adj * cross_mask).sum() // 2)
    
    def local_search(labels, max_sweeps=200):
        labels = labels.copy()
        total_deg = adj.sum(axis=1)
        for sweep in range(max_sweeps):
            if time.time() > deadline:
                break
            same_mask = (labels[None, :] == labels[:, None]).astype(np.int8)
            same_counts = (adj * same_mask).sum(axis=1)
            diff_counts = total_deg - same_counts
            
            flip_mask = diff_counts > same_counts
            if not flip_mask.any():
                break
            labels[flip_mask] = 1 - labels[flip_mask]
        return labels
    
    best_labels = None
    best_cut = -1
    rng = np.random.RandomState(42)
    
    # Alternating start
    labels = np.arange(n) % 2
    labels = local_search(labels)
    c = _cut_size(labels)
    if c > best_cut:
        best_cut = c
        best_labels = labels.copy()
    
    # Random starts with time budget
    while time.time() < deadline - 2:
        labels = rng.randint(0, 2, size=n)
        labels = local_search(labels, max_sweeps=100)
        c = _cut_size(labels)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()
    
    return best_labels
