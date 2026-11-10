import numpy as np
import time

def construct():
    p = 1009
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    # Precompute residues
    # r from 1 to (p-1)//2
    roots = np.arange(1, (p - 1) // 2 + 1)
    residues = (roots * roots) % p
    
    # Build boolean array
    is_residue = np.zeros(p, dtype=bool)
    is_residue[residues] = True
    
    # Find runs of 5 consecutive True values
    # We care about indices 1 to 1004 (inclusive) in the 0-indexed array
    # which correspond to values 1 to 1004.
    # Window size 5.
    # convolve(is_residue, ones(5), 'valid') gives sums for windows starting at 0..1004
    
    kernel = np.ones(5, dtype=int)
    window_sums = np.convolve(is_residue.astype(int), kernel, mode='valid')
    
    # window_sums[i] is sum of is_residue[i:i+5]
    # We need is_residue[s..s+4] to be all True, so window_sums[s] == 5
    # s ranges from 1 to 1004.
    
    valid_starts = np.where(window_sums == 5)[0]
    
    # Filter valid starts to be in [1, 1004]
    # valid_starts are indices in the array, which correspond to the value s directly.
    valid_starts = valid_starts[(valid_starts >= 1) & (valid_starts <= 1004)]
    
    if len(valid_starts) == 0:
        # Fallback to manual search if something went wrong or no run found
        # (Should not happen based on probability, but good for safety)
        for s in range(1, 1005):
            if time.time() > deadline:
                break
            if all(is_residue[s+i] for i in range(5)):
                # Find roots
                # Need to map residue to root
                # Create a dict or array for roots
                root_map = {}
                for r, res in zip(roots, residues):
                    if res not in root_map:
                        root_map[res] = r
                return {
                    'start': s,
                    'roots': [root_map[s+i] for i in range(5)]
                }
        raise RuntimeError('No run of 5 consecutive quadratic residues found')
    
    s = int(valid_starts[0])
    
    # Get roots
    # We need the specific roots for s, s+1, ..., s+4
    # We have `roots` and `residues` arrays.
    # Create a mapping from residue to root.
    # Since residues are unique for r in 1..504, we can use a dictionary or search.
    # For speed, dictionary is fine.
    root_map = {}
    for r, res in zip(roots, residues):
        if res not in root_map:
            root_map[res] = r
            
    r_list = []
    for i in range(5):
        val = s + i
        r_list.append(root_map[val])
        
    return {
        'start': s,
        'roots': r_list
    }
