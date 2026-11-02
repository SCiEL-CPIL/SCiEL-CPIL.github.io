"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_block(blocks):
    """Compute autoconvolution of a block-constant signal.
    blocks: list of (height, width) tuples.
    Returns the full autoconvolution array (length 2*N-1) as a numpy array.
    """
    n_blocks = len(blocks)
    # Build the signal
    sig = np.zeros(N)
    pos = 0
    for h, w in blocks:
        sig[pos:pos+w] = h
        pos += w
    
    # Use FFT for autoconvolution
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    f = np.fft.rfft(sig, n_fft)
    ac = np.fft.irfft(f * np.conj(f), n_fft)[:2*N-1]
    return ac


def _score_blocks(blocks):
    """Compute C3' = max|autoconv| / (integral)^2 for a block-constant signal."""
    sig = np.zeros(N)
    pos = 0
    for h, w in blocks:
        sig[pos:pos+w] = h
        pos += w
    integral = np.sum(sig)
    if abs(integral) < 1e-12:
        return float('inf')
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    f = np.fft.rfft(sig, n_fft)
    ac = np.fft.irfft(f * np.conj(f), n_fft)[:2*N-1]
    max_abs = np.max(np.abs(ac))
    return max_abs / (integral * integral)


def _try_flip_block(blocks, block_idx):
    """Try flipping the sign of a block and check if score improves."""
    new_blocks = list(blocks)
    h, w = new_blocks[block_idx]
    new_blocks[block_idx] = (-h, w)
    return _score_blocks(new_blocks), new_blocks


def _try_merge_blocks(blocks):
    """Try merging adjacent blocks with the same sign."""
    i = 0
    new_blocks = []
    while i < len(blocks):
        if i + 1 < len(blocks) and blocks[i][0] == blocks[i+1][0]:
            new_blocks.append((blocks[i][0], blocks[i][1] + blocks[i+1][1]))
            i += 2
        else:
            new_blocks.append(blocks[i])
            i += 1
    return new_blocks


def _split_block(blocks, block_idx, split_frac):
    """Split a block into two at the given fraction."""
    h, w = blocks[block_idx]
    w1 = max(1, int(w * split_frac))
    w2 = w - w1
    if w2 < 1:
        return None
    new_blocks = list(blocks)
    new_blocks[block_idx] = (h, w1)
    new_blocks.insert(block_idx + 1, (h, w2))
    return new_blocks


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Start with two-block structure: +1 for first half, -1 for second half
    half = N // 2
    best_blocks = [(1.0, half), (-1.0, N - half)]
    best_score = _score_blocks(best_blocks)
    
    # Try different split points for the two-block structure
    for split in range(1, N - 1):
        if time.time() > deadline:
            break
        blocks = [(1.0, split), (-1.0, N - split)]
        s = _score_blocks(blocks)
        if s < best_score:
            best_score = s
            best_blocks = blocks
    
    # Local search: try flipping individual blocks
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_blocks)):
            if time.time() > deadline:
                break
            new_score, new_blocks = _try_flip_block(best_blocks, i)
            if new_score < best_score - 1e-12:
                best_score = new_score
                best_blocks = new_blocks
                improved = True
    
    # Try splitting blocks to create more structure
    for i in range(len(best_blocks)):
        if time.time() > deadline:
            break
        h, w = best_blocks[i]
        if w < 4:
            continue
        # Try splitting at various fractions
        for frac in [0.25, 0.33, 0.5, 0.67, 0.75]:
            if time.time() > deadline:
                break
            split_blocks = _split_block(best_blocks, i, frac)
            if split_blocks is None:
                continue
            # Flip the second half of the split to create a new sign change
            split_blocks[i+1] = (-h, split_blocks[i+1][1])
            s = _score_blocks(split_blocks)
            if s < best_score - 1e-12:
                best_score = s
                best_blocks = split_blocks
                # Merge any adjacent same-sign blocks
                best_blocks = _try_merge_blocks(best_blocks)
    
    # Final merge of adjacent same-sign blocks
    best_blocks = _try_merge_blocks(best_blocks)
    
    # Build the final signal
    sig = np.zeros(N)
    pos = 0
    for h, w in best_blocks:
        sig[pos:pos+w] = h
        pos += w
    
    # Verify score
    final_score = _score_blocks(best_blocks)
    if final_score < best_score:
        best_score = final_score
    
    return sig
# EVOLVE-BLOCK-END
