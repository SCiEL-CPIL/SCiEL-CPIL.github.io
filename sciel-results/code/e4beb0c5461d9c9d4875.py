"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_fft(sig):
    """Compute autoconvolution of a signal using FFT."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    f = np.fft.rfft(sig, n_fft)
    ac = np.fft.irfft(f * np.conj(f), n_fft)[:2*N-1]
    return ac


def _exact_score(sig):
    """Compute C3' = max|autoconv| / (integral)^2."""
    integral = np.sum(sig)
    if abs(integral) < 1e-12:
        return float('inf')
    ac = _autoconv_fft(sig)
    max_abs = np.max(np.abs(ac))
    return max_abs / (integral * integral)


def _sig_to_blocks(sig):
    blocks = []
    i = 0
    while i < N:
        h = sig[i]
        j = i
        while j < N and sig[j] == h:
            j += 1
        blocks.append((h, j - i))
        i = j
    return blocks


def _blocks_to_sig(blocks):
    sig = np.zeros(N)
    pos = 0
    for h, w in blocks:
        sig[pos:pos+w] = h
        pos += w
    return sig


def _merge_adjacent(blocks):
    if not blocks:
        return blocks
    merged = [list(blocks[0])]
    for h, w in blocks[1:]:
        if merged[-1][0] == h:
            merged[-1][1] += w
        else:
            merged.append([h, w])
    return [(h, w) for h, w in merged]


def _score_sig(sig):
    return _exact_score(sig)


def _local_search_blocks(blocks, deadline):
    """Local search on block representation: flip blocks, split blocks."""
    best_blocks = list(blocks)
    best_score = _score_sig(_blocks_to_sig(best_blocks))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try flipping each block
        for i in range(len(best_blocks)):
            if time.time() > deadline:
                return best_blocks, best_score
            new_blocks = list(best_blocks)
            h, w = new_blocks[i]
            new_blocks[i] = (-h, w)
            new_blocks = _merge_adjacent(new_blocks)
            s = _score_sig(_blocks_to_sig(new_blocks))
            if s < best_score - 1e-12:
                best_score = s
                best_blocks = new_blocks
                improved = True
        
        # Try splitting a block and flipping the second part
        for i in range(len(best_blocks)):
            if time.time() > deadline:
                return best_blocks, best_score
            h, w = best_blocks[i]
            if w < 4:
                continue
            for frac in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                if time.time() > deadline:
                    return best_blocks, best_score
                w1 = max(1, int(w * frac))
                w2 = w - w1
                if w2 < 1:
                    continue
                new_blocks = list(best_blocks)
                new_blocks[i] = (h, w1)
                new_blocks.insert(i + 1, (-h, w2))
                new_blocks = _merge_adjacent(new_blocks)
                s = _score_sig(_blocks_to_sig(new_blocks))
                if s < best_score - 1e-12:
                    best_score = s
                    best_blocks = new_blocks
                    improved = True
    
    return best_blocks, best_score


def _elementwise_local_search(sig, deadline):
    """Local search: flip individual elements and small neighborhoods."""
    best_score = _exact_score(sig)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            # Try flipping single element
            new_sig = sig.copy()
            new_sig[i] = -new_sig[i]
            s = _exact_score(new_sig)
            if s < best_score - 1e-12:
                best_score = s
                sig = new_sig
                improved = True
                continue
            
            # Try flipping small neighborhoods (size 2-5)
            for size in [2, 3, 4, 5]:
                if i + size > N:
                    continue
                new_sig = sig.copy()
                new_sig[i:i+size] = -new_sig[i:i+size]
                s = _exact_score(new_sig)
                if s < best_score - 1e-12:
                    best_score = s
                    sig = new_sig
                    improved = True
                    break
    
    return sig, best_score


def _structured_local_search(sig, deadline):
    """Local search with structured perturbations: low-freq sinusoids, block flips."""
    best_score = _exact_score(sig)
    
    # Try various structured perturbations
    perturbation_types = []
    
    # Low-frequency sinusoidal perturbations
    for freq in [1, 2, 3, 4, 5, 8, 10]:
        for phase in [0, 0.5, 1.0]:
            perturbation_types.append(('sinusoid', freq, phase))
    
    # Block-based flips at various scales
    for block_size in [10, 20, 25, 40, 50]:
        for start in range(0, N - block_size + 1, block_size // 2):
            perturbation_types.append(('block_flip', start, block_size))
    
    # Try each perturbation
    for pert_type, param1, param2 in perturbation_types:
        if time.time() > deadline:
            break
        
        new_sig = sig.copy()
        
        if pert_type == 'sinusoid':
            t = np.arange(N)
            wave = np.sin(2 * np.pi * param1 * t / N + param2 * 2 * np.pi)
            # Flip where wave crosses zero (create sign changes)
            mask = wave > 0
            new_sig[mask] = 1.0
            new_sig[~mask] = -1.0
        elif pert_type == 'block_flip':
            start = param1
            size = param2
            if start + size <= N:
                new_sig[start:start+size] = -new_sig[start:start+size]
        
        s = _exact_score(new_sig)
        if s < best_score - 1e-12:
            best_score = s
            sig = new_sig
    
    return sig, best_score


def _generate_initializations():
    """Generate diverse initial signals for local search."""
    inits = []
    
    # 1. Two-block: +1 first half, -1 second half
    sig1 = np.ones(N)
    sig1[N//2:] = -1.0
    inits.append(sig1)
    
    # 2. Three-block: +1, -1, +1
    sig2 = np.ones(N)
    sig2[N//3:2*N//3] = -1.0
    inits.append(sig2)
    
    # 3. Four-block: +1, -1, +1, -1
    sig3 = np.ones(N)
    sig3[N//4:N//2] = -1.0
    sig3[3*N//4:] = -1.0
    inits.append(sig3)
    
    # 4. Alternating blocks of size 25
    sig4 = np.ones(N)
    for i in range(0, N, 50):
        sig4[i:i+25] = 1.0
        sig4[i+25:i+50] = -1.0
    inits.append(sig4)
    
    # 5. Alternating blocks of size 20
    sig5 = np.ones(N)
    for i in range(0, N, 40):
        sig5[i:i+20] = 1.0
        sig5[i+20:i+40] = -1.0
    inits.append(sig5)
    
    # 6. Random with fixed seed
    np.random.seed(42)
    sig6 = np.random.choice([-1.0, 1.0], size=N)
    inits.append(sig6)
    
    # 7. Random with different seed
    np.random.seed(123)
    sig7 = np.random.choice([-1.0, 1.0], size=N)
    inits.append(sig7)
    
    # 8. Sine wave quantized
    t = np.arange(N)
    sig8 = np.sign(np.sin(2 * np.pi * 3 * t / N))
    sig8[sig8 == 0] = 1.0
    inits.append(sig8)
    
    # 9. Square wave with duty cycle 0.4
    sig9 = np.ones(N)
    for i in range(N):
        if (i % 40) < 16:
            sig9[i] = 1.0
        else:
            sig9[i] = -1.0
    inits.append(sig9)
    
    # 10. Triangular wave quantized
    sig10 = np.ones(N)
    for i in range(N):
        phase = (i % 80) / 80.0
        if phase < 0.5:
            sig10[i] = 1.0 if phase < 0.25 else -1.0
        else:
            sig10[i] = -1.0 if phase < 0.75 else 1.0
    inits.append(sig10)
    
    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    best_sig = None
    best_score = float('inf')
    
    # Phase 1: Try multiple initializations with block-based local search
    inits = _generate_initializations()
    
    for init_sig in inits:
        if time.time() > deadline - 30:
            break
        
        # Ensure binary ±1
        init_sig = np.sign(init_sig)
        init_sig[init_sig == 0] = 1.0
        
        # Block-based local search
        blocks = _sig_to_blocks(init_sig)
        blocks, block_score = _local_search_blocks(blocks, deadline - 10)
        
        if block_score < best_score:
            best_score = block_score
            best_sig = _blocks_to_sig(blocks)
    
    if best_sig is None:
        best_sig = np.ones(N)
        best_sig[N//2:] = -1.0
        best_score = _exact_score(best_sig)
    
    # Phase 2: Element-wise local search for fine-grained refinement
    if time.time() < deadline - 15:
        refined_sig, refined_score = _elementwise_local_search(best_sig, deadline - 5)
        if refined_score < best_score:
            best_sig = refined_sig
            best_score = refined_score
    
    # Phase 3: Structured local search with sinusoidal and block perturbations
    if time.time() < deadline - 10:
        struct_sig, struct_score = _structured_local_search(best_sig, deadline - 5)
        if struct_score < best_score:
            best_sig = struct_sig
            best_score = struct_score
    
    # Phase 4: Final block-based pass
    if time.time() < deadline - 5:
        blocks = _sig_to_blocks(best_sig)
        blocks, block_score = _local_search_blocks(blocks, deadline)
        if block_score < best_score:
            best_score = block_score
            best_sig = _blocks_to_sig(blocks)
    
    # Phase 5: Final element-wise pass
    if time.time() < deadline - 2:
        refined_sig, refined_score = _elementwise_local_search(best_sig, deadline)
        if refined_score < best_score:
            best_sig = refined_sig
    
    return best_sig
# EVOLVE-BLOCK-END
