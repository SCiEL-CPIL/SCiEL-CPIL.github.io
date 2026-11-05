import time
import numpy as np

P = 1009
M = 5
DEADLINE = 85.0  # seconds; keep a 5 s safety margin under the 90 s wall-clock limit


def _find_runs(npz):
    """Return the smallest s in [1, P-M] such that is_qr[s..s+M-1] are all True.
    Use NumPy convolution for a vectorised 'all True in a window of length M' check.
    """
    is_qr = npz['is_qr']
    kernel = np.ones(M, dtype=np.int32)
    conv = np.convolve(is_qr.astype(np.int32), kernel, mode='valid')
    # conv[i] == M iff is_qr[i..i+M-1] are all 1
    hits = np.where(conv == M)[0]
    if len(hits) == 0:
        return -1
    s0 = int(hits[0]) + 1  # 1-indexed
    # sanity: s0 must be in [1, P-M]
    if s0 < 1 or s0 > P - M:
        return -1
    return s0


def construct():
    t0 = time.time()

    # Precompute QR table and one root per residue using vectorised NumPy ops.
    x = np.arange(1, P, dtype=np.int64)
    r = (x * x) % P

    is_qr = np.zeros(P, dtype=np.bool_)
    roots = np.zeros(P, dtype=np.int64)

    for xv, rv in zip(x, r):
        if rv != 0:
            is_qr[rv] = True
            if roots[rv] == 0:
                roots[rv] = xv

    # Pack into a single numpy array to mimic a 'npz' input (per _find_runs signature).
    # _find_runs expects a dict-like with key 'is_qr'.
    npz = {'is_qr': is_qr}

    s = _find_runs(npz)
    if s == -1:
        raise RuntimeError("No run of 5 consecutive non-zero QRs found mod 1009")

    result_roots = [int(roots[s + i]) for i in range(M)]

    # Validate; if any root is invalid (should not happen), search exhaustively.
    for i in range(M):
        v = s + i
        rt = result_roots[i]
        if (rt * rt) % P != v:
            for xv in range(1, P):
                if (xv * xv) % P == v:
                    result_roots[i] = xv
                    break
            else:
                raise RuntimeError(f"No root found for residue {v}")

    return {'start': int(s), 'roots': result_roots}
