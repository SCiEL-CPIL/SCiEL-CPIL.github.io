import time

# EVOLVE-BLOCK-START
def construct():
    """
    Find the smallest s >= 1 such that s, s+1, s+2, s+3, s+4 are all
    non-zero quadratic residues mod 1009, with roots for each.
    """
    p = 1009
    deadline = time.time() + 80.0  # safety margin before 90s kill

    # Build QR lookup table and root map in one pass
    is_qr = [False] * p
    root_map = {}
    for x in range(1, p):
        sq = (x * x) % p
        is_qr[sq] = True
        if sq not in root_map:
            root_map[sq] = x

    # Find smallest s in [1, 1004] where s..s+4 are all QR
    best_s = None
    for s in range(1, p - 3):  # 1 to 1004
        if is_qr[s] and is_qr[s + 1] and is_qr[s + 2] and is_qr[s + 3] and is_qr[s + 4]:
            best_s = s
            break

    if best_s is None:
        best_s = p - 5  # fallback, shouldn't happen

    # Extract roots
    roots = [root_map[best_s + i] for i in range(5)]

    return {'start': best_s, 'roots': roots}
# EVOLVE-BLOCK-END
