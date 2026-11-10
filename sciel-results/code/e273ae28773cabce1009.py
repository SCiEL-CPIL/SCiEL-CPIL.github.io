def construct():
    """
    Maximizes the number of disjoint residue classes with distinct moduli in [2, 128].
    Uses the strategy of selecting moduli that are all multiples of a common divisor D,
    and assigning distinct residues modulo D.
    The number of classes k is bounded by k <= D and k <= floor(128/D).
    We iterate through possible D to find the maximum k.
    """
    best_k = 0
    best_D = 1
    for D in range(2, 129):
        max_moduli_count = 128 // D
        # We need k distinct residues mod D, so k <= D.
        # We also need k moduli, so k <= max_moduli_count.
        k = min(D, max_moduli_count)
        if k > best_k:
            best_k = k
            best_D = D
            
    # Construct the solution for the best D found
    # Moduli are D, 2D, ..., kD
    # Residues are 0, 1, ..., k-1
    moduli = [best_D * i for i in range(1, best_k + 1)]
    residues = list(range(best_k))
    return [[r, m] for r, m in zip(residues, moduli)]
