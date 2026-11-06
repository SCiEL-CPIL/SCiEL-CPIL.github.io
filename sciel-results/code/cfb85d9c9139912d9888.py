<<<<<<< SEARCH
def _synthesize_cnot():
    """CNOT gate: just one CNOT."""
    return [["cx", 0, 1]]


def _synthesize_iswap():
    """iSWAP gate: 2 CNOTs + single qubit rotations.
    iSWAP = [[0,0,0,1],[0,0,i,0],[0,i,0,0],[1,0,0,0]]
    Decomposition: A1 ⊗ A2 · CNOT(0→1) · B1 ⊗ B2 · CNOT(1→0) · C1 ⊗ C2
    """
    return [
        # C layer: C1 = [[1, -i], [0, 0]], C2 = [[1, 0], [-i, 0]]
        ["u3", 0, math.pi/2, 0, -math.pi/2],  # C1
        ["u3", 1, math.pi/2, -math.pi/2, 0],   # C2
        ["cx", 1, 0],
        # B layer: B1 = [[1,0],[0,i]], B2 = [[1,0],[0,-i]]
        ["u3", 0, 0, 0, math.pi/2],            # B1 = diag(1, i)
        ["u3", 1, 0, 0, -math.pi/2],           # B2 = diag(1, -i)
        ["cx", 0, 1],
        # A layer: A1 = [[1,0],[0,-i]], A2 = [[1,0],[0,i]]
        ["u3", 0, 0, 0, -math.pi/2],           # A1 = diag(1, -i)
        ["u3", 1, 0, 0, math.pi/2],            # A2 = diag(1, i)
    ]


def _synthesize_swap():
    """SWAP gate: 3 CNOTs."""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _synthesize_toffoli():
    """Toffoli gate: 6 CNOTs (standard decomposition)."""
    # Toffoli |cc> -> |t>
    # Standard decomposition uses 6 CNOTs
    return [
        ["u3", 2, math.pi/2, 0, math.pi/2],
        ["cx", 1, 2],
        ["u3", 2, math.pi/4, 0, -math.pi/2],
        ["cx", 0, 2],
        ["u3", 2, -math.pi/4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, math.pi/4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, -math.pi/4, 0, -math.pi/2],
        ["u3", 1, math.pi/4, 0, math.pi/4],
        ["u3", 2, math.pi/4, 0, math.pi/2],
        ["cx", 1, 2],
        ["u3", 1, -math.pi/4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, math.pi/2, 0, math.pi/2],
    ]
=======
def _synthesize_cnot():
    """CNOT gate: just one CNOT."""
    return [["cx", 0, 1]]


def _synthesize_iswap():
    """iSWAP gate using QSD: 3 CNOTs max, but iSWAP needs 3 CNOTs."""
    # iSWAP = [[0,0,0,1],[0,0,i,0],[0,i,0,0],[1,0,0,0]]
    # Use QSD to decompose
    target = np.array([[0,0,0,1],[0,0,1j,0],[0,1j,0,0],[1,0,0,0]], dtype=complex)
    gates = _qsd_2qubit(target)
    return gates


def _synthesize_swap():
    """SWAP gate: 3 CNOTs."""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _synthesize_toffoli():
    """Toffoli gate using QSD: 6 CNOTs."""
    # Toffoli: |11x> -> |11(1⊕x)>
    target = np.eye(8, dtype=complex)
    target[6, 6] = 0.0
    target[6, 7] = 1.0
    target[7, 6] = 1.0
    target[7, 7] = 0.0
    gates = _qsd_3qubit(target)
    return gates
>>>>>>> REPLACE

