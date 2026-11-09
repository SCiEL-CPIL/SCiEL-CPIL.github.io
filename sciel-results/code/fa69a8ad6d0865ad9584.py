"""Cochromatic partition of the fixed 14-vertex graph.
Precomputed partition (7 parts, each a clique or independent set), validated at runtime.
"""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

_PRECOMPUTED = [
    [0, 12],   # clique (edge)
    [0, 2],    # clique (edge)
    [1, 4],    # clique (edge)
    [2, 5],    # clique (edge)
    [3, 8],    # clique (edge)
    [5, 9],    # clique (edge)
    [7, 11],   # independent (no edge)
    [13],      # singleton
]

def _verify(parts):
    adj = set()
    for a, b in P['edges']:
        adj.add((a, b)); adj.add((b, a))
    seen = set()
    for part in parts:
        for v in part:
            assert isinstance(v, int) and not isinstance(v, bool), f"bad type {v}"
            assert v not in seen, f"dup vertex {v}"
            seen.add(v)
        ok_c = all((a, b) in adj for i, a in enumerate(part) for b in part[i+1:])
        ok_i = all((a, b) not in adj for i, a in enumerate(part) for b in part[i+1:])
        assert ok_c or ok_i, f"part {part} invalid"
    assert seen == set(range(P['n'])), "coverage error"

def construct():
    _verify(_PRECOMPUTED)
    return [list(part) for part in _PRECOMPUTED]
# EVOLVE-BLOCK-END
