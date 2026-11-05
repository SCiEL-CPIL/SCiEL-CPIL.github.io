import time, random
from itertools import combinations

def _diffs(row):
    s=set()
    for i in range(len(row)):
        for j in range(i+1,len(row)):
            s.add(row[j]-row[i])
    return s

def _valid(rows):
    seen=set(); cnt=0
    for r in rows:
        for d in _diffs(r):
            if d in seen: return False
            seen.add(d); cnt+=1
    return True

def _scope(rows): return max(max(r) for r in rows)

def _greedy(max_scope, used):
    row=[0]; cm=0
    while len(row)<7:
        found=False
        for v in range(cm+1, max_scope+1):
            ok=True
            for x in row:
                if (v-x) in used: ok=False; break
            if ok:
                row.append(v)
                for x in row[:-1]: used.add(v-x)
                cm=v; found=True; break
        if not found: return None
    return row

def _run():
    for ms in [105,108,112,120]:
        used=set(); rows=[]
        for _ in range(7):
            r=_greedy(ms, used)
            if r is None: rows=None; break
            rows.append(r); used|= _diffs(r)
        if rows and _valid(rows):
            print(ms, "VALID scope", _scope(rows))
            for r in rows: print(r)
            return
    print("no greedy valid")

_run()
