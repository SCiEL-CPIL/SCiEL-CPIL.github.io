def validate(ops, n, parents, outputs, inputs, B):
    red=set(); slow=set(inputs)
    for op,v in ops:
        if op=='load':
            if v in red: return False
            if v not in slow and v not in inputs: return False
            if len(red)>=B: return False
            red.add(v); 
            if v not in inputs: slow.discard(v)
        elif op=='store':
            if v not in red: return False
            red.discard(v); slow.add(v)
        elif op=='compute':
            if v in red: return False
            if v in inputs: return False
            if any(p not in red for p in parents[v]): return False
            if len(red)>=B: return False
            red.add(v); slow.discard(v)
        elif op=='delete':
            if v not in red: return False
            red.discard(v)
            if v not in inputs: 
                # deleted, not in slow
                pass
        else:
            return False
    if any(o not in slow for o in outputs): return False
    return True
