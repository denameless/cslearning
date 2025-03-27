VARIABLES = {'x','y','z','u','v','w','xx','yy','zz','uu','vv','ww'}

def parse_arguments(s):
    s = s.replace(' ', '')
    args = []
    current = []
    depth = 0
    for c in s:
        if c == ',' and depth == 0:
            args.append(''.join(current))
            current = []
        else:
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            current.append(c)
    args.append(''.join(current))
    return args

def parse_formula(formula):
    parts = formula.split('(', 1)
    if len(parts) != 2 or not parts[1].endswith(')'):
        return None, []
    predicate = parts[0]
    args_str = parts[1][:-1]
    args = parse_arguments(args_str)
    return predicate, args

def decompose_function(term):
    if '(' in term and term.endswith(')'):
        func_part = term.split('(', 1)
        func = func_part[0]
        args_str = func_part[1][:-1]
        args = parse_arguments(args_str)
        return (func, args)
    return None

def is_variable(term):
    return term.lower() in VARIABLES and '(' not in term and ')' not in term

def apply_substitution(term, substitution):
    if not substitution:
        return term
    if is_variable(term):
        return substitution.get(term.lower(), term)
    decomposed = decompose_function(term)
    if decomposed:
        func, args = decomposed
        new_args = [apply_substitution(arg, substitution) for arg in args]
        return f"{func}({','.join(new_args)})"
    return term

def occurs_check(var, term):
    if var == term:
        return True
    decomposed = decompose_function(term)
    if decomposed:
        _, args = decomposed
        for arg in args:
            if occurs_check(var, arg):
                return True
    return False

def unify(t1, t2, substitution):
    s1 = apply_substitution(t1, substitution).lower()
    s2 = apply_substitution(t2, substitution).lower()
    
    if s1 == s2:
        return substitution.copy()
    
    if is_variable(s1) and not is_variable(s2):
        if occurs_check(s1, s2):
            return None
        new_sub = substitution.copy()
        new_sub[s1] = s2
        for key in new_sub:
            new_sub[key] = apply_substitution(new_sub[key], {s1: s2})
        return new_sub
    elif is_variable(s2) and not is_variable(s1):
        if occurs_check(s2, s1):
            return None
        new_sub = substitution.copy()
        new_sub[s2] = s1
        for key in new_sub:
            new_sub[key] = apply_substitution(new_sub[key], {s2: s1})
        return new_sub
    elif is_variable(s1) and is_variable(s2):
        if s1 < s2:
            new_sub = substitution.copy()
            new_sub[s1] = s2
            for key in new_sub:
                new_sub[key] = apply_substitution(new_sub[key], {s1: s2})
            return new_sub
        else:
            new_sub = substitution.copy()
            new_sub[s2] = s1
            for key in new_sub:
                new_sub[key] = apply_substitution(new_sub[key], {s2: s1})
            return new_sub
    else:
        dec1 = decompose_function(s1)
        dec2 = decompose_function(s2)
        if dec1 and dec2:
            func1, args1 = dec1
            func2, args2 = dec2
            if func1 != func2 or len(args1) != len(args2):
                return None
            current_sub = substitution.copy()
            for a, b in zip(args1, args2):
                result = unify(a, b, current_sub)
                if result is None:
                    return None
                current_sub = result
            return current_sub
        else:
            return None

class Literal:
    def __init__(self, formula):
        self.raw = formula
        self.is_neg = formula.startswith('~')
        self.body = formula[1:] if self.is_neg else formula
        self.pred, self.args = parse_formula(self.body)
        if self.pred is None: 
            self.pred = self.body
            self.args = []
    
    def apply_subst(self, subst):
        new_args = [apply_substitution(arg, subst) for arg in self.args]
        return Literal(f"{'~' if self.is_neg else ''}{self.pred}({','.join(new_args)})")
    
    def __repr__(self):
        return ('~' if self.is_neg else '') + self.pred + ('('+','.join(self.args)+')' if self.args else '')

def parse_clause(clause):
    return [Literal(s) for s in clause]

def resolve(c1, c2):
    used_pairs = set()
    for i, lit1 in enumerate(c1):
        for j, lit2 in enumerate(c2):
            if (lit1.is_neg == lit2.is_neg) or (lit1.pred != lit2.pred):
                continue
            
            term1 = f"{lit1.pred}({','.join(lit1.args)})"
            term2 = f"{lit2.pred}({','.join(lit2.args)})"
            mgu = unify(term1, term2, {})
            if mgu is None:
                continue

            if (tuple(sorted(mgu.items())), i, j) in used_pairs:
                continue
            used_pairs.add((tuple(sorted(mgu.items())), i, j))
            
            new_clause = []

            for idx, lit in enumerate(c1):
                if idx != i:
                    new_lit = lit.apply_subst(mgu)
                    new_clause.append(new_lit)
            for idx, lit in enumerate(c2):
                if idx != j:
                    new_lit = lit.apply_subst(mgu)
                    new_clause.append(new_lit)

            seen = set()
            unique = []
            for lit in new_clause:
                key = (lit.is_neg, lit.pred, tuple(lit.args))
                if key not in seen:
                    seen.add(key)
                    unique.append(lit)
            yield unique, mgu, (i+1, j+1)

KB = {
    ('A(tony)',)
    ,('A(mike)',)
    ,('A(john)',)
    ,('L(tony,rain)',)
    ,('L(tony,snow)',)
    ,('~A(x)','S(x)','C(x)'),
    ('~C(y)','~L(y,rain)'),
    ('L(z,snow)','~S(z)'),
    ('~L(tony,u)','~L(mike,u)'),
    ('L(tony,v)','L(mike,v)'),
    ('~A(w)','~C(w)'),
    ('s(w)',)}
clauses = [parse_clause(c) for c in KB]
clause_used = [False] * len(clauses)
derived = set(tuple(sorted(str(l) for l in c)) for c in clauses)

for i, clause in enumerate(clauses):
    print(f"{i+1} {tuple(l.raw for l in clause)}")

step = len(clauses)
while True:
    new_added = False
    for i in range(len(clauses)):
        if clause_used[i]: continue
        for j in range(i+1, len(clauses)):
            if clause_used[j]: continue
            
            c1, c2 = clauses[i], clauses[j]
            for resolvent, mgu, (pos1, pos2) in resolve(c1, c2):
                if not resolvent:
                    print(f"{step+1} R[{i+1}{chr(96+pos1)},{j+1}{chr(96+pos2)}] {mgu} = ()")
                    exit()
                key = tuple(sorted(str(lit) for lit in resolvent))
                if key not in derived:
                    derived.add(key)
                    clauses.append(resolvent)
                    clause_used.append(False)
                    step += 1
                    new_added = True
                    subst_str = str(mgu) if mgu else ""
                    print(f"{step} R[{i+1}{chr(96+pos1)},{j+1}{chr(96+pos2)}] {subst_str} = {tuple(l.raw for l in resolvent)}")
                    
    if not new_added:
        print("No new clauses can be generated.")
        break