import sys
from collections import defaultdict, deque

# =========================================================
# REGEX -> Postfija (tokenización PRO + operador de concat interno)
# =========================================================

# Operadores visibles para el usuario (¡ya no está el '.'):
PREC = {
    '*': 3, '+': 3, '?': 3,
    '·': 2,            # operador interno de concatenación (no tecleable)
    '|': 1
}
RIGHT_ASSOC = {'*', '+', '?'}
OPS = set(['(', ')', '*', '+', '?', '|'])  # sin '.'

# ---------- Tokens ----------
# Cada token es un dict:
#   - type: 'LIT' | 'OP'
#   - val : str (carácter)
# Nota: 'ε' es LIT('ε'). El operador de concatenación es OP('·').

def tokenizar(regex):
    """
    Convierte la cadena regex en una lista de tokens, manejando escapes con '\\'.
    Reglas:
      - '\\x' => LIT('x') para cualquier 'x' (incluye paréntesis y operadores).
      - Espacios no escapados se ignoran.
      - Caracteres en OPS => OP(c) (si NO están escapados).
      - El resto => LIT(c).
    """
    tokens = []
    i = 0
    while i < len(regex):
        c = regex[i]
        if c == '\\':
            if i + 1 >= len(regex):
                raise ValueError("Barra invertida '\\' al final sin carácter a escapar")
            tokens.append({'type': 'LIT', 'val': regex[i+1]})
            i += 2
        else:
            if c.isspace():
                i += 1
                continue
            if c in OPS:
                tokens.append({'type': 'OP', 'val': c})
            else:
                tokens.append({'type': 'LIT', 'val': c})
            i += 1
    return tokens

def es_simbolo_tok(tok):
    return tok['type'] == 'LIT'  # cualquier literal, incluida 'ε'

def tok_is(tok, t, v=None):
    return tok['type'] == t and (v is None or tok['val'] == v)

def insertar_concatenacion(regex):
    """
    Inserta el operador interno '·' entre tokens donde la concatenación es implícita.
    """
    toks = tokenizar(regex)
    if not toks:
        return []

    res = []
    for i, c1 in enumerate(toks):
        res.append(c1)
        if i == len(toks) - 1:
            break
        c2 = toks[i+1]
        # Insertamos concatenación entre:
        #   X ∈ {símbolo, ')', '*', '+', '?'} y
        #   Y ∈ {símbolo, '(', LIT('ε')}
        cond1 = es_simbolo_tok(c1) or (tok_is(c1, 'OP') and c1['val'] in [')', '*', '+', '?'])
        cond2 = es_simbolo_tok(c2) or tok_is(c2, 'OP', '(') or (c2['type'] == 'LIT' and c2['val'] == 'ε')
        if cond1 and cond2:
            res.append({'type': 'OP', 'val': '·'})
    return res

def a_postfija(tokens_conc):
    """
    Shunting-yard sobre tokens (con '·' como concatenación interna).
    Devuelve lista de tokens en postfijo.
    """
    output = []
    opstack = []

    def prec(tok):
        return PREC.get(tok['val'], -1)

    for tok in tokens_conc:
        if es_simbolo_tok(tok):
            output.append(tok)
        elif tok_is(tok, 'OP', '('):
            opstack.append(tok)
        elif tok_is(tok, 'OP', ')'):
            while opstack and not tok_is(opstack[-1], 'OP', '('):
                output.append(opstack.pop())
            if not opstack:
                raise ValueError("Paréntesis desbalanceados")
            opstack.pop()  # '('
        elif tok['type'] == 'OP' and tok['val'] in PREC:
            c = tok['val']
            while (opstack and not tok_is(opstack[-1], 'OP', '(') and
                   (prec(opstack[-1]) > PREC[c] or
                    (prec(opstack[-1]) == PREC[c] and c not in RIGHT_ASSOC))):
                output.append(opstack.pop())
            opstack.append(tok)
        else:
            raise ValueError(f"Símbolo no soportado: {tok}")

    while opstack:
        top = opstack.pop()
        if tok_is(top, 'OP') and top['val'] in ['(', ')']:
            raise ValueError("Paréntesis desbalanceados al final")
        output.append(top)

    return output

def stringify_postfix(post):
    """
    Para mostrar el postfijo a usuario:
    - convertimos '·' (concat interna) en '.' para que sea legible.
    """
    s = []
    for tok in post:
        v = tok['val']
        if tok['type'] == 'OP' and v == '·':
            v = '.'
        s.append(v)
    return ''.join(s)

# =========================================================
# Thompson: Postfija (tokens) -> AFN
# =========================================================

class Estado:
    __slots__ = ("id", "trans", "eps")
    def __init__(self, id_):
        self.id = id_
        self.trans = defaultdict(set)   # Dict[str, Set[Estado]]
        self.eps = set()                # Set[Estado]

class AFN:
    def __init__(self, inicio, aceptacion, estados):
        self.inicio = inicio
        self.aceptacion = aceptacion
        self.estados = estados

def thompson_desde_postfija(post):
    next_id = [0]
    def nuevo_estado():
        e = Estado(next_id[0])
        next_id[0] += 1
        return e

    pila = []
    estados = []

    def push_frag(s, f):
        estados.extend([s, f])
        pila.append((s, f))

    def pop1(op):
        if not pila:
            raise ValueError(f"Falta operando para '{op}'")
        return pila.pop()

    def pop2(op):
        if len(pila) < 2:
            raise ValueError(f"Faltan operandos para '{op}'")
        b1, b2 = pila.pop()
        a1, a2 = pila.pop()
        return a1, a2, b1, b2

    def frag_simbolo(a):
        s = nuevo_estado()
        f = nuevo_estado()
        s.trans[a].add(f)
        push_frag(s, f)

    for tok in post:
        if es_simbolo_tok(tok):
            c = tok['val']
            if c == 'ε':
                s = nuevo_estado()
                f = nuevo_estado()
                s.eps.add(f)
                push_frag(s, f)
            else:
                frag_simbolo(c)
        elif tok['type'] == 'OP':
            op = tok['val']
            if op == '·':  # concatenación
                a1, a2, b1, b2 = pop2('·')
                a2.eps.add(b1)
                push_frag(a1, b2)
            elif op == '|':
                a1, a2, b1, b2 = pop2('|')
                s = nuevo_estado()
                f = nuevo_estado()
                s.eps.update([a1, b1])
                a2.eps.add(f)
                b2.eps.add(f)
                push_frag(s, f)
            elif op == '*':
                a1, a2 = pop1('*')
                s = nuevo_estado()
                f = nuevo_estado()
                s.eps.update([a1, f])
                a2.eps.update([a1, f])
                push_frag(s, f)
            elif op == '+':
                a1, a2 = pop1('+')
                s = nuevo_estado()
                f = nuevo_estado()
                s.eps.add(a1)
                a2.eps.update([a1, f])
                push_frag(s, f)
            elif op == '?':
                a1, a2 = pop1('?')
                s = nuevo_estado()
                f = nuevo_estado()
                s.eps.update([a1, f])
                a2.eps.add(f)
                push_frag(s, f)
            else:
                raise ValueError(f"Operador inesperado {op}")
        else:
            raise ValueError(f"Token inesperado {tok}")

    if len(pila) != 1:
        raise ValueError("Expresión inválida (sobran o faltan fragmentos)")

    inicio, acept = pila.pop()

    # recolectar alcanzables
    vistos = set()
    q = deque([inicio])
    alcance = []
    while q:
        u = q.popleft()
        if u in vistos:
            continue
        vistos.add(u)
        alcance.append(u)
        for v in u.eps:
            if v not in vistos: q.append(v)
        for _, dests in u.trans.items():
            for v in dests:
                if v not in vistos: q.append(v)

    return AFN(inicio, acept, alcance)

# =========================================================
# Simulación de AFN
# =========================================================

def epsilon_cierre(estados):
    pila = list(estados)
    cierre = set(estados)
    while pila:
        u = pila.pop()
        for v in u.eps:
            if v not in cierre:
                cierre.add(v)
                pila.append(v)
    return cierre

def mover(estados, simbolo):
    res = set()
    for u in estados:
        for v in u.trans.get(simbolo, ()):
            res.add(v)
    return res

def acepta(afn, cadena):
    actual = epsilon_cierre({afn.inicio})
    for c in cadena:
        actual = epsilon_cierre(mover(actual, c))
        if not actual:
            break
    return afn.aceptacion in actual

# =========================================================
# AFN -> AFD, dibujo, minimización (igual que antes)
# =========================================================

class EstadoDFA:
    __slots__ = ("id", "trans", "aceptacion", "nfa_set")
    def __init__(self, id_, nfa_set, aceptacion=False):
        self.id = id_
        self.nfa_set = nfa_set
        self.trans = {}
        self.aceptacion = aceptacion

class AFD:
    def __init__(self, inicio, estados):
        self.inicio = inicio
        self.estados = estados

def alfabeto_de_afn(afn):
    A = set()
    for s in afn.estados:
        for sym in s.trans.keys():
            if sym != 'ε':
                A.add(sym)
    return A

def mover_con_epsilon(estados_nfa, simbolo):
    return epsilon_cierre(mover(estados_nfa, simbolo))

def afn_a_afd(afn):
    A = alfabeto_de_afn(afn)
    start_set = frozenset(epsilon_cierre({afn.inicio}))
    start_is_accept = afn.aceptacion in start_set

    dfa_states = []
    mapping = {}
    q0 = EstadoDFA(0, start_set, start_is_accept)
    dfa_states.append(q0)
    mapping[start_set] = q0
    q = deque([q0])
    next_id = 1

    while q:
        u = q.popleft()
        for a in A:
            destino = frozenset(mover_con_epsilon(u.nfa_set, a))
            if not destino:
                continue
            if destino not in mapping:
                nuevo = EstadoDFA(next_id, destino, aceptacion=(afn.aceptacion in destino))
                mapping[destino] = nuevo
                dfa_states.append(nuevo)
                q.append(nuevo)
                next_id += 1
            u.trans[a] = mapping[destino]

    return AFD(q0, dfa_states)

def acepta_afd(afd, cadena):
    actual = afd.inicio
    for c in cadena:
        if c not in actual.trans:
            return False
        actual = actual.trans[c]
    return actual.aceptacion

# --------- Dibujo (Graphviz/NetworkX) ---------
def _render_with_graphviz(edges, start_id, accept_ids, png_path, dot_path):
    try:
        from graphviz import Digraph
        dot = Digraph(format='png')
        dot.attr(rankdir='LR', dpi='180', concentrate='false')
        dot.attr('node', shape='circle')

        nodes = set([start_id])
        for (u, v) in edges.keys():
            nodes.add(u); nodes.add(v)

        for u in nodes:
            if u in accept_ids:
                dot.node(str(u), shape='doublecircle', style='filled', fillcolor='lightgrey')
            else:
                dot.node(str(u), shape='circle')

        dot.node('start', shape='point')
        dot.edge('start', str(start_id), label='')

        for (u, v), lab in edges.items():
            dot.edge(str(u), str(v), label=lab)

        with open(dot_path, 'w', encoding='utf-8') as f:
            f.write(dot.source)

        dot.render(filename=png_path, cleanup=True)
        import os, shutil
        src = png_path + '.png'
        if os.path.exists(src):
            shutil.move(src, png_path)
        return f"Imagen generada con Graphviz: {png_path}"
    except Exception:
        return None

def _render_with_networkx(nodes, edges, start_id, accept_ids, png_path):
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for (u, v), lab in edges.items():
        G.add_edge(u, v, label=lab)

    pos = None
    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        pos = graphviz_layout(G, prog='dot')
    except Exception:
        try:
            from networkx.drawing.nx_agraph import graphviz_layout as gv2
            pos = gv2(G, prog='dot')
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=0.8, iterations=200)

    accept_nodes = list(accept_ids)
    others = [n for n in G.nodes if n not in accept_nodes and n != start_id]

    plt.figure(figsize=(9, 6))
    nx.draw_networkx_nodes(G, pos, nodelist=others, node_size=620, linewidths=1.2)
    nx.draw_networkx_nodes(G, pos, nodelist=[start_id], node_shape='s', node_size=760, linewidths=1.2)
    nx.draw_networkx_nodes(G, pos, nodelist=accept_nodes, node_size=760, linewidths=1.5, node_color='#e0e0e0')
    nx.draw_networkx_nodes(G, pos, nodelist=accept_nodes, node_size=880, linewidths=2.2, node_color='none')

    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes()}, font_size=10)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=24, width=1.6, connectionstyle='arc3,rad=0.16')

    e_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=e_labels, font_size=9,
                                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()
    return f"Imagen generada con NetworkX/Matplotlib: {png_path}"

def dibujar_afn(afn, nombre_png, nombre_dot):
    from collections import defaultdict as _dd
    edge_labels = _dd(list)
    nodes = set()
    for s in afn.estados:
        nodes.add(s.id)
        for v in s.eps:
            edge_labels[(s.id, v.id)].append('ε')
        for sym, dests in s.trans.items():
            for v in dests:
                edge_labels[(s.id, v.id)].append(sym)
    edges = {(u, v): '|'.join(sorted(set(labs))) for (u, v), labs in edge_labels.items()}

    ok = _render_with_graphviz(edges, afn.inicio.id, {afn.aceptacion.id}, nombre_png, nombre_dot)
    if ok:
        return ok
    return _render_with_networkx(nodes, edges, afn.inicio.id, {afn.aceptacion.id}, nombre_png)

def dibujar_afd(afd, nombre_png, nombre_dot):
    from collections import defaultdict as _dd
    edge_labels = _dd(list)
    nodes = set()
    accept_ids = set()
    for s in afd.estados:
        nodes.add(s.id)
        if s.aceptacion:
            accept_ids.add(s.id)
        for sym, v in s.trans.items():
            edge_labels[(s.id, v.id)].append(sym)
    edges = {(u, v): '|'.join(sorted(set(labs))) for (u, v), labs in edge_labels.items()}

    ok = _render_with_graphviz(edges, afd.inicio.id, accept_ids, nombre_png, nombre_dot)
    if ok:
        return ok
    return _render_with_networkx(nodes, edges, afd.inicio.id, accept_ids, nombre_png)

# =========================================================
# Minimizacion de AFD (Hopcroft)
# =========================================================

def dfa_alfabeto(afd):
    A = set()
    for s in afd.estados:
        for a in s.trans.keys():
            A.add(a)
    return A

def dfa_estados_reachables(afd):
    vistos = set()
    orden = []
    q = deque([afd.inicio])
    while q:
        u = q.popleft()
        if u.id in vistos:
            continue
        vistos.add(u.id)
        orden.append(u)
        for v in u.trans.values():
            if v.id not in vistos:
                q.append(v)
    old_to_new = {}
    nuevos = []
    for i, s in enumerate(orden):
        ns = EstadoDFA(i, s.nfa_set, s.aceptacion)
        old_to_new[s.id] = ns
        nuevos.append(ns)
    for s in orden:
        ns = old_to_new[s.id]
        for a, v in s.trans.items():
            if v.id in old_to_new:
                ns.trans[a] = old_to_new[v.id]
    inicio = old_to_new[afd.inicio.id]
    return AFD(inicio, nuevos)

def dfa_completar_con_sumidero(afd):
    A = dfa_alfabeto(afd)
    old_to_new = {}
    nuevos = []
    for i, s in enumerate(afd.estados):
        ns = EstadoDFA(i, s.nfa_set, s.aceptacion)
        old_to_new[s.id] = ns
        nuevos.append(ns)

    sink = EstadoDFA(len(nuevos), None, False)

    for s in afd.estados:
        ns = old_to_new[s.id]
        for a, v in s.trans.items():
            ns.trans[a] = old_to_new[v.id]

    need_sink = False
    for ns in nuevos:
        for a in A:
            if a not in ns.trans:
                ns.trans[a] = sink
                need_sink = True
    if need_sink:
        for a in A:
            sink.trans[a] = sink
        nuevos.append(sink)
    inicio = old_to_new[afd.inicio.id]
    return AFD(inicio, nuevos)

def minimizar_afd(afd):
    afd1 = dfa_estados_reachables(afd)
    afd2 = dfa_completar_con_sumidero(afd1)

    estados = afd2.estados
    A = sorted(dfa_alfabeto(afd2))

    idx_of = {s.id: i for i, s in enumerate(estados)}
    by_idx = estados

    sym_to_pos = {a: k for k, a in enumerate(A)}
    delta = [[None]*len(A) for _ in by_idx]
    for i, s in enumerate(by_idx):
        for a, v in s.trans.items():
            delta[i][sym_to_pos[a]] = idx_of[v.id]
    for i in range(len(by_idx)):
        for k in range(len(A)):
            if delta[i][k] is None:
                delta[i][k] = i

    F = set(i for i, s in enumerate(by_idx) if s.aceptacion)
    NF = set(range(len(by_idx))) - F
    P = [F, NF] if F and NF else [F] if F else [NF]
    from collections import deque as _dq
    W = _dq([set(g) for g in P])

    from collections import defaultdict as _dd
    inv = [_dd(set) for _ in A]
    for i in range(len(by_idx)):
        for k in range(len(A)):
            j = delta[i][k]
            inv[k][j].add(i)

    while W:
        Aset = W.popleft()
        for k in range(len(A)):
            X = set()
            for j in Aset:
                X |= inv[k][j]
            newP = []
            for B in P:
                inter = B & X
                diff = B - X
                if inter and diff:
                    newP.extend([inter, diff])
                    if B in W:
                        W.remove(B)
                        W.append(inter); W.append(diff)
                    else:
                        W.append(inter if len(inter) <= len(diff) else diff)
                else:
                    newP.append(B)
            P = newP

    block_of_state = {}
    for b, B in enumerate(P):
        for i in B:
            block_of_state[i] = b

    min_states = []
    for b, B in enumerate(P):
        acept = any(by_idx[i].aceptacion for i in B)
        ms = EstadoDFA(b, None, acept)
        min_states.append(ms)

    for b, B in enumerate(P):
        rep = next(iter(B))
        for k, a in enumerate(A):
            j = delta[rep][k]
            bj = block_of_state[j]
            min_states[b].trans[a] = min_states[bj]

    b0 = block_of_state[idx_of[afd2.inicio.id]]
    afd_min = AFD(min_states[b0], min_states)
    return dfa_estados_reachables(afd_min)

# =========================================================
# Entrada / salida
# =========================================================

def procesar_linea(linea, idx):
    if ';' in linea:
        izq, der = linea.split(';', 1)
        return izq.strip(), der.strip()
    return linea.strip(), None

def main():
    if len(sys.argv) != 2:
        print("Uso: python proyecto_1.py expresiones.txt")
        sys.exit(1)

    ruta = sys.argv[1]
    with open(ruta, 'r', encoding='utf-8') as f:
        lineas = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]

    for i, linea in enumerate(lineas, start=1):
        regex, w = procesar_linea(linea, i)
        print(f"\n[{i}] r = {regex}")
        if w is None:
            try:
                w = input("  Cadena w (ENTER para vacío): ")
            except EOFError:
                w = ""
        print(f"  w = {w!r}")

        try:
            tokens_conc = insertar_concatenacion(regex)
            post = a_postfija(tokens_conc)
            afn = thompson_desde_postfija(post)

            print(f"  Postfix: {stringify_postfix(post)}")

            msg = dibujar_afn(afn, f"nfa_{i}.png", f"nfa_{i}.dot")
            print(" ", msg)

            ok = acepta(afn, w)
            print(f"  w ∈ L(r) con AFN: {'sí' if ok else 'no'}")

            afd = afn_a_afd(afn)

            msg_dfa = dibujar_afd(afd, f"dfa_{i}.png", f"dfa_{i}.dot")
            print(" ", msg_dfa)

            ok_dfa = acepta_afd(afd, w)
            print(f"  w ∈ L(r) con AFD: {'sí' if ok_dfa else 'no'}")

            afd_min = minimizar_afd(afd)

            msg_min = dibujar_afd(afd_min, f"dfa_min_{i}.png", f"dfa_min_{i}.dot")
            print(" ", msg_min)

            ok_min = acepta_afd(afd_min, w)
            print(f"  w ∈ L(r) con AFD MIN: {'sí' if ok_min else 'no'}")

        except Exception as e:
            print("  Error al procesar la expresión:", e)

if __name__ == "__main__":
    main()
