import sys
import re
from collections import defaultdict, deque

# =========================================================
# Utilidades de REGEX -> Postfija
# =========================================================

PREC = {
    '*': 3, '+': 3, '?': 3,
    '.': 2,
    '|': 1
}
RIGHT_ASSOC = {'*', '+', '?'}  

def es_simbolo(c):
    return (
        c == 'ε' or
        c.isalnum() or
        c in ['_', '#']
    )

def insertar_concatenacion(regex):
    """Inserta el operador '.' donde la concatenación es implicita."""
    res = []
    for i, c1 in enumerate(regex):
        if c1 == ' ':
            continue
        res.append(c1)
        if i == len(regex)-1:
            break
        if regex[i+1] == ' ':
            continue
        c2 = regex[i+1]
        if (
            (es_simbolo(c1) or c1 in [')', '*', '+', '?']) and
            (es_simbolo(c2) or c2 in ['(', 'ε'])
        ):
            res.append('.')
    return ''.join(res)

def a_postfija(regex):
    """Convierte regex infija a postfija (shunting-yard)."""
    output = []
    opstack = []
    for c in regex:
        if c == ' ':
            continue
        if es_simbolo(c):
            output.append(c)
        elif c == '(':
            opstack.append(c)
        elif c == ')':
            while opstack and opstack[-1] != '(':
                output.append(opstack.pop())
            if not opstack:
                raise ValueError("Paréntesis desbalanceados")
            opstack.pop()
        elif c in PREC:
            while (opstack and opstack[-1] != '(' and
                   (PREC[opstack[-1]] > PREC[c] or
                    (PREC[opstack[-1]] == PREC[c] and c not in RIGHT_ASSOC))):
                output.append(opstack.pop())
            opstack.append(c)
        else:
            raise ValueError(f"Símbolo no soportado: {c!r}")

    while opstack:
        top = opstack.pop()
        if top in '()':
            raise ValueError("Paréntesis desbalanceados al final")
        output.append(top)
    return ''.join(output)

# =========================================================
# Thompson: Postfix -> AFN
# =========================================================

class Estado:
    __slots__ = ("id", "trans", "eps")
    def __init__(self, id_):
        self.id = id_
        self.trans = defaultdict(set)  
        self.eps = set()               

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

    def frag_simbolo(a):
        s = nuevo_estado()
        f = nuevo_estado()
        s.trans[a].add(f)
        estados.extend([s, f])
        pila.append((s, f))

    for c in post:
        if es_simbolo(c):
            if c == 'ε':
                s = nuevo_estado()
                f = nuevo_estado()
                s.eps.add(f)
                estados.extend([s, f])
                pila.append((s, f))
            else:
                frag_simbolo(c)
        elif c == '.':
            b1, b2 = pila.pop()
            a1, a2 = pila.pop()
            a2.eps.add(b1)
            pila.append((a1, b2))
        elif c == '|':
            b1, b2 = pila.pop()
            a1, a2 = pila.pop()
            s = nuevo_estado()
            f = nuevo_estado()
            s.eps.update([a1, b1])
            a2.eps.add(f)
            b2.eps.add(f)
            estados.extend([s, f])
            pila.append((s, f))
        elif c == '*':
            a1, a2 = pila.pop()
            s = nuevo_estado()
            f = nuevo_estado()
            s.eps.update([a1, f])
            a2.eps.update([a1, f])
            estados.extend([s, f])
            pila.append((s, f))
        elif c == '+':
            a1, a2 = pila.pop()
            s = nuevo_estado()
            f = nuevo_estado()
            s.eps.add(a1)
            a2.eps.update([a1, f])
            estados.extend([s, f])
            pila.append((s, f))
        elif c == '?':
            a1, a2 = pila.pop()
            s = nuevo_estado()
            f = nuevo_estado()
            s.eps.update([a1, f])
            a2.eps.add(f)
            estados.extend([s, f])
            pila.append((s, f))
        else:
            raise ValueError(f"Operador inesperado {c}")

    if len(pila) != 1:
        raise ValueError("Expresión inválida (sobran fragmentos)")

    inicio, acept = pila.pop()

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
# Simulacion de AFN
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
# Dibujo del AFN
# =========================================================

def dibujar_afn(afn, nombre_png, nombre_dot):
    try:
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        for s in afn.estados:
            G.add_node(s.id)

        edge_labels = defaultdict(list)
        for s in afn.estados:
            for v in s.eps:
                edge_labels[(s.id, v.id)].append('ε')
            for sym, dests in s.trans.items():
                for v in dests:
                    edge_labels[(s.id, v.id)].append(sym)
        for (u, v), labs in edge_labels.items():
            G.add_edge(u, v, label='|'.join(sorted(set(labs))))

        pos = nx.spring_layout(G, seed=42)
        accept_nodes = [afn.aceptacion.id]
        start_node = afn.inicio.id
        others = [n for n in G.nodes() if n not in accept_nodes and n != start_node]

        nx.draw_networkx_nodes(G, pos, nodelist=others)
        nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_shape='s')
        nx.draw_networkx_nodes(G, pos, nodelist=accept_nodes, node_size=900)
        nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes()})
        nx.draw_networkx_edges(G, pos, arrows=True)
        nx.draw_networkx_edge_labels(G, pos,
                                     edge_labels={(u, v): d['label'] for u, v, d in G.edges(data=True)})
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(nombre_png, dpi=170)
        plt.close()
        return f"Imagen AFN generada: {nombre_png}"
    except Exception as e:
        # Fallback: DOT
        try:
            with open(nombre_dot, 'w', encoding='utf-8') as f:
                f.write("digraph NFA {\n  rankdir=LR;\n")
                f.write(f'  node [shape=circle];\n')
                for s in afn.estados:
                    shape = "doublecircle" if s is afn.aceptacion else "circle"
                    f.write(f'  {s.id} [shape={shape}];\n')
                f.write('  start [shape=point];\n')
                f.write(f'  start -> {afn.inicio.id};\n')
                def w(u, v, lab):
                    return f'  {u} -> {v} [label="{lab}"];\n'
                for s in afn.estados:
                    for v in s.eps:
                        f.write(w(s.id, v.id, 'ε'))
                    for sym, dests in s.trans.items():
                        for v in dests:
                            f.write(w(s.id, v.id, sym))
                f.write("}\n")
            return f"No se pudo dibujar AFN con networkx/matplotlib ({type(e).__name__}). Se generó DOT: {nombre_dot}"
        except Exception as e2:
            return f"No se pudo generar imagen ni DOT del AFN: {e2}"

# =========================================================
# AFN -> AFD 
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

def dibujar_afd(afd, nombre_png, nombre_dot):
    try:
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        for s in afd.estados:
            G.add_node(s.id)

        edge_labels = defaultdict(list)
        for s in afd.estados:
            for sym, v in s.trans.items():
                edge_labels[(s.id, v.id)].append(sym)
        for (u, v), labs in edge_labels.items():
            G.add_edge(u, v, label='|'.join(sorted(set(labs))))

        pos = nx.spring_layout(G, seed=123)
        accept_nodes = [s.id for s in afd.estados if s.aceptacion]
        start_node = afd.inicio.id
        others = [n for n in G.nodes() if n not in accept_nodes and n != start_node]

        nx.draw_networkx_nodes(G, pos, nodelist=others)
        nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_shape='s')
        nx.draw_networkx_nodes(G, pos, nodelist=accept_nodes, node_size=900)
        nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes()})
        nx.draw_networkx_edges(G, pos, arrows=True)
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels={(u, v): d['label'] for u, v, d in G.edges(data=True)}
        )
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(nombre_png, dpi=170)
        plt.close()
        return f"Imagen AFD generada: {nombre_png}"
    except Exception as e:
        try:
            with open(nombre_dot, 'w', encoding='utf-8') as f:
                f.write("digraph DFA {\n  rankdir=LR;\n")
                f.write(f'  node [shape=circle];\n')
                for s in afd.estados:
                    shape = "doublecircle" if s.aceptacion else "circle"
                    f.write(f'  {s.id} [shape={shape}];\n')
                f.write('  start [shape=point];\n')
                f.write(f'  start -> {afd.inicio.id};\n')
                def w(u, v, lab):
                    return f'  {u} -> {v} [label="{lab}"];\n'
                for s in afd.estados:
                    for sym, v in s.trans.items():
                        f.write(w(s.id, v.id, sym))
                f.write("}\n")
            return f"No se pudo dibujar AFD con networkx/matplotlib ({type(e).__name__}). Se generó DOT: {nombre_dot}"
        except Exception as e2:
            return f"No se pudo generar imagen ni DOT del AFD: {e2}"

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
    """Devuelve lista de estados alcanzables desde el inicio y un mapping id->estado."""
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
    """Devuelve AFD total añadiendo (si hace falta) un estado sumidero con bucles."""
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
    """
    Minimiza un AFD usando Hopcroft:
      1) eliminar inalcanzables
      2) completar con sumidero (total)
      3) particionar (acept./no-acept.) y refinar
      4) construir AFD mínimo
    """

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

    inv = [defaultdict(set) for _ in A]  
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

    block_id = {id_block: i for i, id_block in enumerate(range(len(P)))}
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

    afd_min = dfa_estados_reachables(afd_min)
    return afd_min

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
            regl = insertar_concatenacion(regex)
            post = a_postfija(regl)
            afn = thompson_desde_postfija(post)

            print(f"  Postfix: {post}")

            # Dibujo AFN
            msg = dibujar_afn(afn, f"nfa_{i}.png", f"nfa_{i}.dot")
            print(" ", msg)

            # Simulación AFN
            ok = acepta(afn, w)
            print(f"  w ∈ L(r) con AFN: {'sí' if ok else 'no'}")

            # AFN -> AFD
            afd = afn_a_afd(afn)

            # Dibujo AFD
            msg_dfa = dibujar_afd(afd, f"dfa_{i}.png", f"dfa_{i}.dot")
            print(" ", msg_dfa)

            # Simulacion AFD
            ok_dfa = acepta_afd(afd, w)
            print(f"  w ∈ L(r) con AFD: {'sí' if ok_dfa else 'no'}")

            # AFD -> AFD MIN 
            afd_min = minimizar_afd(afd)

            # Dibujo AFD Minimizado
            msg_min = dibujar_afd(afd_min, f"dfa_min_{i}.png", f"dfa_min_{i}.dot")
            print(" ", msg_min)

            # Simulacion AFD mínimo
            ok_min = acepta_afd(afd_min, w)
            print(f"  w ∈ L(r) con AFD MIN: {'sí' if ok_min else 'no'}")

        except Exception as e:
            print("  Error al procesar la expresión:", e)


if __name__ == "__main__":
    main()
