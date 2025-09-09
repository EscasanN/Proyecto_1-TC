# Proyecto 1 - Teoria de la Computacion

Proyecto 1 de la clase de Teoria de la Computacion.
 [link del video presentación](https://uvggt-my.sharepoint.com/:v:/g/personal/ram23601_uvg_edu_gt/ESpP-Ia1K9hJhXrm3QHWmkEB6tY6SR-FdS3fBg8Raho0Qg?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=e4sKky)

 # Hecho por:
 - Eliazar Canastuj
 - Nelson Escalante
 - Diego Ramírez

---

## ✨ Características

* Conversión **infix → postfix** con *shunting-yard*.
* Construcción de **AFN (Thompson)**.
* **Simulación** del AFN.
* Conversión **AFN → AFD** (construcción por subconjuntos).
* **Minimización** del AFD (Hopcroft).
* **Render** de AFN/AFD/AFD-min en **PNG** y **DOT** (Graphviz si está disponible; si no, *fallback* con NetworkX/Matplotlib).
*.

---

## 📦 Requisitos

* **Python** 3.8+ (recomendado 3.10+)
* Paquetes Python:

  * `graphviz` (opcional para render de alta calidad)
  * `networkx` y `matplotlib` (usados como *fallback*)

Instalación rápida:

```bash
pip install graphviz networkx matplotlib
```

> Para usar Graphviz “de verdad” (no solo la librería Python), instala también el **binario** del sistema. Si no lo tienes, el programa usará el *fallback* con NetworkX/Matplotlib.

---

## ▶️ Uso

1. Crea un archivo de entrada, por ejemplo `expresiones.txt`, con una o más líneas. Cada línea puede ser:

   * `REGEX ; w`  → regex y cadena de prueba en la misma línea
   * `REGEX`      → solo regex; la cadena `w` se pedirá por consola

   Las líneas que empiezan con `#` se ignoran como comentarios.

2. Ejecuta:

```bash
python proyecto_1.py expresiones.txt
```

El programa imprime la regex, su **postfix**, y si `w` está en el lenguaje **con AFN / AFD / AFD MIN**. También genera los archivos PNG/DOT.



---

## 🗂️ Formato de entrada

Ejemplo de `expresiones.txt`:

```text
# 1) Alternativa y cierre
(a|b)*abb ; abb

# 2) Escapes de operadores y literales
\(a\|b\)\* ; (a|b)*

# 3) Uso de ε
(a|ε)b? ; ab
(a|ε)b? ; b

# 4) Literales entre texto y opcional
if(a|x|t)+\{y\}(else\{n\})? ; ifatx{y}else{n}

# 5) Tu ejemplo con bloques y '*'
\?(((.|ε)?!?)\*)+ ; ?*.*.*
```

---

## 🖼️ Salida generada

Para cada línea (empezando en 1):

* `nfa_1.png` / `nfa_1.dot` → AFN
* `dfa_1.png` / `dfa_1.dot` → AFD
* `dfa_min_1.png` / `dfa_min_1.dot` → AFD minimizado

Si Graphviz está disponible, los PNG vienen con flechas y layout optimizados; si no, se genera igualmente con NetworkX/Matplotlib.

---

## ✅ Ejemplos rápidos

* **Acepta**: `\?(((.|ε)?!?)\*)+` con `w = ?*` o `w = ?.*!*` (tras el `?` inicial, uno o más bloques que siempre terminan en `*`).

* **Rechaza**: la misma regex con `w = ?.,` (no aparece `*` para cerrar el primer bloque).

* **Acepta**: `if(a|x|t)+\{y\}(else\{n\})?` con `w = ifatx{y}` o `w = ift{y}else{n}`.

* **Rechaza**: la misma regex con `w = if(a){y}` (paréntesis no están permitidos como tales, salvo que los escapes como literales).

---



## 📁 Estructura

* `proyecto_1.py` — script principal.
* `nfa_*.png|dot`, `dfa_*.png|dot`, `dfa_min_*.png|dot` — artefactos de salida.

---
