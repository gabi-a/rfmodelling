import numpy as np
import skrf as rf
from skrf.circuit import Circuit
from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution


# ------------------------------------------------------------
# Resonator parameters
# ------------------------------------------------------------

# f0 = 500e6
# w0 = 2 * np.pi * f0

# Q0 = 600
# beta = 1.0

# L = 21.4e-9
# Z_0 = 50
# Ngaps = 4

# Ceq = 1 / (w0**2 * L)
# Cgap = Ngaps * Ceq

# Lseg = L / Ngaps

# Rs_total = w0 * L / Q0
# Rseg = Rs_total / Ngaps

# Rpar = Q0 * w0 * L
# Rgap = Rpar / Ngaps**2

# Zdiff_target = 2 * Z_0

# # Fixed coupling capacitors
# Cc_eff_est = np.sqrt(beta / (Rgap * Zdiff_target * w0**2))
# Cc_leg_fixed = 2 * Cc_eff_est

# print(f"Ceq          = {Ceq:.4e} F = {Ceq * 1e12:.3f} pF")
# print(f"Cgap         = {Cgap:.4e} F = {Cgap * 1e12:.3f} pF")
# print(f"Lseg         = {Lseg:.4e} H = {Lseg * 1e9:.3f} nH")
# print(f"Rseg         = {Rseg:.4f} ohm")
# print(f"Rpar         = {Rpar:.2f} ohm")
# print(f"Rgap         = {Rgap:.2f} ohm")
# print(f"Cc_leg_fixed = {Cc_leg_fixed:.4e} F = {Cc_leg_fixed * 1e15:.2f} fF per side")


# ------------------------------------------------------------
# Frequency and media
# ------------------------------------------------------------

# freq = rf.Frequency(start=430, stop=570, unit='MHz', npoints=3001)
# media = rf.media.DefinedGammaZ0(freq, z0=Z_0)

# f_array = freq.f
# idx_f0 = np.argmin(np.abs(f_array - f0))


# ------------------------------------------------------------
# Ideal 180-degree hybrid / balun
# ------------------------------------------------------------

def ideal_180_hybrid(freq, z0=50, name='ideal_180_hybrid'):
    nfreq = len(freq)
    s = np.zeros((nfreq, 4, 4), dtype=complex)

    a = 1 / np.sqrt(2)

    Smat = a * np.array([
        [0,  1, -1,  0],
        [1,  0,  0,  1],
        [-1, 0,  0,  1],
        [0,  1,  1,  0],
    ], dtype=complex)

    s[:] = Smat

    return rf.Network(
        frequency=freq,
        s=s,
        z0=z0 * np.ones((nfreq, 4)),
        name=name
    )


# ------------------------------------------------------------
# Helpers for tunable elements
# ------------------------------------------------------------

def make_series_element(media, kind, value, name):
    if kind == "C":
        return media.capacitor(value, name=name)
    elif kind == "L":
        return media.inductor(value, name=name)
    else:
        raise ValueError("kind must be 'C' or 'L'")


def make_shunt_element(media, kind, value, name):
    if kind == "C":
        return media.capacitor(value, name=name)
    elif kind == "L":
        return media.inductor(value, name=name)
    else:
        raise ValueError("kind must be 'C' or 'L'")


def format_value(kind, value):
    if kind == "C":
        return f"{value * 1e12:.4f} pF"
    else:
        return f"{value * 1e9:.4f} nH"


# ------------------------------------------------------------
# Balanced LGR with arbitrary series/shunt tuning topology
# ------------------------------------------------------------

def make_balanced_tuned_lgr(media, freq, series_kind, series_value, shunt_kind, shunt_value, Cc_leg_fixed, Cgap, Rseg, Lseg, Z0) -> Circuit:
    """
    Floating balanced 4-gap LGR driven through an ideal 180-degree hybrid.

    Tuning network:
        shunt element is connected across the differential pair.
        one equal series element is inserted in each balanced leg.

    The shunt element is placed after the series elements, closer to the load.
    This is usually the better L-match orientation for a resonator load.
    """

    port1 = Circuit.Port(freq, 'port1', z0=Z0)
    gnd = Circuit.Ground(freq, name='gnd')

    balun = ideal_180_hybrid(freq, z0=Z0, name='balun')

    Riso = media.resistor(Z0, name='Riso')

    Xs_plus = make_series_element(media, series_kind, series_value, f'{series_kind}_plus')
    Xs_minus = make_series_element(media, series_kind, series_value, f'{series_kind}_minus')

    Xsh = make_shunt_element(media, shunt_kind, shunt_value, name=f'{shunt_kind}_shunt')

    Cc_plus = media.capacitor(Cc_leg_fixed, name='Cc_plus')
    Cc_minus = media.capacitor(Cc_leg_fixed, name='Cc_minus')

    Cg1 = media.capacitor(Cgap, name='Cg1')
    Cg2 = media.capacitor(Cgap, name='Cg2')
    Cg3 = media.capacitor(Cgap, name='Cg3')
    Cg4 = media.capacitor(Cgap, name='Cg4')

    R1 = media.resistor(Rseg, name='R1')
    R2 = media.resistor(Rseg, name='R2')
    R3 = media.resistor(Rseg, name='R3')
    R4 = media.resistor(Rseg, name='R4')

    L1 = media.inductor(Lseg, name='L1')
    L2 = media.inductor(Lseg, name='L2')
    L3 = media.inductor(Lseg, name='L3')
    L4 = media.inductor(Lseg, name='L4')

    cnx = [
        # Single-ended input into hybrid
        [(port1, 0), (balun, 0)],

        # Isolated port terminated
        [(balun, 3), (Riso, 0)],
        [(Riso, 1), (gnd, 0)],

        # Series tuning elements at balanced output
        [(balun, 1), (Xs_plus, 0)],
        [(balun, 2), (Xs_minus, 0)],

        # Differential shunt tuning element after series elements
        [(Xs_plus, 1), (Xsh, 0), (Cc_plus, 0)],
        [(Xs_minus, 1), (Xsh, 1), (Cc_minus, 0)],

        # Floating LGR driven gap nodes
        [(Cc_plus, 1), (Cg1, 0), (L4, 1)],
        [(Cc_minus, 1), (Cg1, 1), (R1, 0)],

        # Segment 1
        [(R1, 1), (L1, 0)],
        [(L1, 1), (Cg2, 0)],

        # Segment 2
        [(Cg2, 1), (R2, 0)],
        [(R2, 1), (L2, 0)],
        [(L2, 1), (Cg3, 0)],

        # Segment 3
        [(Cg3, 1), (R3, 0)],
        [(R3, 1), (L3, 0)],
        [(L3, 1), (Cg4, 0)],

        # Segment 4, closing floating loop
        [(Cg4, 1), (R4, 0)],
        [(R4, 1), (L4, 0)],
    ]

    return Circuit(cnx)


import networkx as nx
import plotly.graph_objects as go


def plot_circuit_3d(circuit):
    """Extracts the graph from a scikit-rf Circuit object and plots it

    in an interactive 3D window using Plotly.
    """
    # 1. Extract the underlying NetworkX graph
    G = circuit.G

    # 2. Compute 3D positions using a spring layout engine
    pos = nx.spring_layout(G, dim=3, seed=42)

    # --- Prepare Node Data ---
    Xn, Yn, Zn = [], [], []
    node_colors = []
    node_labels = []

    for node in G.nodes():
        Xn.append(pos[node][0])
        Yn.append(pos[node][1])
        Zn.append(pos[node][2])
        node_labels.append(str(node))

        # Color-code nodes dynamically based on type
        node_str = str(node).lower()
        if "port" in node_str:
            node_colors.append("#FF5722")  # Vibrant Orange for External Ports
        elif "ground" in node_str or "gnd" in node_str:
            node_colors.append("#4CAF50")  # Green for Ground intersections
        else:
            node_colors.append("#2196F3")  # Blue for regular RF Components

    # --- Prepare Edge Data & Midpoint Labels ---
    Xe, Ye, Ze = [], [], []
    Xm, Ym, Zm = [], [], []
    edge_labels = []

    for u, v, d in G.edges(data=True):
        # Line paths (separated by None so Plotly draws individual lines)
        Xe += [pos[u][0], pos[v][0], None]
        Ye += [pos[u][1], pos[v][1], None]
        Ze += [pos[u][2], pos[v][2], None]

        # Calculate midpoints for edge labels
        Xm.append((pos[u][0] + pos[v][0]) / 2)
        Ym.append((pos[u][1] + pos[v][1]) / 2)
        Zm.append((pos[u][2] + pos[v][2]) / 2)

        # Clean up the edge port labels from scikit-rf data
        port_num = d.get("port", "")
        edge_labels.append(f"P{port_num}" if port_num != "" else "")

    # --- Build Plotly Traces ---
    # Edges (Lines)
    trace_edges = go.Scatter3d(
        x=Xe,
        y=Ye,
        z=Ze,
        mode="lines",
        line=dict(color="#CFD8DC", width=3),
        hoverinfo="none",
    )

    # Nodes (Markers + Labels)
    trace_nodes = go.Scatter3d(
        x=Xn,
        y=Yn,
        z=Zn,
        mode="markers+text",
        marker=dict(
            symbol="circle",
            size=10,
            color=node_colors,
            line=dict(color="#FFFFFF", width=1),
        ),
        text=node_labels,
        textposition="top center",
        hoverinfo="text",
        textfont=dict(size=11, color="#263238"),
    )

    # Edge Labels (Text centered on the connections)
    trace_edge_labels = go.Scatter3d(
        x=Xm,
        y=Ym,
        z=Zm,
        mode="text",
        text=edge_labels,
        textposition="middle center",  # <-- Fixed this line
        hoverinfo="none",
        textfont=dict(size=9, color="#E91E63"),
    )

    # --- Layout & Axis Styling ---
    # We hide background grids to keep the topology clear
    no_axis = dict(
        showbackground=False,
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        title="",
    )

    layout = go.Layout(
        title=dict(
            text="Interactive 3D Circuit Topology",
            x=0.5,
            font=dict(size=16),
        ),
        width=900,
        height=800,
        showlegend=False,
        scene=dict(xaxis=no_axis, yaxis=no_axis, zaxis=no_axis),
        margin=dict(t=60, b=10, l=10, r=10),
        hovermode="closest",
    )

    # Compile and show
    fig = go.Figure(
        data=[trace_edges, trace_nodes, trace_edge_labels], layout=layout
    )
    fig.show()