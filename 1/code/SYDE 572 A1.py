import math
import numpy as np
import matplotlib.pyplot as plt
import os

class NumFunc:
    # Wraps f(x) and produces f'(x), f''(x) via central differences
    def __init__(self, f, h=1e-5):
        self.f = f
        self.h = h

    def __call__(self, x):
        return self.f(x)

    def d1(self, x):
        h = self.h
        return (self.f(x + h) - self.f(x - h)) / (2 * h)

    def d2(self, x):
        h = self.h
        return (self.f(x + h) - 2 * self.f(x) + self.f(x - h)) / (h ** 2)


def distSQ(x, x0, y0, f):
    return (x - x0) ** 2 + (f(x) - y0) ** 2


def dist(x, x0, y0, f):
    return distSQ(x, x0, y0, f)**0.5


# Newton-Raphson on D'(x) = 0

def find_distance_newton(x0, y0, f, df, ddf, initial_guess=0.0, tolerance=1e-7, max_iter=100):


    x = initial_guess
    history = [x]
    for _ in range(max_iter):
        D_prime = 2 * (x - x0) + 2 * (f(x) - y0) * df(x)
        D_double_prime = 2 + 2 * (df(x) ** 2) + 2 * (f(x) - y0) * ddf(x)

        next_x = x - D_prime / D_double_prime
        history.append(next_x)
        if abs(next_x - x) < tolerance:
            x = next_x
            break
        x = next_x

    shortest_distance = dist(x, x0, y0, f)
    return shortest_distance, x, history


# Golden Section Search on D(x) directly
def golden_section_search(x0, y0, f, a, b, tolerance=1e-7):

    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi

    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f_x1 = distSQ(x1, x0, y0, f)
    f_x2 = distSQ(x2, x0, y0, f)

    history = [(a, b)]
    while abs(b - a) > tolerance:
        if f_x1 < f_x2:
            b = x2
            x2 = x1
            f_x2 = f_x1
            x1 = a + resphi * (b - a)
            f_x1 = distSQ(x1, x0, y0, f)
        else:
            a = x1
            x1 = x2
            f_x1 = f_x2
            x2 = b - resphi * (b - a)
            f_x2 = distSQ(x2, x0, y0, f)
        history.append((a, b))

    best_x = (a + b) / 2
    return dist(best_x, x0, y0, f), best_x, history


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def plot_combined(x0, y0, f, hist_n, hist_g, xrange, title, fname, func_label="f(x)"):
    """
    One figure, three panels: Newton path, Golden-section brackets, D(x) with
    both methods marked. Replaces plot_newton + plot_golden + plot_D_of_x
    when you want a single file per case instead of three.
    """
    xs = np.linspace(*xrange, 400)
    ys = [f(v) for v in xs]
 
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax1, ax2, ax3 = axes
 
    # --- Panel 1: Newton on the curve ---
    ax1.plot(xs, ys, color="#2f6fed", linewidth=2, label=func_label)
    ax1.scatter([x0], [y0], color="#e63946", zorder=5, s=60, label=f"Point ({x0},{y0})")
    hx = hist_n
    hy = [f(v) for v in hx]
    ax1.plot(hx, hy, "o--", color="#f4a300", alpha=0.8, markersize=4, label="Newton iterates")
    xf, yf = hx[-1], f(hx[-1])
    ax1.plot([x0, xf], [y0, yf], color="#2a9d8f", linewidth=2.5,
              label=f"d = {dist(xf, x0, y0, f):.4f}")
    ax1.scatter([xf], [yf], color="#2a9d8f", zorder=6, s=60)
    ax1.set_title("Newton-Raphson")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
 
    # --- Panel 2: Golden section brackets on the curve ---
    ax2.plot(xs, ys, color="#2f6fed", linewidth=2, label=func_label)
    ax2.scatter([x0], [y0], color="#e63946", zorder=5, s=60, label=f"Point ({x0},{y0})")
    n = len(hist_g)
    for i, (a, b) in enumerate(hist_g):
        alpha = 0.15 + 0.7 * (i / max(n - 1, 1))
        ax2.axvspan(a, b, color="#f4a300", alpha=alpha * 0.15)
    a_final, b_final = hist_g[-1]
    xg = (a_final + b_final) / 2
    yg = f(xg)
    ax2.plot([x0, xg], [y0, yg], color="#2a9d8f", linewidth=2.5,
              label=f"d = {dist(xg, x0, y0, f):.4f}")
    ax2.scatter([xg], [yg], color="#2a9d8f", zorder=6, s=60)
    ax2.set_title(f"Golden Section ({n} iters)")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
 
    # --- Panel 3: D(x) with both methods marked ---
    Ds = [distSQ(v, x0, y0, f) for v in xs]
    ax3.plot(xs, Ds, color="#6a4c93", linewidth=2, label="D(x)")
    nD = [distSQ(v, x0, y0, f) for v in hx]
    ax3.plot(hx, nD, "o--", color="#f4a300", markersize=4, label="Newton iterates")
    ax3.axvspan(a_final, b_final, color="#2a9d8f", alpha=0.25, label="Final golden bracket")
    ax3.set_title("D(x) objective")
    ax3.set_xlabel("x"); ax3.set_ylabel("D(x)")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)
 
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
 

def plot_newton(x0, y0, f, history, xrange, title, fname, func_label="f(x)"):
    xs = np.linspace(*xrange, 400)
    ys = [f(v) for v in xs]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(xs, ys, color="#2f6fed", linewidth=2, label=func_label)
    ax.scatter([x0], [y0], color="#e63946", zorder=5, s=70, label=f"Point ({x0}, {y0})")

    # iterate path
    hx = history
    hy = [f(v) for v in hx]
    ax.plot(hx, hy, "o--", color="#f4a300", alpha=0.8, label="Newton iterates")
    for i, (hxi, hyi) in enumerate(zip(hx, hy)):
        ax.annotate(str(i), (hxi, hyi), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color="#7a4b00")
        # connector from query point to iterate (shows the distance being minimized)
        ax.plot([x0, hxi], [y0, hyi], color="#f4a300", alpha=0.15, linewidth=1)

    xf, yf = hx[-1], f(hx[-1])
    ax.plot([x0, xf], [y0, yf], color="#2a9d8f", linewidth=2.5,
             label=f"Shortest distance = {dist(xf, x0, y0, f):.4f}")
    ax.scatter([xf], [yf], color="#2a9d8f", zorder=6, s=70)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_golden(x0, y0, f, history, xrange, title, fname, func_label="f(x)"):
    xs = np.linspace(*xrange, 400)
    ys = [f(v) for v in xs]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(xs, ys, color="#2f6fed", linewidth=2, label=func_label)
    ax.scatter([x0], [y0], color="#e63946", zorder=5, s=70, label=f"Point ({x0}, {y0})")

    n = len(history)
    for i, (a, b) in enumerate(history):
        # fade older brackets, darken later ones
        alpha = 0.15 + 0.7 * (i / max(n - 1, 1))
        ax.axvspan(a, b, color="#f4a300", alpha=alpha * 0.15)

    a_final, b_final = history[-1]
    xf = (a_final + b_final) / 2
    yf = f(xf)
    ax.plot([x0, xf], [y0, yf], color="#2a9d8f", linewidth=2.5,
             label=f"Shortest distance = {dist(xf, x0, y0, f):.4f}")
    ax.scatter([xf], [yf], color="#2a9d8f", zorder=6, s=70)

    ax.set_title(title + f"\n({n} iterations, bracket shrinks left→right)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_D_of_x(x0, y0, f, xrange, newton_hist, golden_bracket_final,
                 title, fname):
    """Plot D(x) itself with Newton iterates marked on it."""
    xs = np.linspace(*xrange, 400)
    Ds = [distSQ(v, x0, y0, f) for v in xs]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, Ds, color="#6a4c93", linewidth=2, label="D(x) = squared distance")

    nx = newton_hist
    nD = [distSQ(v, x0, y0, f) for v in nx]
    ax.plot(nx, nD, "o--", color="#f4a300", label="Newton iterates")
    for i, (xi, di) in enumerate(zip(nx, nD)):
        ax.annotate(str(i), (xi, di), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)

    a, b = golden_bracket_final
    xg = (a + b) / 2
    ax.axvspan(a, b, color="#2a9d8f", alpha=0.25, label="Final golden-section bracket")
    ax.scatter([xg], [distSQ(xg, x0, y0, f)], color="#2a9d8f", zorder=5)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("D(x)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)

os.makedirs("plots", exist_ok=True)

# ============================================================
# the assignment's parabola  y = x^2 + 5
# points: (0,0), (-4,0), (-8,0), (2,0), (6,0)
f = lambda x: x ** 2 + 5
df = lambda x: 2 * x
ddf = lambda x: 2.0
 
points = [(0, 0), (-4, 0), (-8, 0), (2, 0), (6, 0)]
 
# Only make full 3-panel plots for these points (change freely).
# The rest still get computed and reported in the results table.
PLOT_THESE = {0, -4, -8, 2, 6}
 
print("=" * 65)
print("PARABOLA  y = x^2 + 5")
print("=" * 65)
print(f"{'point':>12} | {'Newton d':>10} | {'x*':>8} | {'Golden d':>10} | {'x*':>8}")
 
for (x0, y0) in points:
    d_n, x_n, hist_n = find_distance_newton(x0, y0, f, df, ddf, initial_guess=0.0)
    a, b = (min(0, x0) - 1, max(0, x0) + 1)
    d_g, x_g, hist_g = golden_section_search(x0, y0, f, a, b)
 
    print(f"({x0:>3},{y0:>2}) | {d_n:10.6f} | {x_n:8.4f} | {d_g:10.6f} | {x_g:8.4f}")
 
    if x0 in PLOT_THESE:
        span = (min(0, x0) - 2, max(0, x0) + 2)
        plot_combined(x0, y0, f, hist_n, hist_g, span,
                      title=f"Point ({x0},{y0}) to y = x^2 + 5",
                      fname=f"plots/parabola_{x0}.png",
                      func_label="y = x^2 + 5")
 
# a different parabola, to show the code generalizes
print()
print("=" * 65)
print("OTHER PARABOLA  y = 2x^2 - 3x + 1, point (5, -2)")
print("=" * 65)
f2 = lambda x: 2 * x ** 2 - 3 * x + 1
df2 = lambda x: 4 * x - 3
ddf2 = lambda x: 4.0
x0, y0 = 5, -2
d_n, x_n, hist_n = find_distance_newton(x0, y0, f2, df2, ddf2, initial_guess=0.0)
d_g, x_g, hist_g = golden_section_search(x0, y0, f2, -5, 10)
print(f"Newton:  d = {d_n:.6f} at x* = {x_n:.6f}")
print(f"Golden:  d = {d_g:.6f} at x* = {x_g:.6f}")
plot_combined(x0, y0, f2, hist_n, hist_g, (-3, 8),
              title="Point (5,-2) to y = 2x^2 - 3x + 1",
              fname="plots/other_parabola.png",
              func_label="y = 2x^2 - 3x + 1")
 
# non-polynomial functions (numerical derivatives)
print()
print("=" * 65)
print("NON-POLYNOMIAL FUNCTIONS (numerical derivatives)")
print("=" * 65)
 
nonpoly_cases = [
    ("exponential: y = e^x",     lambda x: math.exp(x),  (3, 1),    (-2, 3),     "exp"),
    ("logarithm: y = ln(x)",     lambda x: math.log(x),  (5, -3),   (0.01, 10),  "log"),
    ("square root: y = sqrt(x)", lambda x: math.sqrt(x), (2, 6),    (0.001, 12), "sqrt"),
    ("reciprocal: y = 1/x",      lambda x: 1 / x,        (4, -4),   (0.2, 10),   "recip"),
]
 
for label, fn, (x0, y0), bracket, name in nonpoly_cases:
    guess = (bracket[0] + bracket[1]) / 2
    nf = NumFunc(fn)
    d_n, x_n, hist_n = find_distance_newton(x0, y0, fn, nf.d1, nf.d2, initial_guess=guess)
    d_g, x_g, hist_g = golden_section_search(x0, y0, fn, *bracket)
    print(f"{label:28s} | Newton d={d_n:8.4f} (x*={x_n:7.4f}) | "
          f"Golden d={d_g:8.4f} (x*={x_g:7.4f})")
    plot_combined(x0, y0, fn, hist_n, hist_g, bracket,
                  title=f"{label}, point ({x0},{y0})",
                  fname=f"plots/{name}.png",
                  func_label=label)

# PART 2

import numpy as np
import matplotlib.pyplot as plt


def mse_line(m, b, xs, ys):
    xs, ys = np.asarray(xs), np.asarray(ys)
    return np.mean((ys - (m * xs + b)) ** 2)


def mse_parabola(a, b, c, xs, ys):
    xs, ys = np.asarray(xs), np.asarray(ys)
    return np.mean((ys - (a * xs ** 2 + b * xs + c)) ** 2)


# ---------------------------------------------------------------------
# LINE FIT: y = m x + b
# ---------------------------------------------------------------------
def fit_line_coordinate_descent(xs, ys, m0=0.0, b0=0.0, n_iter=15):
    """Alternately solve for m (fixing b), then b (fixing m)."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    n = len(xs)
    sx, sy = xs.sum(), ys.sum()
    sx2, sxy = (xs ** 2).sum(), (xs * ys).sum()
    xbar, ybar = sx / n, sy / n

    m, b = m0, b0
    history = [(m, b, mse_line(m, b, xs, ys))]
    for _ in range(n_iter):
        m = (sxy - b * sx) / sx2
        b = ybar - m * xbar
        history.append((m, b, mse_line(m, b, xs, ys)))
    return m, b, history


def fit_line_newton(xs, ys, m0=0.0, b0=0.0, n_iter=5):
    """Full multivariate Newton-Raphson on (m, b). Converges in 1 step
    since MSE is quadratic, but we still iterate to show the (trivial)
    convergence history."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    n = len(xs)

    m, b = m0, b0
    history = [(m, b, mse_line(m, b, xs, ys))]
    for _ in range(n_iter):
        resid = ys - (m * xs + b)
        grad = np.array([
            -(2 / n) * np.sum(xs * resid),
            -(2 / n) * np.sum(resid)
        ])
        H = (2 / n) * np.array([
            [np.sum(xs ** 2), np.sum(xs)],
            [np.sum(xs),      n]
        ])
        step = np.linalg.solve(H, grad)
        m, b = np.array([m, b]) - step
        history.append((m, b, mse_line(m, b, xs, ys)))
        if np.linalg.norm(step) < 1e-12:
            break
    return m, b, history


# ---------------------------------------------------------------------
# PARABOLA FIT: y = a x^2 + b x + c
# ---------------------------------------------------------------------
def fit_parabola_coordinate_descent(xs, ys, a0=0.0, b0=0.0, c0=0.0, n_iter=20):
    """Cycle through a, b, c: fix the other two, solve the third exactly."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    n = len(xs)
    sx  = xs.sum();        sx2 = (xs**2).sum()
    sx3 = (xs**3).sum();   sx4 = (xs**4).sum()
    sy  = ys.sum()
    sxy = (xs*ys).sum();   sx2y = (xs**2*ys).sum()

    a, b, c = a0, b0, c0
    history = [(a, b, c, mse_parabola(a, b, c, xs, ys))]
    for _ in range(n_iter):
        a = (sx2y - b*sx3 - c*sx2) / sx4
        b = (sxy - a*sx3 - c*sx) / sx2
        c = (sy - a*sx2 - b*sx) / n
        history.append((a, b, c, mse_parabola(a, b, c, xs, ys)))
    return a, b, c, history


def fit_parabola_newton(xs, ys, a0=0.0, b0=0.0, c0=0.0, n_iter=5):
    """Full multivariate Newton-Raphson on (a, b, c)."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    n = len(xs)

    params = np.array([a0, b0, c0], dtype=float)
    history = [(*params, mse_parabola(*params, xs, ys))]
    for _ in range(n_iter):
        a, b, c = params
        resid = ys - (a*xs**2 + b*xs + c)
        grad = -(2/n) * np.array([
            np.sum(resid * xs**2),
            np.sum(resid * xs),
            np.sum(resid)
        ])
        H = (2/n) * np.array([
            [np.sum(xs**4), np.sum(xs**3), np.sum(xs**2)],
            [np.sum(xs**3), np.sum(xs**2), np.sum(xs)],
            [np.sum(xs**2), np.sum(xs),    n]
        ])
        step = np.linalg.solve(H, grad)
        params = params - step
        history.append((*params, mse_parabola(*params, xs, ys)))
        if np.linalg.norm(step) < 1e-12:
            break
    return (*params, history)


def plot_line_fit_progress(xs, ys, history, title, fname, show_every=1):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    xs_arr = np.array(xs)
    xr = np.linspace(xs_arr.min() - 1, xs_arr.max() + 1, 100)
    n = len(history)
    for i, (m, b, e) in enumerate(history):
        if i % show_every != 0 and i != n - 1:
            continue
        alpha = 0.15 + 0.75 * (i / (n - 1))
        color = "#2a9d8f" if i == n - 1 else "#f4a300"
        lw = 2.5 if i == n - 1 else 1.2
        label = f"final: y={m:.3f}x+{b:.3f}" if i == n - 1 else (f"iter {i}" if i == 0 else None)
        ax1.plot(xr, m * xr + b, color=color, alpha=alpha, linewidth=lw, label=label)
    ax1.scatter(xs, ys, color="#e63946", zorder=5, s=60, label="data")
    ax1.set_title("Line fit — iterations")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    errs = [e for (_, _, e) in history]
    ax2.plot(range(len(errs)), errs, "o-", color="#6a4c93")
    ax2.set_yscale("log")
    ax2.set_title("MSE vs iteration")
    ax2.set_xlabel("iteration"); ax2.set_ylabel("MSE (log scale)")
    ax2.grid(alpha=0.3)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_parabola_fit_progress(xs, ys, history, title, fname, show_every=2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    xs_arr = np.array(xs)
    xr = np.linspace(xs_arr.min() - 1, xs_arr.max() + 1, 100)
    n = len(history)
    for i, (a, b, c, e) in enumerate(history):
        if i % show_every != 0 and i != n - 1:
            continue
        alpha = 0.15 + 0.75 * (i / (n - 1))
        color = "#2a9d8f" if i == n - 1 else "#f4a300"
        lw = 2.5 if i == n - 1 else 1.2
        label = f"final: y={a:.3f}x²+{b:.3f}x+{c:.3f}" if i == n - 1 else (f"iter {i}" if i == 0 else None)
        ax1.plot(xr, a * xr ** 2 + b * xr + c, color=color, alpha=alpha, linewidth=lw, label=label)
    ax1.scatter(xs, ys, color="#e63946", zorder=5, s=60, label="data")
    ax1.set_title("Parabola fit — iterations")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    errs = [e for (_, _, _, e) in history]
    ax2.plot(range(len(errs)), errs, "o-", color="#6a4c93")
    ax2.set_yscale("log")
    ax2.set_title("MSE vs iteration")
    ax2.set_xlabel("iteration"); ax2.set_ylabel("MSE (log scale)")
    ax2.grid(alpha=0.3)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_line_param_path(xs, ys, history_cd, history_newton, title, fname):
    xs, ys = np.asarray(xs), np.asarray(ys)
    ms = np.linspace(-1, 5, 200)
    bs = np.linspace(-4, 3, 200)
    M, B = np.meshgrid(ms, bs)
    Z = np.zeros_like(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            Z[i, j] = mse_line(M[i, j], B[i, j], xs, ys)
    fig, ax = plt.subplots(figsize=(7, 6))
    cs = ax.contour(M, B, Z, levels=25, cmap="viridis", alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=7)
    cd_m = [h[0] for h in history_cd]
    cd_b = [h[1] for h in history_cd]
    ax.plot(cd_m, cd_b, "o-", color="#f4a300", markersize=4, label="Coordinate descent path")
    nt_m = [h[0] for h in history_newton]
    nt_b = [h[1] for h in history_newton]
    ax.plot(nt_m, nt_b, "s-", color="#e63946", markersize=8, label="Multivariate Newton path")
    ax.scatter([2.3], [-0.2], color="#2a9d8f", marker="*", s=200, zorder=6, label="Closed-form solution")
    ax.set_xlabel("m (slope)")
    ax.set_ylabel("b (intercept)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


xs2 = [0, 2, 1, 3]
ys2 = [0.5, 3.5, 1.5, 7.5]

print("=" * 60)
print("LINE FIT: y = m x + b")
print("=" * 60)
m_cd, b_cd, hist_cd = fit_line_coordinate_descent(xs2, ys2, n_iter=15)
print(f"Coordinate descent -> m={m_cd:.6f}, b={b_cd:.6f}, MSE={mse_line(m_cd,b_cd,xs2,ys2):.6f}")
m_nt, b_nt, hist_nt = fit_line_newton(xs2, ys2, n_iter=3)
print(f"Multivariate Newton -> m={m_nt:.6f}, b={b_nt:.6f}, MSE={mse_line(m_nt,b_nt,xs2,ys2):.6f}  (converges in 1 step)")
plot_line_fit_progress(xs2, ys2, hist_cd, "Line fit — coordinate descent", "plots/line_coord_descent.png", show_every=2)
plot_line_param_path(xs2, ys2, hist_cd, hist_nt, "MSE(m,b) contours + optimization paths", "plots/line_param_path.png")

print()
print("=" * 60)
print("PARABOLA FIT: y = a x^2 + b x + c")
print("=" * 60)
a_cd, b2_cd, c_cd, hist_p_cd = fit_parabola_coordinate_descent(xs2, ys2, n_iter=20)
print(f"Coordinate descent -> a={a_cd:.6f}, b={b2_cd:.6f}, c={c_cd:.6f}, MSE={mse_parabola(a_cd,b2_cd,c_cd,xs2,ys2):.6f}")
a_nt, b2_nt, c_nt, hist_p_nt = fit_parabola_newton(xs2, ys2, n_iter=3)
print(f"Multivariate Newton -> a={a_nt:.6f}, b={b2_nt:.6f}, c={c_nt:.6f}, MSE={mse_parabola(a_nt,b2_nt,c_nt,xs2,ys2):.6f}  (converges in 1 step)")
plot_parabola_fit_progress(xs2, ys2, hist_p_cd, "Parabola fit — coordinate descent", "plots/parabola_coord_descent.png", show_every=3)