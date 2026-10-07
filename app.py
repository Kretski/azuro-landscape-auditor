# -*- coding: utf-8 -*-
"""
AZURO Landscape Auditor — Pre-Solve Demo (revised wording)
===========================================================
Two pre-solve checks, computable from a few numbers, no solver needed.
They have DIFFERENT evidential status, and the UI says so:

  1. Penalty check (one-hot constraints)
     - RIGOROUS part: if P > g_max, where
           g_max = max_i ( |c_i| + sum_j |J_ij| )
       (largest total coefficient magnitude touching one variable), then
       every single-flip local minimum has all one-hot groups feasible.
       Reason: flipping one variable changes the objective by at most
       g_max, while repairing a group with sum 0 or 2 lowers the penalty
       by P (and by >= 3P when the sum is >= 3). Sufficient, conservative.
     - EMPIRICAL part: in one small synthetic family (25 groups x 4
       variables, greedy local search + restarts) the penalty at which 50%
       of runs were feasible was about 1.25 x max coefficient with ~3
       couplings per variable, and about 2.1 x with ~12 couplings per
       variable. It depends on coupling density and on the solver. It is
       NOT a general constant and NOT validated beyond that family.

  2. Correlation check (continuous weights)
     - DERIVED, not measured: for equal-variance, equicorrelated assets the
       risk Hessian restricted to sum-preserving reallocations has
       curvature proportional to (1 - rho). For a FIXED ABSOLUTE tolerance
       the near-optimal region therefore has radius ~ 1/sqrt(1 - rho).
       This is geometry of a quadratic form. In our own synthetic
       sweep the fitted exponent was about -0.83 rather than -0.5; one
       possible reason is the tolerance definition (relative instead of
       absolute), which we have not tested. Ignores long-only/box constraints and unequal
       variances. Treat it as qualitative: the region widens without bound
       as rho -> 1.

No real client data is needed. The numbers you enter are processed on the
Space server; this app does not store them. For a post-solve audit (P(q),
Entry->x* gap) on your own solver output, use the contact link at the bottom.
"""

import gradio as gr
import numpy as np

try:  # available on Hugging Face Spaces, missing when run locally
    import spaces
    _gpu = spaces.GPU
except ImportError:
    def _gpu(fn):
        return fn

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

.gradio-container {
    font-family: 'IBM Plex Sans', sans-serif !important;
    max-width: 880px !important;
    margin: auto !important;
}
/* Cream background only in light mode. In dark mode the theme's light text would be
   invisible on it (hidden tab label, hidden intro text), so the theme background is kept. */
html:not(.dark) body:not(.dark) .gradio-container:not(.dark) {
    background: #f7f5f0 !important;
}
h1, h2, h3 {
    font-family: 'Source Serif 4', serif !important;
}
.verdict-ok { background: #dde8e0; border-left: 4px solid #2f5d42; padding: 16px 20px; color: #23241f !important; }
.verdict-caution { background: #f2e6cd; border-left: 4px solid #9c6b1f; padding: 16px 20px; color: #23241f !important; }
.verdict-flag { background: #f0ddd7; border-left: 4px solid #8c3b2e; padding: 16px 20px; color: #23241f !important; }
.verdict-ok b, .verdict-caution b, .verdict-flag b { color: #23241f !important; }
/* In dark mode the grey helper text (#6b6a5f) is too dim; lighten it. */
.dark [style*="6b6a5f"] { color: #b9b7ab !important; }
footer { display: none !important; }
"""

BRAND_HEADER = """
<div style="border-bottom: 2px solid #23241f; padding-bottom: 12px; margin-bottom: 20px;
            display: flex; justify-content: space-between; align-items: baseline;">
  <span style="font-family: 'Source Serif 4', serif; font-weight: 700; font-size: 24px;">AZURO</span>
  <span style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #6b6a5f;">
    Pre-Solve Checks — Interactive Demo
  </span>
</div>
<p style="font-size: 14px; color: #6b6a5f; line-height: 1.6;">
Two quick checks computed from a few numbers, before you run a solver. They are not equally
strong: the penalty check has a rigorous sufficient condition plus an empirical rule of thumb
from one small synthetic test family; the correlation check is a derived geometric statement,
not a measurement. Each result says which is which. The numbers you enter are processed on the
Space server; this app does not store them.
</p>
"""

# Empirical reference points (one synthetic one-hot family, greedy local search).
# Feasible fraction of runs vs penalty / max-coefficient, small samples:
#   ~3 couplings/variable : 0.85x -> 0%,  1.0x -> 17-37%, 1.25x -> 25-50%, 1.5x -> 54-77%, 2.0x -> 100%
#   ~12 couplings/variable: <=1.5x -> 0%, 2.0x -> 46%,   3.0x -> 100%
EMP_LOW = 1.0    # below this ratio: feasibility <= ~37% in both tested families
EMP_HIGH = 3.0   # at/above this ratio: no infeasible runs seen in either family


# ----------------------------------------------------------------------
# Check 1 — Penalty (one-hot constraints)
# ----------------------------------------------------------------------
def check_penalty(penalty_weight, max_coef, g_max):
    if max_coef <= 0 or g_max <= 0:
        return "<p>Both coefficient inputs must be positive.</p>"
    if g_max < max_coef:
        return ("<p>The largest total magnitude touching one variable (|c_i| + &Sigma;|J_ij|) "
                "cannot be smaller than the largest single coefficient. Please adjust.</p>")

    ratio = penalty_weight / max_coef
    # g_max <= (1 + couplings) * max_coef, so couplings >= g_max/max_coef - 1
    min_couplings = g_max / max_coef - 1
    dense_note = ""
    if min_couplings > 12:
        dense_note = (f" <b>Note:</b> g_max / max coefficient = {g_max / max_coef:.1f} implies at "
                      f"least about {min_couplings:.0f} couplings per variable, denser than the "
                      f"~3 and ~12 per variable we tested, so the empirical thresholds may not apply.")

    if penalty_weight > g_max:
        cls, verdict = "verdict-ok", "SUFFICIENT — above the rigorous bound"
        detail = (f"P = <b>{penalty_weight:.2f}</b> exceeds g_max = <b>{g_max:.2f}</b>. For one-hot "
                  f"(sum&nbsp;=&nbsp;1) groups, every single-flip local minimum is then feasible. "
                  f"This bound is conservative: it grows with how strongly each variable is coupled, "
                  f"while the empirical 50% point in our small tests was only about "
                  f"1.25&ndash;2.1&times; the largest single coefficient. A smaller P may "
                  f"therefore also work, but only this bound is a guarantee.")
    elif ratio < EMP_LOW:
        cls, verdict = "verdict-flag", "FLAG — well below empirical thresholds"
        detail = (f"P / max coefficient = <b>{ratio:.2f}&times;</b>, and P is below the rigorous bound "
                  f"({g_max:.2f}). In our small synthetic tests, at ratios below {EMP_LOW:.1f}&times; "
                  f"at most about a third of local-search runs returned feasible solutions. "
                  f"Expect many constraint violations.")
    elif ratio < EMP_HIGH:
        cls, verdict = "verdict-caution", "UNCERTAIN — depends on your problem"
        detail = (f"P / max coefficient = <b>{ratio:.2f}&times;</b>; P is below the rigorous bound "
                  f"({g_max:.2f}). In our tests this range gave anywhere from 0% to 100% feasible runs: "
                  f"the 50% point was about 1.25&times; with ~3 couplings per variable and about "
                  f"2.1&times; with ~12. Denser coupling needs a larger penalty. Check feasibility "
                  f"directly on your own solver output.")
    else:
        cls, verdict = "verdict-caution", "NOT GUARANTEED — no violations seen in small tests"
        detail = (f"P / max coefficient = <b>{ratio:.2f}&times;</b>. In our small tests no infeasible runs "
                  f"occurred at ratios &ge; {EMP_HIGH:.1f}&times;, but P is still below the rigorous bound "
                  f"({g_max:.2f}), so feasibility is not guaranteed for your problem.")

    if penalty_weight <= g_max:
        detail += dense_note

    return f"""
    <div class="{cls}">
    <b>{verdict}</b><br><br>{detail}
    </div>
    <p style="font-size:12px; color:#6b6a5f; margin-top:10px;">
    Rigorous part: P &gt; max_i(|c_i| + &Sigma;_j|J_ij|) &rArr; all single-flip
    local minima satisfy every one-hot constraint (flipping a variable moves the objective by at most that
    amount; repairing a group with sum 0 or 2 lowers the penalty by P). Empirical part: one small synthetic
    family (25&times;4 one-hot groups), one greedy local-search solver; thresholds are solver- and
    density-dependent, not universal constants. The bound assumes disjoint one-hot groups, a penalty of
    the form P&middot;(&Sigma;x&minus;1)&sup2;, and c_i, J_ij taken from the objective
    without the penalty terms.
    </p>
    """


# ----------------------------------------------------------------------
# Check 2 — Correlated-asset degeneracy (derived, qualitative)
# ----------------------------------------------------------------------
def check_correlation(rho):
    rho_calc = min(rho, 0.995)
    curvature = 1 - rho_calc
    multiple = 1.0 / np.sqrt(curvature)   # relative to rho = 0

    if rho < 0.5:
        cls, verdict = "verdict-ok", "Low redundancy (illustrative band)"
        detail = (f"At &rho;={rho:.2f}, the near-optimal region along weight reallocations is "
                  f"<b>{multiple:.1f}&times;</b> the uncorrelated radius (fixed absolute tolerance). "
                  f"Weights inside this group stay comparatively well determined.")
    elif rho < 0.8:
        cls, verdict = "verdict-caution", "Moderate redundancy (illustrative band)"
        detail = (f"At &rho;={rho:.2f}, the near-optimal region is <b>{multiple:.1f}&times;</b> the "
                  f"uncorrelated radius. How weight is split inside this cluster is becoming less "
                  f"determined by the objective.")
    else:
        cls, verdict = "verdict-flag", "High redundancy (illustrative band)"
        detail = (f"At &rho;={rho:.2f}, the near-optimal region is <b>{multiple:.1f}&times;</b> the "
                  f"uncorrelated radius. Assets in this cluster are close to interchangeable under the "
                  f"risk term; the exact split a solver returns carries little information.")

    return f"""
    <div class="{cls}">
    <b>{verdict}</b><br><br>{detail}
    </div>
    <p style="font-size:12px; color:#6b6a5f; margin-top:10px;">
    Derived, not measured: for equal-variance, equicorrelated assets the risk curvature along
    sum-preserving reallocations is proportional to (1&minus;&rho;), so for a fixed absolute tolerance the
    near-optimal radius scales as 1/&radic;(1&minus;&rho;). In our own synthetic sweep the fitted
    exponent was about &minus;0.83 rather than &minus;0.5; one possible reason is the tolerance
    definition (relative instead of absolute), which we have not tested. Assumes unconstrained weights
    (no long-only/box limits) and equal variances. The band edges (0.5, 0.8) are illustrative conventions.
    </p>
    """


@_gpu
def correlation_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rho_vals = np.linspace(0, 0.99, 200)
    spread = 1.0 / np.sqrt(1 - rho_vals)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(rho_vals, spread, color="#8c3b2e", linewidth=2)
    ax.set_xlabel("Correlation \u03c1")
    ax.set_ylabel("Relative near-optimal radius")
    ax.set_title("Theory curve (fixed absolute tolerance), not measured data", fontsize=11)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
# Gradio 6 moved `css` from Blocks() to launch(); support 5.x (the Space) and 6.x (local runs).
_GRADIO_MAJOR = int(gr.__version__.split(".")[0])
_BLOCKS_KW = {} if _GRADIO_MAJOR >= 6 else {"css": CUSTOM_CSS}
_LAUNCH_KW = {"css": CUSTOM_CSS} if _GRADIO_MAJOR >= 6 else {}

with gr.Blocks(title="AZURO — Pre-Solve Checks", **_BLOCKS_KW) as demo:
    gr.HTML(BRAND_HEADER)

    with gr.Tab("Penalty Check (one-hot)"):
        gr.Markdown(
            "For QUBO one-hot constraints encoded as penalty terms. Enter your penalty, your largest "
            "single coefficient, and the largest total coefficient magnitude touching any one variable "
            "(|c_i| + sum of |J_ij| over that variable's couplings). The rigorous bound assumes that "
            "each variable belongs to exactly one one-hot group, that the penalty has the form "
            "P*(sum of the group's variables - 1)^2, and that c_i and J_ij come from the objective "
            "only, without the penalty terms."
        )
        with gr.Row():
            penalty_input = gr.Slider(0.1, 30, value=2.0, step=0.05, label="Penalty weight (P)")
            scale_input = gr.Slider(0.1, 10, value=1.0, step=0.05,
                                    label="Largest single coefficient magnitude")
            gmax_input = gr.Slider(0.1, 100, value=5.0, step=0.1,
                                   label="Largest total magnitude touching one variable (g_max)")
        penalty_output = gr.HTML()
        penalty_btn = gr.Button("Check", variant="primary")
        penalty_btn.click(check_penalty, [penalty_input, scale_input, gmax_input], penalty_output)

    with gr.Tab("Correlation Check (derived)"):
        gr.Markdown(
            "For a cluster of equicorrelated assets with continuous weights: how wide is the region of "
            "near-optimal weight splits compared with uncorrelated assets? This is a derived geometric "
            "statement, shown qualitatively."
        )
        rho_input = gr.Slider(0.0, 0.99, value=0.5, step=0.01, label="Pairwise correlation \u03c1")
        correlation_output = gr.HTML()
        plot_output = gr.Plot()
        corr_btn = gr.Button("Check", variant="primary")
        corr_btn.click(check_correlation, rho_input, correlation_output)
        corr_btn.click(correlation_plot, None, plot_output)

    gr.HTML("""
    <hr style="margin-top:30px; border-color:#d8d4c8;">
    <p style="font-size:13px; color:#6b6a5f; line-height:1.6; margin-top:16px;">
    Status: the penalty rule of thumb comes from one small synthetic test family and one local-search
    solver; the correlation statement is a derivation. Neither is a general validated law. The
    post-solve audit (P(q) degeneracy, Entry&rarr;x* gap) works on your own solver output and is
    described in the concept note below. A topological-symmetry mechanism that we tested was not
    supported by the evidence and is not used here. Note: the LOW / INTERMEDIATE / HIGH labels in
    the concept note were derived against an unconstrained random null; for constrained (K-of-N)
    problems they are being recalibrated.
    <br><br>
    Concept note: <a href="https://doi.org/10.5281/zenodo.22738805" target="_blank">
    doi.org/10.5281/zenodo.22738805</a><br>
    Contact: Dimitar Kretski &middot; ORCID 0000-0001-5108-2243
    </p>
    """)

if __name__ == "__main__":
    demo.launch(**_LAUNCH_KW)
