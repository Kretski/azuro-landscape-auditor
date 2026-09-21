# -*- coding: utf-8 -*-
"""
AZURO Landscape Auditor — Interactive Pre-Solve Demo
======================================================
Hugging Face Space demonstrating the two VALIDATED closed-form
pre-solve risk indicators (no solver required, results in
milliseconds):

  1. Penalty calibration check — P* >= ~1.25 x max coefficient
     magnitude, validated scale-invariant across a 3x coefficient
     range (0% relative deviation in testing).
  2. Correlated-asset degeneracy check — near-optimal weight spread
     grows as ~1/sqrt(1-rho) as correlation rho -> 1, derived from the
     portfolio risk Hessian curvature (1-rho), confirmed R^2=0.94
     against the theoretical exponent.

This demo runs entirely on SYNTHETIC / user-entered parameters —
no real client data is uploaded or required, addressing the
confidentiality concern that stops real instances from being shared
in a public demo. For a full post-solve audit (P(q), Entry->x* gap)
on your own solver output, use the contact link at the bottom.
"""

import gradio as gr
import numpy as np
import spaces

# ----------------------------------------------------------------------
# Styling — muted paper palette + serif headers, matching the report
# mockup aesthetic (institutional, not consumer SaaS)
# ----------------------------------------------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

.gradio-container {
    font-family: 'IBM Plex Sans', sans-serif !important;
    background: #f7f5f0 !important;
    max-width: 880px !important;
    margin: auto !important;
}
h1, h2, h3 {
    font-family: 'Source Serif 4', serif !important;
}
.verdict-ok { background: #dde8e0; border-left: 4px solid #2f5d42; padding: 16px 20px; color: #23241f !important; }
.verdict-caution { background: #f2e6cd; border-left: 4px solid #9c6b1f; padding: 16px 20px; color: #23241f !important; }
.verdict-flag { background: #f0ddd7; border-left: 4px solid #8c3b2e; padding: 16px 20px; color: #23241f !important; }
.verdict-ok b, .verdict-caution b, .verdict-flag b { color: #23241f !important; }
footer { display: none !important; }
"""

BRAND_HEADER = """
<div style="border-bottom: 2px solid #23241f; padding-bottom: 12px; margin-bottom: 20px;
            display: flex; justify-content: space-between; align-items: baseline;">
  <span style="font-family: 'Source Serif 4', serif; font-weight: 700; font-size: 24px;">AZURO</span>
  <span style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #6b6a5f;">
    Pre-Solve Risk Indicators — Interactive Demo
  </span>
</div>
<p style="font-size: 14px; color: #6b6a5f; line-height: 1.6;">
Two closed-form checks, computable before you ever run a solver. Adjust the sliders
below to test your own scenario — no data is uploaded or stored; everything runs
in your browser session on the numbers you enter.
</p>
"""


# ----------------------------------------------------------------------
# Check 1 — Penalty calibration
# ----------------------------------------------------------------------
def check_penalty(penalty_weight, coefficient_scale):
    if coefficient_scale <= 0:
        return "<p>Coefficient scale must be positive.</p>"
    ratio = penalty_weight / coefficient_scale
    threshold = 1.25

    if ratio >= threshold * 1.3:
        cls, verdict = "verdict-ok", "OK — well above threshold"
        detail = (f"Your penalty-to-coefficient ratio is <b>{ratio:.2f}×</b>, comfortably above "
                   f"the validated minimum of <b>{threshold}×</b>. Feasibility should be reliable.")
    elif ratio >= threshold:
        cls, verdict = "verdict-caution", "OK — near threshold"
        detail = (f"Your ratio is <b>{ratio:.2f}×</b>, just above the validated minimum of "
                   f"<b>{threshold}×</b>. Should be feasible, but with little margin — consider "
                   f"a small increase for robustness.")
    else:
        cls, verdict = "verdict-flag", "FLAG — below threshold"
        shortfall = (threshold - ratio) / threshold * 100
        detail = (f"Your ratio is <b>{ratio:.2f}×</b>, below the validated minimum of "
                   f"<b>{threshold}×</b> ({shortfall:.0f}% short). Expect a meaningful fraction "
                   f"of your solver's near-optimal solutions to violate your constraints. "
                   f"Increase your penalty weight to at least {threshold * coefficient_scale:.2f}.")

    return f"""
    <div class="{cls}">
    <b>{verdict}</b><br><br>{detail}
    </div>
    <p style="font-size:12px; color:#6b6a5f; margin-top:10px;">
    Rule: P* ≈ 1.25 × max coefficient magnitude in your objective — validated scale-invariant
    across a 3× change in coefficient scale (0% relative deviation).
    </p>
    """


# ----------------------------------------------------------------------
# Check 2 — Correlated-asset degeneracy
# ----------------------------------------------------------------------
def check_correlation(rho):
    rho_calc = min(rho, 0.995)
    curvature = 1 - rho_calc
    relative_spread = 1.0 / np.sqrt(curvature)
    baseline = 1.0 / np.sqrt(1.0)  # rho=0 reference

    multiple = relative_spread / baseline

    if rho < 0.5:
        cls, verdict = "verdict-ok", "OK — low redundancy"
        detail = (f"At correlation ρ={rho:.2f}, near-optimal allocation spread is "
                   f"<b>{multiple:.1f}×</b> the uncorrelated baseline. Asset weights within "
                   f"this group should be fairly well-determined.")
    elif rho < 0.8:
        cls, verdict = "verdict-caution", "CAUTION — moderate redundancy"
        detail = (f"At correlation ρ={rho:.2f}, near-optimal allocation spread is "
                   f"<b>{multiple:.1f}×</b> the uncorrelated baseline. How weight is split "
                   f"within this cluster is becoming less determined by the optimizer.")
    else:
        cls, verdict = "verdict-flag", "FLAG — high redundancy"
        detail = (f"At correlation ρ={rho:.2f}, near-optimal allocation spread is "
                   f"<b>{multiple:.1f}×</b> the uncorrelated baseline. Assets in this cluster "
                   f"are close to fully interchangeable — the specific split your solver "
                   f"returns is close to arbitrary.")

    return f"""
    <div class="{cls}">
    <b>{verdict}</b><br><br>{detail}
    </div>
    <p style="font-size:12px; color:#6b6a5f; margin-top:10px;">
    Rule: near-optimal weight spread grows as ~1/√(1−ρ) as correlation ρ→1 — derived from the
    portfolio risk Hessian curvature (1−ρ), confirmed R²=0.94 against the theoretical exponent.
    </p>
    """


@spaces.GPU
def correlation_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rho_vals = np.linspace(0, 0.99, 200)
    spread = 1.0 / np.sqrt(1 - rho_vals)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(rho_vals, spread, color="#8c3b2e", linewidth=2)
    ax.set_xlabel("Correlation ρ")
    ax.set_ylabel("Relative near-optimal spread")
    ax.set_title("Near-optimal weight spread vs. asset correlation", fontsize=11)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
with gr.Blocks(css=CUSTOM_CSS, title="AZURO — Pre-Solve Risk Demo") as demo:
    gr.HTML(BRAND_HEADER)

    with gr.Tab("Penalty Calibration Check"):
        gr.Markdown(
            "For QUBO constraints encoded as penalty terms (one-hot, cardinality, etc.), "
            "check whether your penalty weight is high enough for reliable feasibility."
        )
        with gr.Row():
            penalty_input = gr.Slider(0.1, 10, value=1.0, step=0.05, label="Penalty weight (P)")
            scale_input = gr.Slider(0.1, 10, value=1.0, step=0.05,
                                     label="Max coefficient magnitude in objective")
        penalty_output = gr.HTML()
        penalty_btn = gr.Button("Check", variant="primary")
        penalty_btn.click(check_penalty, [penalty_input, scale_input], penalty_output)

    with gr.Tab("Correlation Degeneracy Check"):
        gr.Markdown(
            "For a cluster of correlated assets in a portfolio-style problem, check how "
            "'locked in' the optimal weight split is likely to be."
        )
        rho_input = gr.Slider(0.0, 0.99, value=0.5, step=0.01, label="Pairwise correlation ρ")
        correlation_output = gr.HTML()
        plot_output = gr.Plot()
        corr_btn = gr.Button("Check", variant="primary")
        corr_btn.click(check_correlation, rho_input, correlation_output)
        corr_btn.click(correlation_plot, None, plot_output)

    gr.HTML("""
    <hr style="margin-top:30px; border-color:#d8d4c8;">
    <p style="font-size:13px; color:#6b6a5f; line-height:1.6; margin-top:16px;">
    These two checks are validated, closed-form, and require only structural parameters —
    no sensitive data. For the full post-solve audit (P(q) degeneracy, Entry→x* false-confidence
    gap) on your own solver output, or to test a third mechanism we investigated and found
    <b>not</b> supported (topological symmetry — reported for completeness, not used here):
    <br><br>
    Methodology &amp; validation: <a href="https://doi.org/10.5281/zenodo.21941962" target="_blank">
    doi.org/10.5281/zenodo.21941962</a><br>
    Contact: Dimitar Kretski · ORCID 0000-0001-5108-2243
    </p>
    """)

if __name__ == "__main__":
    demo.launch()
