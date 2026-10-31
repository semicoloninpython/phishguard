"""Look & feel helpers for the Streamlit app (CSS, cards, Plotly style)."""
import streamlit as st

PALETTE = {"phish": "#f87171", "legit": "#34d399", "warn": "#fbbf24", "accent": "#38bdf8", "violet": "#a78bfa",
           "panel": "#111c2d", "line": "#26364c", "muted": "#94a3b8"}
SEQ = ["#38bdf8", "#a78bfa", "#34d399", "#fbbf24", "#f472b6", "#fb923c", "#22d3ee", "#c084fc", "#4ade80", "#f87171", "#94a3b8"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--ink:#e8eef8;--muted:#94a3b8;--line:rgba(148,163,184,.18);--panel:#111c2d;--accent:#38bdf8}
.stApp{background:radial-gradient(ellipse at 85% -5%,#16314f 0%,#0a1220 45%,#070d17 100%);font-family:'DM Sans',sans-serif}
[data-testid="stHeader"]{background:rgba(8,14,24,.85)}
[data-testid="stSidebar"]{background:#0b1422;border-right:1px solid var(--line)}
.block-container{max-width:1380px;padding-top:1.6rem;padding-bottom:3rem}
h1,h2,h3,h4{font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.02em}
footer{visibility:hidden}
.pg-brand{font-family:'Space Grotesk';font-weight:700;font-size:24px;margin:0}.pg-brand span{color:var(--accent)}
.pg-kicker{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:#7dd3fc;font-weight:700}
.pg-hero{padding:30px 34px;border:1px solid rgba(125,211,252,.22);border-radius:24px;margin:6px 0 22px;
 background:linear-gradient(115deg,rgba(22,49,79,.95),rgba(13,27,46,.95) 60%,rgba(18,58,84,.9));box-shadow:0 18px 55px rgba(0,0,0,.25)}
.pg-hero h1{font-size:clamp(30px,4vw,46px);margin:8px 0;line-height:1.08}
.pg-hero p{font-size:15px;color:#b7c8dc;margin:0;max-width:820px;line-height:1.6}
.pg-card{background:linear-gradient(145deg,rgba(20,33,52,.97),rgba(14,25,41,.97));border:1px solid var(--line);border-radius:18px;padding:18px 20px;min-height:116px;height:100%}
.pg-card .l{font-size:11px;text-transform:uppercase;letter-spacing:.13em;color:#8ea3bd;font-weight:700;margin-bottom:8px}
.pg-card .v{font-family:'Space Grotesk';font-size:28px;font-weight:700;color:#f4f8ff;line-height:1.15}
.pg-card .s{font-size:12.5px;color:#91a3ba;margin-top:6px}
.pg-card.good{border-color:rgba(52,211,153,.4)}.pg-card.bad{border-color:rgba(248,113,113,.4)}.pg-card.warn{border-color:rgba(251,191,36,.4)}
.pg-section{font-family:'Space Grotesk';font-size:21px;font-weight:700;margin:26px 0 4px}
.pg-sub{color:#9aabc0;font-size:13.5px;margin-bottom:12px}
.pg-flow{display:flex;gap:8px;flex-wrap:wrap;align-items:stretch;margin:8px 0 6px}
.pg-step{flex:1 1 130px;background:#111e30;border:1px solid var(--line);border-radius:14px;padding:12px 14px;position:relative}
.pg-step b{display:block;color:#7dd3fc;font-size:11px;letter-spacing:.12em;margin-bottom:4px}
.pg-step span{font-size:13px;color:#c5d2e3}
.pg-insight{border-left:3px solid var(--accent);background:rgba(17,30,48,.8);border-radius:0 14px 14px 0;padding:12px 16px;margin:8px 0;color:#cdd9e8;font-size:14px}
.pg-verdict{border-radius:18px;padding:22px 24px;border:1px solid var(--line);background:#101c2e;margin:14px 0}
.pg-verdict.safe{border-color:rgba(52,211,153,.4);background:linear-gradient(120deg,rgba(6,78,59,.25),rgba(16,28,46,.95))}
.pg-verdict.warning{border-color:rgba(251,191,36,.4);background:linear-gradient(120deg,rgba(120,78,10,.25),rgba(16,28,46,.95))}
.pg-verdict.danger{border-color:rgba(248,113,113,.45);background:linear-gradient(120deg,rgba(127,29,29,.28),rgba(16,28,46,.95))}
.pg-verdict h2{margin:4px 0 6px;font-size:27px}
.pg-find{padding:12px 15px;border:1px solid var(--line);border-radius:13px;background:rgba(17,29,47,.8);margin:8px 0}
.pg-find b{color:#edf4ff}.pg-find p{font-size:13px;color:#aab9cc;margin:3px 0 0}
.pg-pill{display:inline-block;padding:2px 10px;border-radius:99px;font-size:11px;font-weight:700;letter-spacing:.05em;margin-right:6px}
.pg-pill.ok{background:rgba(52,211,153,.15);color:#6ee7b7}.pg-pill.no{background:rgba(248,113,113,.15);color:#fca5a5}
[data-testid="stMetric"]{background:#111e30;border:1px solid var(--line);border-radius:15px;padding:12px 16px}
[data-testid="stPlotlyChart"]{background:#111c2d;border:1px solid var(--line);border-radius:16px;padding:6px}
div.stButton>button,div.stFormSubmitButton>button{border-radius:12px;font-weight:700;min-height:44px}
.stTabs [data-baseweb="tab-list"]{gap:6px}
.stTabs [data-baseweb="tab"]{background:#111e30;border-radius:10px 10px 0 0;padding:8px 16px}
.stTabs [aria-selected="true"]{background:#1d3b61!important}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(kicker, title, text):
    st.markdown(f'<div class="pg-hero"><div class="pg-kicker">{kicker}</div><h1>{title}</h1><p>{text}</p></div>',
                unsafe_allow_html=True)


def header(kicker, title, text=""):
    st.markdown(f'<div class="pg-kicker">{kicker}</div><h1 style="margin:2px 0 4px">{title}</h1>'
                f'<div class="pg-sub">{text}</div>', unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<div class="pg-section">{title}</div><div class="pg-sub">{sub}</div>', unsafe_allow_html=True)


def card(label, value, sub="", tone=""):
    st.markdown(f'<div class="pg-card {tone}"><div class="l">{label}</div><div class="v">{value}</div>'
                f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="pg-insight">{text}</div>', unsafe_allow_html=True)


def flow(steps):
    html = "".join(f'<div class="pg-step"><b>{i + 1:02d} · {t}</b><span>{d}</span></div>' for i, (t, d) in enumerate(steps))
    st.markdown(f'<div class="pg-flow">{html}</div>', unsafe_allow_html=True)


def style_fig(fig, height=420):
    fig.update_layout(template="plotly_dark", paper_bgcolor=PALETTE["panel"], plot_bgcolor=PALETTE["panel"],
                      font=dict(color="#c9d5e4", family="DM Sans", size=12), height=height,
                      margin=dict(l=20, r=20, t=50, b=40), legend_title_text="", colorway=SEQ)
    fig.update_xaxes(gridcolor="#1f2e44", linecolor="#34445b")
    fig.update_yaxes(gridcolor="#1f2e44", linecolor="#34445b")
    return fig
