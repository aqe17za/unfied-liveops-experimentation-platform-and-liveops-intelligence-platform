"""Shared theme — builds the unified dashboard's CSS entirely from config.yaml.

Both LiveOps and Experimentation pages share this single orange/dark theme
(sourced from Project 1's original aesthetic). No hex values are hardcoded
here beyond structural CSS (borders, shadows) that reference the config
colors via string interpolation.
"""
from pathlib import Path

import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"


@st.cache_data
def load_config() -> dict:
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


def get_theme_colors() -> dict:
    """Single source of theme colors — Project 1's orange/dark palette."""
    config = load_config()
    liveops_colors = config['liveops']['dashboard']
    return {
        'primary': liveops_colors['primary_color'],
        'secondary': liveops_colors['secondary_color'],
        'background': liveops_colors['background_color'],
        'text': liveops_colors['text_color'],
        'accent': liveops_colors['accent_color'],
        'success': liveops_colors['success_color'],
        'danger': liveops_colors['danger_color'],
        'warning': liveops_colors['warning_color'],
        'tier_colors': liveops_colors['tier_colors'],
        'priority_colors': liveops_colors['priority_colors'],
    }


def inject_global_css():
    """Apply the unified orange/dark theme CSS. Call once at app startup."""
    c = get_theme_colors()

    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="st-"], .stApp {{
            font-family: 'Inter', sans-serif !important;
        }}

        .stApp {{
            background-color: {c['background']};
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #121212 0%, {c['secondary']} 100%);
            border-right: 1px solid {c['primary']};
        }}

        .main .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }}

        h1 {{ color: {c['primary']} !important; font-size: 1.75rem !important; font-weight: 700 !important; letter-spacing: -0.5px; }}
        h2 {{ color: {c['primary']} !important; font-size: 1.3rem !important; font-weight: 600 !important; }}
        h3 {{ color: {c['text']} !important; font-size: 1.05rem !important; font-weight: 600 !important; }}

        [data-testid="metric-container"], [data-testid="stMetric"] {{
            background: linear-gradient(135deg, {c['secondary']} 0%, #252525 100%);
            border: 1px solid {c['primary']};
            border-radius: 10px;
            padding: 16px !important;
            box-shadow: 0 4px 15px rgba(255, 75, 0, 0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        [data-testid="metric-container"]:hover, [data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 75, 0, 0.2);
        }}
        [data-testid="stMetricLabel"] {{
            color: #AAAAAA !important;
            font-size: 0.78rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}
        [data-testid="stMetricValue"] {{
            color: {c['text']} !important;
            font-size: 1.7rem !important;
            font-weight: 700 !important;
            line-height: 1.2;
        }}
        [data-testid="stMetricDelta"] {{ font-size: 0.8rem !important; }}

        .stButton > button {{
            background: linear-gradient(135deg, {c['primary']}, #CC3B00);
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.875rem;
            padding: 0.5rem 1.25rem;
            transition: all 0.2s ease;
            box-shadow: 0 2px 8px rgba(255, 75, 0, 0.3);
            letter-spacing: 0.3px;
        }}
        .stButton > button:hover {{
            background: linear-gradient(135deg, #FF6020, #E04000);
            box-shadow: 0 4px 15px rgba(255, 75, 0, 0.5);
            transform: translateY(-1px);
        }}
        .stButton > button:active {{ transform: translateY(0); }}

        .stTextInput > div > div > input {{
            background-color: {c['secondary']};
            color: {c['text']};
            border: 1px solid #333;
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
            font-size: 0.9rem;
            transition: border-color 0.2s ease;
        }}
        .stTextInput > div > div > input:focus {{
            border-color: {c['primary']};
            box-shadow: 0 0 0 2px rgba(255,75,0,0.15);
        }}

        .stSelectbox > div > div {{
            background-color: {c['secondary']};
            color: {c['text']};
            border-color: #333;
            border-radius: 6px;
        }}

        [data-testid="stRadio"] > div {{ gap: 4px; }}
        [data-testid="stRadio"] label {{
            background: transparent;
            border-radius: 6px;
            padding: 6px 10px;
            transition: background 0.15s ease;
            cursor: pointer;
            font-size: 0.875rem !important;
        }}
        [data-testid="stRadio"] label:hover {{ background: rgba(255, 75, 0, 0.1); }}

        hr {{ border-color: #2a2a2a; margin: 1rem 0; }}

        [data-testid="stExpander"] {{
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            background: #161616;
            margin-bottom: 8px;
        }}
        [data-testid="stExpander"] details summary {{
            display: flex !important;
            align-items: center !important;
            gap: 8px !important;
            padding: 10px 14px !important;
            cursor: pointer;
            list-style: none;
        }}
        [data-testid="stExpander"] details summary::-webkit-details-marker {{ display: none; }}
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {{
            color: {c['primary']} !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            margin: 0 !important;
        }}
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] {{
            color: {c['primary']} !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }}
        [data-testid="stExpander"] [data-testid="stIconMaterial"],
        [data-testid="stExpander"] [data-testid="stIconEmoji"] {{
            color: {c['primary']} !important;
            fill: {c['primary']} !important;
            flex-shrink: 0;
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid #2a2a2a;
            border-radius: 8px;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            background-color: #1a1a1a;
            border-radius: 8px 8px 0 0;
            gap: 2px;
            padding: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            color: #888888;
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s;
        }}
        .stTabs [aria-selected="true"] {{
            color: {c['primary']} !important;
            background: rgba(255,75,0,0.1) !important;
        }}

        .stCode {{ border-radius: 8px; border: 1px solid #2a2a2a; }}

        [data-testid="stCaptionContainer"] {{ color: #888888 !important; font-size: 0.8rem !important; }}

        .stAlert {{ border-radius: 8px; }}

        .alert-box {{
            background: linear-gradient(135deg, #1e1a17 0%, #231e1a 100%);
            border: 1px solid {c['primary']};
            border-left: 4px solid {c['primary']};
            border-radius: 8px;
            padding: 14px 18px;
            margin: 10px 0;
            font-size: 0.9rem;
        }}
        .alert-critical {{
            background: linear-gradient(135deg, #1e0a0a 0%, #2a0d0d 100%);
            border: 1px solid {c['danger']};
            border-left: 4px solid {c['danger']};
            border-radius: 8px;
            padding: 14px 18px;
            margin: 10px 0;
            font-size: 0.9rem;
        }}
        .alert-success {{
            background: linear-gradient(135deg, #0a1e0a 0%, #0d2a0d 100%);
            border: 1px solid {c['success']};
            border-left: 4px solid {c['success']};
            border-radius: 8px;
            padding: 14px 18px;
            margin: 10px 0;
            font-size: 0.9rem;
        }}

        .stat-card {{
            background: linear-gradient(135deg, #1a1a1a 0%, #222222 100%);
            border-radius: 10px;
            padding: 18px;
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .stat-card:hover {{ transform: translateY(-3px); }}

        .sidebar-brand {{
            text-align: center;
            padding: 20px 0 10px;
            border-bottom: 1px solid #2a2a2a;
            margin-bottom: 16px;
        }}

        .player-header {{
            border-radius: 10px;
            padding: 20px 24px;
            margin-bottom: 18px;
            border: 2px solid;
            background: linear-gradient(135deg, #1a1a1a 0%, #222 100%);
        }}

        .progress-container {{
            background: #2a2a2a;
            border-radius: 4px;
            height: 8px;
            margin: 4px 0 10px 0;
            overflow: hidden;
        }}

        /* Panel — used by Experimentation pages */
        .exp-panel {{
            background: {c['secondary']};
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 32px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 32px;
        }}

        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: #1a1a1a; }}
        ::-webkit-scrollbar-thumb {{ background: {c['primary']}; border-radius: 3px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: #FF6020; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def make_plotly_dark(fig, title=None, height=400):
    """Apply consistent dark theme to any plotly figure."""
    c = get_theme_colors()
    layout_args = dict(
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#111111',
        font=dict(color='#CCCCCC', family='Inter, sans-serif', size=12),
        height=height,
        margin=dict(l=40, r=20, t=55, b=40),
        legend=dict(
            bgcolor='rgba(26,26,26,0.8)',
            bordercolor='#333333',
            borderwidth=1,
            font=dict(color='#CCCCCC')
        ),
        xaxis=dict(
            gridcolor='#252525',
            linecolor='#333333',
            tickfont=dict(color='#AAAAAA'),
            zerolinecolor='#333333',
        ),
        yaxis=dict(
            gridcolor='#252525',
            linecolor='#333333',
            tickfont=dict(color='#AAAAAA'),
            zerolinecolor='#333333',
        )
    )
    if title:
        layout_args['title'] = dict(
            text=title,
            font=dict(size=13, color='#FF6030'),
            x=0,
            xanchor='left',
            pad=dict(l=0)
        )
    fig.update_layout(**layout_args)
    return fig


def kpi_card(value, label, color=None, subtitle=None):
    """Render a colored KPI stat card as HTML."""
    color = color or get_theme_colors()['primary']
    sub_html = f'<div style="color:#888;font-size:0.8rem;margin-top:4px;">{subtitle}</div>' if subtitle else ''
    return f"""
    <div class="stat-card" style="border: 1px solid {color}; box-shadow: 0 4px 15px {color}22;">
        <div style="color:{color}; font-size:2rem; font-weight:700; line-height:1.1;">{value}</div>
        <div style="color:#AAAAAA; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.5px; margin-top:6px;">{label}</div>
        {sub_html}
    </div>
    """
