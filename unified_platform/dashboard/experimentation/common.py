"""Shared UI building blocks for the Experimentation pages.

Colors are pulled from the unified orange/dark theme (dashboard/theme.py) —
the "info" tone that was previously blue (#60a5fa) now uses the theme's
accent (amber/orange) color instead, so no blue remains anywhere in the
merged app.
"""
from html import escape

import pandas as pd
import streamlit as st

from dashboard.experimentation.data_loader import (
    badge_for_decision,
    symbol_for_status,
    tone_for_decision,
    tone_for_status,
)
from dashboard.theme import get_theme_colors

EMPTY_STATE = "No experiment selected. Choose an experiment in the sidebar."


def _tone_colors():
    c = get_theme_colors()
    return {
        "success": (c['success'], "rgba(0, 204, 85, 0.1)", "rgba(0, 204, 85, 0.2)"),
        "danger": (c['danger'], "rgba(255, 34, 34, 0.1)", "rgba(255, 34, 34, 0.2)"),
        "warning": (c['warning'], "rgba(255, 187, 0, 0.1)", "rgba(255, 187, 0, 0.2)"),
        "info": (c['accent'], "rgba(255, 165, 0, 0.1)", "rgba(255, 165, 0, 0.2)"),
        "neutral": ("#cbd5e1", "rgba(148, 163, 184, 0.1)", "rgba(148, 163, 184, 0.2)"),
    }


def _safe(value, fallback="N/A"):
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    return value


def _warn_if_error(name, err):
    if err:
        st.warning(f"{name}: {err}")


def _page_title(title, subtitle=None):
    st.markdown(
        f'<div style="margin: 10px 0 32px;"><div style="font-size: 28px; font-weight: 600; color: #f8fafc; letter-spacing: -0.5px;">{escape(str(title))}</div>'
        f'<div style="color: #94a3b8; font-size: 16px; margin-top: 8px;">{escape(str(subtitle)) if subtitle else ""}</div></div>',
        unsafe_allow_html=True
    )


def _decision_pill(decision):
    marker = badge_for_decision(decision)
    tone = tone_for_decision(decision)
    label = escape(str(_safe(decision)))

    text_color, bg_color, border_color = _tone_colors().get(tone, _tone_colors()["neutral"])

    return (
        f'<span style="display: inline-flex; align-items: center; gap: 8px; '
        f'padding: 6px 14px; border-radius: 999px; font-size: 13px; font-weight: 600; '
        f'text-transform: uppercase; color: {text_color}; background: {bg_color}; border: 1px solid {border_color};">'
        f'<span>{escape(marker)}</span><span>{label}</span></span>'
    )


def _status_pill(value, label=None):
    marker = symbol_for_status(value)
    tone = tone_for_status(value)
    text = label if label is not None else value

    text_color, bg_color, _ = _tone_colors().get(tone, _tone_colors()["neutral"])

    return (
        f'<span style="display: inline-flex; align-items: center; gap: 6px; '
        f'padding: 4px 12px; border-radius: 999px; font-size: 13px; font-weight: 500; '
        f'color: {text_color}; background: {bg_color};">'
        f'<span>{escape(marker)}</span><span>{escape(str(text))}</span></span>'
    )


def _rationale_textbox(title: str, text: str):
    """Shared component to render a standardized rationale textbox."""
    st.text_area(title, value=text, height=100, disabled=True)


def _colored_kv_grid(items):
    html = ['<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 24px; margin-bottom: 32px;">']
    colors = _tone_colors()
    neutral_bg = get_theme_colors()['secondary']
    for label, value, tone in items:
        text_color, bg_color, border_color = colors.get(tone, colors["neutral"])
        if tone == "neutral":
            text_color, bg_color, border_color = "#f8fafc", neutral_bg, "rgba(255, 255, 255, 0.05)"
        html.append(
            f'<div style="background: {bg_color}; border: 1px solid {border_color}; '
            f'border-radius: 12px; padding: 24px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">'
            f'<div style="color: #94a3b8; font-size: 13px; font-weight: 600; text-transform: uppercase; margin-bottom: 8px;">{escape(str(label))}</div>'
            f'<div style="font-size: 22px; font-weight: 600; color: {text_color};">{escape(str(_safe(value)))}</div>'
            '</div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _kv_grid(items):
    _colored_kv_grid([(label, value, "neutral") for label, value in items])


def _panel(body):
    st.markdown(f'<div class="exp-panel">{body}</div>', unsafe_allow_html=True)


def _format_float(value, digits=4, fallback="N/A"):
    if value is None or pd.isna(value):
        return fallback
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_percent(value, digits=1):
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.{digits}%}"
    except (TypeError, ValueError):
        return str(value)


def _setup_chart(ax, fig):
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    ax.grid(axis="y", color="#ffffff", alpha=0.1, linewidth=0.8, linestyle="--")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#ffffff")
    ax.spines["bottom"].set_alpha(0.2)
    ax.tick_params(colors="#94a3b8", length=0)
    ax.xaxis.label.set_color("#cbd5e1")
    ax.yaxis.label.set_color("#cbd5e1")
