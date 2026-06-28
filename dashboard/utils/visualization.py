"""
Visualization Utility
Centralized Plotly chart configurations with clean, light theme
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Optional, Dict

# Color Palette (Light, Professional)
COLORS = {
    'primary': '#1E40AF',      # Deep Corporate Blue
    'secondary': '#0D9488',    # Teal
    'success': '#16A34A',      # Forest Green
    'danger': '#DC2626',       # Crimson Red
    'warning': '#F59E0B',      # Amber
    'info': '#3B82F6',         # Bright Blue
    'gray': '#64748B',         # Slate Gray
    'light_gray': '#94A3B8',   # Light Slate
    'text_primary': '#0F172A', # Dark Slate
}

# Chart defaults
CHART_HEIGHT = 400
CHART_TEMPLATE = 'plotly_white'


def create_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = CHART_HEIGHT,
    show_legend: bool = True
) -> go.Figure:
    """
    Create a clean line chart.
    
    Args:
        df: DataFrame
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        color_col: Column name for grouping (optional)
        title: Chart title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        height: Chart height in pixels
        show_legend: Whether to show legend
        
    Returns:
        go.Figure: Plotly figure
    """
    if color_col:
        fig = px.line(
            df, 
            x=x_col, 
            y=y_col, 
            color=color_col,
            template=CHART_TEMPLATE,
            color_discrete_sequence=[COLORS['primary'], COLORS['secondary'], 
                                    COLORS['success'], COLORS['warning'], 
                                    COLORS['danger']]
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='lines',
            line=dict(color=COLORS['primary'], width=2),
            name=y_col
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#0F172A')),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        hovermode='x unified',
        showlegend=show_legend,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', color='#475569')
    )
    
    fig.update_xaxes(showgrid=False, showline=True, linecolor='#E2E8F0')
    fig.update_yaxes(showgrid=True, gridcolor='#F1F5F9', showline=True, linecolor='#E2E8F0')
    
    return fig


def create_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    orientation: str = 'v',
    color: str = COLORS['primary'],
    height: int = CHART_HEIGHT
) -> go.Figure:
    """
    Create a clean bar chart.
    
    Args:
        df: DataFrame
        x_col: Column name for x-axis (or y-axis if horizontal)
        y_col: Column name for y-axis (or x-axis if horizontal)
        title: Chart title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        orientation: 'v' for vertical, 'h' for horizontal
        color: Bar color
        height: Chart height in pixels
        
    Returns:
        go.Figure: Plotly figure
    """
    if orientation == 'h':
        fig = go.Figure(go.Bar(
            x=df[y_col],
            y=df[x_col],
            orientation='h',
            marker=dict(color=color),
            text=df[y_col],
            texttemplate='%{text:.2s}',
            textposition='outside'
        ))
        fig.update_xaxes(title=yaxis_title)
        fig.update_yaxes(title=xaxis_title)
    else:
        fig = go.Figure(go.Bar(
            x=df[x_col],
            y=df[y_col],
            marker=dict(color=color),
            text=df[y_col],
            texttemplate='%{text:.2s}',
            textposition='outside'
        ))
        fig.update_xaxes(title=xaxis_title)
        fig.update_yaxes(title=yaxis_title)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#0F172A')),
        height=height,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', color='#475569'),
        showlegend=False
    )
    
    fig.update_xaxes(showgrid=False, showline=True, linecolor='#E2E8F0')
    fig.update_yaxes(showgrid=True, gridcolor='#F1F5F9', showline=True, linecolor='#E2E8F0')
    
    return fig


def create_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    show_diagonal: bool = True,
    height: int = CHART_HEIGHT
) -> go.Figure:
    """
    Create a scatter plot (for actual vs predicted).
    
    Args:
        df: DataFrame
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        title: Chart title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        show_diagonal: Whether to show perfect prediction line
        height: Chart height in pixels
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = go.Figure()
    
    # Scatter points
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='markers',
        marker=dict(
            color=COLORS['primary'],
            size=6,
            opacity=0.6,
            line=dict(width=0)
        ),
        name='Data Points'
    ))
    
    # Diagonal line (perfect prediction)
    if show_diagonal:
        min_val = min(df[x_col].min(), df[y_col].min())
        max_val = max(df[x_col].max(), df[y_col].max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color=COLORS['danger'], width=2, dash='dash'),
            name='Perfect Prediction'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#0F172A')),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', color='#475569'),
        showlegend=True
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='#F1F5F9', showline=True, linecolor='#E2E8F0')
    fig.update_yaxes(showgrid=True, gridcolor='#F1F5F9', showline=True, linecolor='#E2E8F0')
    
    return fig


def create_box_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = CHART_HEIGHT
) -> go.Figure:
    """
    Create a box plot for distribution analysis.
    
    Args:
        df: DataFrame
        x_col: Column name for x-axis (categories)
        y_col: Column name for y-axis (values)
        title: Chart title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        height: Chart height in pixels
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = px.box(
        df,
        x=x_col,
        y=y_col,
        template=CHART_TEMPLATE,
        color_discrete_sequence=[COLORS['primary']]
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#0F172A')),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', color='#475569'),
        showlegend=False
    )
    
    fig.update_xaxes(showgrid=False, showline=True, linecolor='#E2E8F0')
    fig.update_yaxes(showgrid=True, gridcolor='#F1F5F9', showline=True, linecolor='#E2E8F0')
    
    return fig


def create_pie_chart(
    values: List[float],
    labels: List[str],
    title: str = "",
    height: int = CHART_HEIGHT,
    colors: Optional[List[str]] = None,
) -> go.Figure:
    _default_colors = [
        COLORS['primary'], COLORS['secondary'], COLORS['success'],
        COLORS['warning'], COLORS['danger'], COLORS['info'],
    ]
    slice_colors = colors if colors is not None else _default_colors
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=slice_colors[:len(labels)]),
        textinfo='label+percent',
        textposition='auto',
        hole=0.3,
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#0F172A')),
        height=height,
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', color='#475569'),
        showlegend=True,
    )
    
    return fig

def create_heatmap(
    df: pd.DataFrame,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = CHART_HEIGHT,
    colorscale: str = 'Blues'
) -> go.Figure:
    """
    Create a correlation heatmap.
    
    Args:
        df: Correlation matrix (DataFrame)
        title: Chart title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        height: Chart height in pixels
        colorscale: Plotly colorscale name
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        colorscale=colorscale,
        text=df.values,
        texttemplate='%{text:.2f}',
        textfont=dict(size=10),
        colorbar=dict(title='Correlation')
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#0F172A')),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', color='#475569')
    )
    
    return fig


def create_histogram(
    df: pd.DataFrame,
    x_col: str,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "Frequency",
    nbins: int = 30,
    height: int = CHART_HEIGHT
) -> go.Figure:
    """
    Create a histogram with KDE overlay.
    
    Args:
        df: DataFrame
        x_col: Column name for values
        title: Chart title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        nbins: Number of bins
        height: Chart height in pixels
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df[x_col],
        nbinsx=nbins,
        marker=dict(color=COLORS['primary'], opacity=0.7),
        name='Histogram'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#0F172A')),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', color='#475569'),
        showlegend=False,
        bargap=0.1
    )
    
    fig.update_xaxes(showgrid=False, showline=True, linecolor='#E2E8F0')
    fig.update_yaxes(showgrid=True, gridcolor='#F1F5F9', showline=True, linecolor='#E2E8F0')
    
    return fig


def create_multi_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    labels: List[str],
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = CHART_HEIGHT
) -> go.Figure:
    """
    Create multi-line chart for comparison.
    
    Args:
        df: DataFrame
        x_col: Column name for x-axis
        y_cols: List of column names for y-axes
        labels: List of labels for each line
        title: Chart title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        height: Chart height in pixels
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = go.Figure()
    
    colors_list = [COLORS['primary'], COLORS['secondary'], COLORS['success'], 
                   COLORS['warning'], COLORS['danger']]
    
    for idx, (col, label) in enumerate(zip(y_cols, labels)):
        line_style = dict(color=colors_list[idx % len(colors_list)], width=2)
        if idx > 0:  # Make prediction lines dashed
            line_style['dash'] = 'dash'
        
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[col],
            mode='lines',
            name=label,
            line=line_style
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#0F172A')),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', color='#475569'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(showgrid=False, showline=True, linecolor='#E2E8F0')
    fig.update_yaxes(showgrid=True, gridcolor='#F1F5F9', showline=True, linecolor='#E2E8F0')
    
    return fig


def create_horizontal_bar_chart(
    labels: List[str],
    values: List[float],
    title: str = "",
    xaxis_title: str = "",
    height: int = CHART_HEIGHT,
    color_scale: Optional[List[str]] = None
) -> go.Figure:
    """
    Create horizontal bar chart (useful for feature importance, rankings).
    
    Args:
        labels: List of labels (y-axis)
        values: List of values (x-axis)
        title: Chart title
        xaxis_title: X-axis label
        height: Chart height in pixels
        color_scale: Optional list of colors for each bar
        
    Returns:
        go.Figure: Plotly figure
    """
    if color_scale is None:
        colors = [COLORS['primary']] * len(labels)
    else:
        colors = color_scale
    
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{v:.2f}%" if v < 100 else f"{v:,.0f}" for v in values],
        textposition='outside',
        textfont=dict(size=11, color=COLORS['text_primary'])
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#0F172A')),
        xaxis_title=xaxis_title,
        yaxis=dict(autorange="reversed"),
        height=height,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', color='#475569'),
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
        yaxis_showgrid=False
    )
    
    return fig


def create_grouped_bar_chart(
    categories: List[str],
    values_dict: Dict[str, List[float]],
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = CHART_HEIGHT
) -> go.Figure:
    """
    Create grouped bar chart (e.g., model comparison).
    
    Args:
        categories: List of category labels (x-axis)
        values_dict: Dictionary of {label: values_list}
        title: Chart title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        height: Chart height in pixels
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = go.Figure()
    
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['success']]
    
    for i, (label, values) in enumerate(values_dict.items()):
        fig.add_trace(go.Bar(
            name=label,
            x=categories,
            y=values,
            marker_color=colors[i % len(colors)],
            text=[f"{v:,.2f}" for v in values],
            textposition='outside'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#0F172A')),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', color='#475569'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#F1F5F9')
    )
    
    return fig


def format_currency(value: float) -> str:
    """
    Format number as Indonesian Rupiah.
    
    Args:
        value: Numeric value
        
    Returns:
        str: Formatted string (e.g., "Rp 15.000")
    """
    return f"Rp {value:,.0f}".replace(',', '.')


def format_percentage(value: float) -> str:
    """
    Format number as percentage.
    
    Args:
        value: Numeric value
        
    Returns:
        str: Formatted string (e.g., "9.34%")
    """
    return f"{value:.2f}%"
