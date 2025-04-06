import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

st.set_page_config(layout="wide", page_title="Online Shoppers Intention")

# Custom CSS to improve UI with dark mode compatibility
st.markdown("""
<style>
    /* Card styling for metrics with dark mode compatibility */
    .metric-card {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 1rem;
        color: #CCCCCC;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #FFFFFF;
    }
    h1, h2, h3 {
        color: #FFFFFF !important;
    }
    .stPlotlyChart {
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/online_shoppers_intention.csv")
    return df

df = load_data()

# Title with emoji and formatting
st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>🛒 Online Shoppers Intention Dashboard</h1>", unsafe_allow_html=True)

# Create sidebar with nicer formatting
with st.sidebar:
    st.markdown("### 🔍 Dashboard Filters")
    st.markdown("---")
    
    month_filter = st.multiselect(
        "Select Month(s)", 
        options=sorted(df['Month'].unique()), 
        default=df['Month'].unique()
    )
    
    visitor_filter = st.multiselect(
        "Visitor Type", 
        options=df['VisitorType'].unique(), 
        default=df['VisitorType'].unique()
    )
    
    weekend_filter = st.selectbox(
        "Weekend Session", 
        options=["All", True, False]
    )

# Apply filters
filtered_df = df[
    df['Month'].isin(month_filter) &
    df['VisitorType'].isin(visitor_filter)
]
if weekend_filter != "All":
    filtered_df = filtered_df[filtered_df['Weekend'] == weekend_filter]

# --- Dashboard Overview Section ---
st.markdown("<h2>📊 Dashboard Overview</h2>", unsafe_allow_html=True)

# Creating custom metrics that will be visible in dark mode
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Sessions</div>
        <div class="metric-value">{len(filtered_df):,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Purchase Rate</div>
        <div class="metric-value">{(filtered_df['Revenue'].mean() * 100):.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Non-Purchase Rate</div>
        <div class="metric-value">{(100 - filtered_df['Revenue'].mean() * 100):.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    weekend_pct = (filtered_df['Weekend'] == True).mean() * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Weekend Sessions</div>
        <div class="metric-value">{weekend_pct:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Function to create dark-mode friendly charts
def create_dark_mode_chart(data, title, xlabel=None, ylabel="Sessions", figsize=(5, 3)):
    # Create figure with dark background
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0E1117')
    ax.set_facecolor('#1E1E1E')
    
    # Plot the data
    data.plot(kind='bar', ax=ax, color=['#FF9671', '#00D2FC'], width=0.7)
    
    # Set labels with light colors
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color='white')
    ax.set_ylabel(ylabel, fontsize=10, color='white')
    ax.set_title(title, fontsize=12, pad=10, color='white')
    
    # Format ticks with light colors
    ax.tick_params(colors='white')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    # Format legend with light colors
    ax.legend(["No Purchase", "Purchase"], fontsize=9, loc='upper right', facecolor='#1E1E1E', labelcolor='white')
    
    # Format spines with light colors
    for spine in ax.spines.values():
        spine.set_color('#555555')
    
    # Add grid with subtle lines
    ax.grid(True, linestyle='--', alpha=0.3, color='#555555')
    
    # Make sure y-axis uses integer ticks
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Tight layout
    plt.tight_layout()
    
    return fig

# Create a two-column layout
left_col, right_col = st.columns(2)

with left_col:
    # Monthly Analysis Section
    st.markdown("<h3>📅 Monthly Analysis</h3>", unsafe_allow_html=True)
    
    # Prepare month data
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    filtered_df['Month'] = pd.Categorical(filtered_df['Month'], categories=month_order, ordered=True)
    month_grouped = filtered_df.groupby(['Month', 'Revenue']).size().unstack(fill_value=0).reindex(month_order)
    
    # Create the chart
    fig1 = create_dark_mode_chart(month_grouped, "Monthly Traffic & Conversions")
    st.pyplot(fig1)

with right_col:
    # Visitor Type Analysis
    st.markdown("<h3>👤 Visitor Type Analysis</h3>", unsafe_allow_html=True)
    
    # Prepare visitor data
    visitor_grouped = filtered_df.groupby(['VisitorType', 'Revenue']).size().unstack(fill_value=0)
    
    # Create the chart
    fig2 = create_dark_mode_chart(visitor_grouped, "Visitor Types & Conversions")
    st.pyplot(fig2)

# Second row with two columns
left_col2, right_col2 = st.columns(2)

with left_col2:
    # Weekend vs Weekday Analysis
    st.markdown("<h3>📆 Weekend vs Weekday Analysis</h3>", unsafe_allow_html=True)
    
    # Prepare weekend data
    weekend_grouped = filtered_df.groupby(['Weekend', 'Revenue']).size().unstack(fill_value=0)
    weekend_grouped.index = ['Weekday', 'Weekend'] if len(weekend_grouped) > 1 else ['Weekday'] if False not in weekend_grouped.index else ['Weekend']
    
    # Create the chart
    fig3 = create_dark_mode_chart(weekend_grouped, "Weekend vs Weekday Performance")
    st.pyplot(fig3)

with right_col2:
    # Special Day Analysis
    st.markdown("<h3>🎉 Special Day Analysis</h3>", unsafe_allow_html=True)
    
    # Prepare special day data
    specialday_grouped = filtered_df.groupby(['SpecialDay', 'Revenue']).size().unstack(fill_value=0).sort_index()
    
    # Create the chart
    fig4 = create_dark_mode_chart(specialday_grouped, "Special Day Impact on Purchases", xlabel="Special Day Value")
    st.pyplot(fig4)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #AAAAAA; font-size: 0.8em;'>Dashboard created with Streamlit • Data Analytics Team</p>", unsafe_allow_html=True)