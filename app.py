import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="NYC Taxi Trip Analysis", 
    layout="wide",
    page_icon="🚖"
)

# Custom styling
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    h1 {color: #2c3e50; font-weight: 700;}
    h2 {color: #3498db; border-bottom: 2px solid #3498db; padding-bottom: 8px;}
    h3 {color: #2c3e50;}
    .sidebar .sidebar-content {
        background-color: #2c3e50; 
        color: white;
        padding: 20px;
    }
    .st-bb {background-color: white;}
    .st-at {background-color: #3498db;}
    .css-1aumxhk {background-color: #2c3e50; color: white;}
    .metric-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-label {
        font-size: 14px;
        color: #7f8c8d;
        font-weight: 500;
    }
    .metric-value {
        font-size: 24px;
        color: #2c3e50;
        font-weight: 700;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        background-color: #ecf0f1;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }
    .plot-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🚖 NYC Green Taxi Trip Analysis")
st.markdown("""
Explore NYC taxi trip patterns by adjusting the parameters in the sidebar. 
The visualizations will update based on your selections.
""")

# Sample data generation function
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n_samples = 5000  # Increased sample size for better visualizations
    
    data = {
        'trip_duration': np.random.normal(15, 5, n_samples).clip(1, 60),
        'trip_distance': np.random.normal(3, 1.5, n_samples).clip(0.1, 15),
        'total_amount': np.random.normal(20, 8, n_samples).clip(2.5, 100),
        'tip_amount': np.random.normal(3, 1.5, n_samples).clip(0, 10),
        'passenger_count': np.random.randint(1, 6, n_samples),
        'weekday': np.random.choice(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            n_samples,
            p=[0.15, 0.15, 0.15, 0.15, 0.15, 0.125, 0.125]  # Fixed probabilities to sum to 1
        ),
        'hourofday': np.random.randint(0, 24, n_samples),
        'payment_type': np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.6, 0.3, 0.05, 0.02, 0.02, 0.01]),
        'trip_type': np.random.choice([1, 2], n_samples, p=[0.9, 0.1]),
        'PULocationID': np.random.randint(1, 264, n_samples),
        'DOLocationID': np.random.randint(1, 264, n_samples),
        'fare_amount': np.random.normal(15, 5, n_samples).clip(2.5, 50),
        'extra': np.random.choice([0, 0.5, 1, 2.5], n_samples, p=[0.7, 0.2, 0.08, 0.02]),
        'mta_tax': np.random.choice([0, 0.5], n_samples, p=[0.1, 0.9]),
        'improvement_surcharge': np.random.choice([0, 0.3], n_samples, p=[0.05, 0.95]),
        'congestion_surcharge': np.random.choice([0, 2.5], n_samples, p=[0.3, 0.7])
    }
    
    df = pd.DataFrame(data)
    df['payment_type'] = df['payment_type'].map({
        1: 'Credit card', 
        2: 'Cash', 
        3: 'No charge', 
        4: 'Dispute', 
        5: 'Unknown', 
        6: 'Voided trip'
    })
    df['trip_type'] = df['trip_type'].map({1: 'Street-hail', 2: 'Dispatch'})
    
    # Calculate total amount if not already set
    df['total_amount'] = df['fare_amount'] + df['extra'] + df['mta_tax'] + df['tip_amount'] + df['improvement_surcharge'] + df['congestion_surcharge']
    
    return df

df = generate_sample_data()

# Sidebar for user inputs
with st.sidebar:
    st.header("🔍 Analysis Parameters")
    st.markdown("---")
    
    # Time filters
    st.subheader("Time Filters")
    selected_weekdays = st.multiselect(
        "Select weekdays",
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        default=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )
    
    time_range = st.slider(
        "Select hour range",
        0, 23, (6, 20),
        help="Filter trips by pickup hour"
    )
    
    # Trip characteristics
    st.subheader("Trip Characteristics")
    passenger_count = st.slider(
        "Passenger count",
        1, 6, (1, 6),
        help="Filter by number of passengers"
    )
    
    trip_distance_range = st.slider(
        "Trip distance (miles)",
        0.0, 20.0, (0.5, 10.0),
        help="Filter by trip distance in miles"
    )
    
    payment_types = st.multiselect(
        "Payment types",
        df['payment_type'].unique(),
        default=['Credit card', 'Cash'],
        help="Select payment methods to include"
    )

# Filter data based on user inputs
filtered_df = df[
    (df['weekday'].isin(selected_weekdays)) &
    (df['hourofday'] >= time_range[0]) &
    (df['hourofday'] <= time_range[1]) &
    (df['passenger_count'] >= passenger_count[0]) &
    (df['passenger_count'] <= passenger_count[1]) &
    (df['trip_distance'] >= trip_distance_range[0]) &
    (df['trip_distance'] <= trip_distance_range[1]) &
    (df['payment_type'].isin(payment_types))
]

# Main content
st.subheader(f"📊 Analyzing {len(filtered_df):,} trips matching your criteria")

# Key metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-container"><div class="metric-label">Avg. Trip Duration</div>'
                f'<div class="metric-value">{filtered_df["trip_duration"].mean():.1f} min</div></div>', 
                unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-container"><div class="metric-label">Avg. Total Amount</div>'
                f'<div class="metric-value">${filtered_df["total_amount"].mean():.2f}</div></div>', 
                unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-container"><div class="metric-label">Avg. Distance</div>'
                f'<div class="metric-value">{filtered_df["trip_distance"].mean():.1f} miles</div></div>', 
                unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-container"><div class="metric-label">Avg. Tip Amount</div>'
                f'<div class="metric-value">${filtered_df["tip_amount"].mean():.2f}</div></div>', 
                unsafe_allow_html=True)

# Visualization tabs
tab1, tab2, tab3 = st.tabs(["📅 Temporal Patterns", "💰 Financial Analysis", "🚦 Trip Characteristics"])

with tab1:
    st.subheader("Temporal Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Trips by hour
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.write("### Trips by Hour of Day")
        hourly_counts = filtered_df['hourofday'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=hourly_counts.index, y=hourly_counts.values, color='#3498db', ax=ax)
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Number of Trips', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Trips by weekday
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.write("### Trips by Weekday")
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_counts = filtered_df['weekday'].value_counts().reindex(weekday_order)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=weekday_counts.index, y=weekday_counts.values, palette='Blues_d', ax=ax)
        ax.set_xlabel('Weekday', fontsize=12)
        ax.set_ylabel('Number of Trips', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Heatmap of trips by hour and weekday
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.write("### Trip Frequency Heatmap (Hour vs Weekday)")
    heatmap_data = filtered_df.groupby(['weekday', 'hourofday']).size().unstack()
    heatmap_data = heatmap_data.reindex(weekday_order)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax, linewidths=0.5)
    ax.set_title('Number of Trips by Hour and Weekday', fontsize=14)
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Weekday', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("Financial Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Payment type distribution
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.write("### Payment Type Distribution")
        payment_counts = filtered_df['payment_type'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#34495e']
        payment_counts.plot.pie(autopct='%1.1f%%', ax=ax, colors=colors, 
                              textprops={'fontsize': 12}, wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
        ax.set_ylabel('')
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Average fare by payment type
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.write("### Average Fare by Payment Type")
        avg_fare = filtered_df.groupby('payment_type')['total_amount'].mean().sort_values()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=avg_fare.values, y=avg_fare.index, palette='Blues_d', ax=ax)
        ax.set_xlabel('Average Total Amount ($)', fontsize=12)
        ax.set_ylabel('Payment Type', fontsize=12)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Relationship between distance and fare
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.write("### Trip Distance vs Total Amount")
    sample_df = filtered_df.sample(min(1000, len(filtered_df)))  # Sample for better performance
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=sample_df, x='trip_distance', y='total_amount', 
                   alpha=0.6, color='#3498db', ax=ax)
    ax.set_xlabel('Trip Distance (miles)', fontsize=12)
    ax.set_ylabel('Total Amount ($)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.subheader("Trip Characteristics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Passenger count distribution
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.write("### Passenger Count Distribution")
        passenger_counts = filtered_df['passenger_count'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=passenger_counts.index, y=passenger_counts.values, 
                   palette='Blues_d', ax=ax)
        ax.set_xlabel('Passenger Count', fontsize=12)
        ax.set_ylabel('Number of Trips', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Trip duration distribution
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.write("### Trip Duration Distribution")
        duration_data = filtered_df[filtered_df['trip_duration'] <= filtered_df['trip_duration'].quantile(0.95)]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(duration_data['trip_duration'], bins=30, kde=True, 
                    color='#3498db', ax=ax)
        ax.set_xlabel('Trip Duration (minutes)', fontsize=12)
        ax.set_ylabel('Number of Trips', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Top pickup locations
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.write("### Top 10 Pickup Locations")
    top_pickups = filtered_df['PULocationID'].value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top_pickups.index, y=top_pickups.values, palette='Blues_d', ax=ax)
    ax.set_xlabel('Pickup Location ID', fontsize=12)
    ax.set_ylabel('Number of Trips', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# Add footer
st.markdown("---")
st.markdown("""
**Note:** This dashboard provides interactive analysis of NYC taxi trip patterns based on your selected parameters. 
Data is simulated for demonstration purposes.
""")