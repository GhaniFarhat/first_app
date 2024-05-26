import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot, boxcox
from sqlalchemy import create_engine
import mysql.connector
import warnings
warnings.filterwarnings('ignore')

# Streamlit app
st.title("Data Visualization Dashboard")

# Database connection using secrets
host = st.secrets["database"]["host"]
port = st.secrets["database"]["port"]
database = st.secrets["database"]["database"]
username = st.secrets["database"]["username"]
password = st.secrets["database"]["password"]

# Connect to MySQL server
try:
    connection = mysql.connector.connect(
        host=host,
        user=username,
        password=password,
        database=database,
        port=port
    )
    st.success("Database connection established!")
except mysql.connector.Error as err:
    st.error(f"Error: {err}")
    st.stop()

# Defined SQL query to select data from the table
sql_query = "SELECT * FROM agricultural_combined_dataset"

# Fetch data from MySQL database into a DataFrame
try:
    data = pd.read_sql(sql_query, connection)
except Exception as e:
    st.error(f"Error reading data: {e}")
    st.stop()

# QQ Plot
st.header('QQ Plot')
column_name = "Amount_Euro"
numerical_data = data[column_name].dropna()
transformed_data, lambda_value = boxcox(numerical_data + 1)
fig, ax = plt.subplots()
probplot(transformed_data, dist="norm", plot=ax)
ax.set_title('QQ-plot for Normality Test (Box-Cox Transformed Data)')
ax.set_xlabel('Theoretical Quantiles')
ax.set_ylabel('Ordered Values')
st.pyplot(fig)

# Ireland vs Greece
st.header('Ireland vs Greece')
ireland_aggregate = data[(data['Country'] == 'Ireland') & (data['Category'] == 'Animal Skins & Furs')]['Amount_Euro'].sum()
greece_aggregate = data[(data['Country'] == 'Greece') & (data['Category'] == 'Animal Skins & Furs')]['Amount_Euro'].sum()
aggregated_data = pd.DataFrame({'Country': ['Ireland', 'Greece'], 'Amount_Euro': [ireland_aggregate, greece_aggregate]})
fig, ax = plt.subplots()
aggregated_data.plot(kind='bar', x='Country', y='Amount_Euro', legend=False, color=['blue', 'green'], ax=ax)
ax.set_title('Aggregated Amount_Euro for Animal Skins & Furs (Ireland vs. Greece)')
ax.set_xlabel('Country')
ax.set_ylabel('Amount_Euro')
st.pyplot(fig)

# Total Labour Persons
st.header('Total Labour Persons')
agg_data = data.groupby(['Country'])['total_labour_persons'].sum().reset_index()
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x='Country', y='total_labour_persons', data=agg_data, ax=ax)
ax.set_title('Total Labour Persons by Country (Aggregated)')
ax.set_xlabel('Country')
ax.set_ylabel('Total Labour Persons')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)

# Trends over Time
st.header('Trends over Time')
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
category = 'Animal Feed'
category_data = data[data['Category'] == category]
grouped_data = category_data.groupby('Year').agg({'Amount_Euro': 'sum', 'Quantity_Tonnes': 'sum'}).reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=grouped_data, x='Year', y='Amount_Euro', marker='o', label='Amount (€)', ax=ax)
sns.lineplot(data=grouped_data, x='Year', y='Quantity_Tonnes', marker='o', label='Quantity_Tonnes', ax=ax)
ax.set_title(f'Trends over Time for {category}')
ax.set_xlabel('Year')
ax.set_ylabel('Value')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Increase in Amount over the Years
st.header('Increase in Amount over the Years')
yearly_totals = data.groupby('Year')['Amount_Euro'].sum().reset_index()
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(yearly_totals['Year'], yearly_totals['Amount_Euro'], marker='o', linestyle='-')
ax.set_title('Increase in Amount (€) Over the Years')
ax.set_xlabel('Year')
ax.set_ylabel('Total Amount (€)')
ax.grid(True)
st.pyplot(fig)

# Data Columns Histogram
st.header('Data Columns Histogram')
columns_to_visualize = ['Amount_Euro', 'Quantity_Tonnes', 'total_labour_persons', 'Total_land_use_for_number_of_farm']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, col in enumerate(columns_to_visualize):
    sns.histplot(data[col], kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
st.pyplot(fig)

# Bar Chart for Mean Quantity (Tonnes)
st.header('Bar Chart for Mean Quantity (Tonnes)')
mean_quantity = data.groupby('Category')['Quantity_Tonnes'].mean()
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(mean_quantity.index, mean_quantity.values, color='lightblue')
ax.set_title('Mean Quantity (Tonnes) for Each Category')
ax.set_xlabel('Category')
ax.set_ylabel('Mean Quantity_Tonnes')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)

# Correlation Matrix
st.header('Correlation Matrix')
columns_of_interest = ['Amount_Euro', 'Quantity_Tonnes']
subset_data = data[columns_of_interest]
correlation_matrix = subset_data.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
ax.set_title('Correlation Heatmap between Amount and Quantity')
st.pyplot(fig)

# Scatter Plot
st.header('Scatter Plot')
correlation = data['standard_output_EUR'].corr(data['farms_number'])
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='standard_output_EUR', y='farms_number', data=data, s=100, ax=ax)
sns.regplot(x='standard_output_EUR', y='farms_number', data=data, scatter=False, color='red', ax=ax)
ax.set_title('Standard Output (EUR) vs. Farms Number')
ax.set_xlabel('Standard Output (EUR)')
ax.set_ylabel('Farms Number')
ax.text(600, 55000, f'Correlation: {correlation:.2f}', fontsize=12)
st.pyplot(fig)

# Country Bar Plot
st.header('Country Bar Plot')
data_filtered = data[data['Country'] != 'United Kingdom']
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='Country', y='standard_output_EUR', data=data_filtered, palette='husl', ax=ax)
ax.set_title('Standard Output (EUR) by Country (Excluding United Kingdom)')
ax.set_xlabel('Country')
ax.set_ylabel('Standard Output (EUR)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)

# Comparison Bar Plot
st.header('Comparison Bar Plot')
ireland_data = data[data['Country'] == 'Ireland']
european_countries_data = data[data['Country'] != 'Ireland']
key_indicators = ['farms_number', 'used_agricultural_area_ha', 'total_labour_persons', 'total_labour_AWU']
for indicator in key_indicators:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Country', y=indicator, data=european_countries_data, palette='viridis', ax=ax)
    ax.plot(ireland_data['Country'], ireland_data[indicator], marker='o', color='red', label='Ireland')
    ax.set_title(f'Comparison of {indicator.replace("_", " ").title()} in European Countries with Ireland')
    ax.set_xlabel('Country')
    ax.set_ylabel(indicator.replace("_", " ").title())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend()
    st.pyplot(fig)

# Farm Sizes Over Time
st.header('Farm Sizes Over Time')
farm_sizes = data.groupby('Year').agg({
    'farms_SO_zero': 'sum',
    'farms_SO_less2000': 'sum',
    'farms_SO_2000_3999': 'sum',
    'farms_SO_4000_7999': 'sum',
    'farms_SO_8000_14999': 'sum',
    'farms_SO_15000_24999': 'sum',
    'farms_SO_25000_49999': 'sum',
    'farms_SO_50000_99999': 'sum',
    'farms_SO_100000_249999': 'sum',
    'farms_SO_250000_499999': 'sum',
    'farms_SO_500000_orover': 'sum',
}).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
for col in farm_sizes.columns[1:]:
    sns.lineplot(x='Year', y=col, data=farm_sizes, ax=ax, label=col)
ax.set_title('Farm Sizes Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Farms')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig)

# Close the connection
connection.close()
