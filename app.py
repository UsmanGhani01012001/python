from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load pre-trained model
with open('kmeans_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file, parse_dates=["InvoiceDate"])
            df['CustomerID'] = df['CustomerID'].astype(str)
            df['Description'] = df['Description'].fillna(df['Description'].mode()[0])
            df = df.drop_duplicates()
            df['Amount'] = df['Quantity'] * df['UnitPrice']

            # RFM Feature Engineering
            monetary = df.groupby('CustomerID')['Amount'].sum().reset_index()
            recency = df.groupby('CustomerID')['InvoiceDate'].max().apply(
                lambda x: (df['InvoiceDate'].max() - x).days).reset_index()
            recency.columns = ['CustomerID', 'Recency']
            frequency = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
            frequency.columns = ['CustomerID', 'Frequency']

            rfm = pd.merge(pd.merge(monetary, recency, on='CustomerID'), frequency, on='CustomerID')

            def remove_outliers_iqr(df, cols):
                for col in cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower) & (df[col] <= upper)]
                return df

            rfm_clean = remove_outliers_iqr(rfm, ['Amount', 'Recency', 'Frequency'])
            rfm_unscaled = rfm_clean.copy()

            # Scaling and Predicting Clusters
            scaler = StandardScaler()
            rfm_clean[['Amount', 'Recency', 'Frequency']] = scaler.fit_transform(rfm_clean[['Amount', 'Recency', 'Frequency']])
            rfm_unscaled['ClusterId'] = model.predict(rfm_clean[['Amount', 'Recency', 'Frequency']])

            # Plot and save stripplots
            for feature in ['Amount', 'Recency', 'Frequency']:
                plt.figure(figsize=(10, 6))
                sns.stripplot(x='ClusterId', y=feature, data=rfm_unscaled, palette='Set1', jitter=True, size=5, alpha=0.7)
                plt.title(f'Distribution of {feature} by Cluster')
                plt.xlabel('Cluster ID')
                plt.ylabel(feature)
                plt.grid(True)
                plot_path = f'static/{feature}_plot.png'
                plt.savefig(plot_path)
                plt.close()

            return render_template('index.html',
                                   amount_img='static/Amount_plot.png',
                                   recency_img='static/Recency_plot.png',
                                   frequency_img='static/Frequency_plot.png')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
