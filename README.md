# Wholesale Customer Segmentation

This project is a web application for segmenting wholesale customers based on their spending on different product categories. The application is built with Streamlit and uses machine learning algorithms to perform customer segmentation.

## Description

The application allows users to:

*   Load wholesale customer data from a CSV file.
*   Perform K-Means clustering to group customers into segments.
*   Perform Hierarchical clustering to create a hierarchy of customer segments.
*   Visualize the clustering results using 3D scatter plots and dendrograms.
*   View a preview of the raw data.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/wholesale-customer-segmentation.git
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Make sure you have the `Wholesale customers data.csv` file in the same directory as the application.
2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
3.  The application will open in your web browser. You can then use the sidebar to select the clustering algorithm and its parameters.

## Data

The dataset used in this project is the "Wholesale customers data" from the UCI Machine Learning Repository. It contains information about the annual spending of wholesale customers on different product categories.

## Algorithms

The application implements the following clustering algorithms:

*   **K-Means Clustering:** An iterative algorithm that partitions the data into K pre-defined number of clusters.
*   **Agglomerative Clustering:** A hierarchical clustering algorithm that builds a hierarchy of clusters.

## Contributors

*   SMARAK DAS
*   GAGAN KUMAR
*   RUDRANARAYAN TRIPATHY
*   ANSHUMAN PANIGRAHI
*   RUDRANARAYAN DEBATA
