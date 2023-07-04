# Inc 5000 Exploration

## Description

This project is an exploratory data analysis of the 2022 Inc. 5000. The goal of the project is to provide a comprehensive analysis of the data, using Python to clean, analyze, and visualize the data. The data was scraped from the Inc. 5000 website using BeautifulSoup4 to send a request to their API, and the JSON data returned was cleaned for analysis.

## Project

The project is a data analysis of the Inc. 5000 list for the year 2022. It uses Python and data analysis libraries to process and visualize information on America's fastest growing companies. The data was collected by making a request to the Inc. 5000 API, and the returned JSON data was cleaned and transformed into a pandas DataFrame for analysis. The project is publicly deployed using Streamlit.

## Usage examples

The main script of the project is `inc5000dash.py`, which can be run locally using 'streamlit run inc5000dash.py' to start the analysis process. The script includes data cleaning, exploratory data analysis, and visualization of the Inc. 5000 data.

## Issues or limitations of the project

As the project is based on the 2022 Inc. 5000 data, it may not be applicable or accurate for other years or datasets. The data was scraped from the Inc. 5000 website, and any changes to their API or data structure may inhibit future data acquisition efforts.

## Future features

Once the next Inc. 5000 report is released, this project will be updated with the next report, assuming the api endpoint is still functional. Over time, with data from multiple years, we will get an idea of short and long-term trends in the market.

## Technologies

The project is built with Python and uses the following libraries:

- pandas
- plotly.express
- scikit-learn
- numpy
- streamlit
- json

These libraries were chosen for their robust data processing, visualization capabilities, and ability to create interactive web applications.

## Details about use

To use the project, clone the repository and run the `inc5000dash.py` script. The script will scrape the data, clean it, perform the analysis, and visualize the results in a Streamlit application.

## Contribution guidelines

Contributions are welcome. Please fork the repository and create a pull request with your changes.

## Credits

This project was created by danmarino1. Thank you to the folks at Inc. for publishing such a comprehensive report of organizations.


## Dependencies

The project has the following dependencies:

- pandas==1.3.5
- pip==23.1
- plotly.express
- scikit-learn==1.0.2
- numpy
- streamlit
- json
