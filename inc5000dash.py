import numpy as np
import pandas as pd
import streamlit as st
import json
import plotly.express as px

from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title='Inc. 5000', page_icon=':chart_with_upwards_trend:')
st.title('Looking into the Inc. 5000')
st.write("""I am often fascinated by the companies that show the most promise
         towards remarkable innovations.
         Think about how far we've come just in the past 25 years:
         Google, Amazon, Uber, and now OpenAI have all been considered
         young companies in this quarter century alone, and some still are.
         Companies that move quickly to innovate, those who grow substantially,
         those companies are the ones we should keep our eyes on when
         imagining the potential our future has.
         And for some, they're great to keep an eye on for remarkable returns.
         Come with me as we explore the 5000 fastest growing companies in the US.""")
st.write("See the full 2022 Inc.5000 report here: https://www.inc.com/inc5000/2022")
st.button("Skip to Analysis",)
st.header("Extracting the data")
background_info = st.expander("Want to learn more about this dataset?")
with background_info:
    st.info("""When Inc. released their Inc. 5000 list in 2022,
    I greatly enjoyed scrolling through it and learning about the different companies
    that earned their spots on the list. While scrolling through the nicely presented
    website was informative, I realized there wasn't a central spot where this data
    was readily available.\nI quickly realized that if this list could be extracted
    and analyzed, it could reveal a tremendous amount of insight about the new organizations
    disrupting their fields and changing the world.
    \n I decided the best course of action would be to create a web scraper and
     download the data myself. After several attempts, I found a
    soltution that provided me with the json file you see below (more on that story in a future post).
    \nFollow me as we clean and explore this unique dataset and the business trends it reveals!""")

    st.write("""Inc. has a robots.txt file that specifies what can be accessed by web scrapers.
                 It is listed below, and it does not forbid scrapers from accessing the Inc. 5000.
                 Respect towards site oweners' requests is critical, and you should always consult
                 a robots.txt file if you undertake a similar project""")
    st.write('It can be found here: https://www.inc.com/robots.txt')
with open('inc_500_all_data.json') as access_json:
     read_document = json.load(access_json)

#---------------------------------------------------------------------
st.header("Cleaning up the data")
st.write("JSON data, while still structured data, doesn't often present immediately available data to use without some digging and cleaning.")
st.write("With our raw data, let's clean it up and ensure it's ready for a thorough analysis.")
st.write("Here's an example of one of our first company in the list")
# Getting the top ranked company (index 0 in the list of companies)
st.write(read_document['companies'][0].values())
st.write("As you can see, it's not in a usable format, so we'll have to clean it up using Pandas")
json_details_expander = st.expander('If you want to know a bit more about extracting the company data from JSON')
with json_details_expander:
    st.write("This file came from an api endpoint that the site called when loading in the list. It's a JSON file with a three main objects. The first is site information, and the second holds most of the company information on the site. This is the information we want")
    st.write(read_document['companies'][0])

st.write("Let's convert this JSON data into a pandas dataframe:")
company_info =  read_document['companies']
inc5000_companies = pd.DataFrame(company_info)
st.dataframe(inc5000_companies.sample(5))

# Fixing dtypes for year
inc5000_companies.drop_duplicates(subset='company')

# No companies were founded in the year 0. These are being changed to np.nan
inc5000_companies['founded'] = inc5000_companies['founded'].replace(0, np.nan)

st.write("There are some columns with only null values and others that exist for subsets of companies")
# Count the number of nulls in each column
null_counts = inc5000_companies.isnull().sum()

# Combine the null counts with the column names and dtypes
metadata = pd.DataFrame({
    "Column Name": inc5000_companies.columns,
    "Number of Nulls": null_counts,
    "Data Type": inc5000_companies.dtypes,
})
st.dataframe(metadata) #inc5000_companies.info() was not working with streamlit. I took another approach here

st.write(f'Original Columns: {inc5000_companies.columns}')
old_column_count = len(inc5000_companies.columns)
inc5000_companies.dropna(axis = 1,
                         thresh=5 ,
                         inplace=True) # Drop columns with more than 5 non-null values
new_column_count = len(inc5000_companies.columns)
st.write(f"{old_column_count - new_column_count} columns removed. {new_column_count} columns remaining.")

st.write("I'm going to narrow this dataset down to some particularly interesting columns")
cols_to_keep = ['rank', 'company', 'workers', 'previous_workers', 'ceo_gender',
'website', 'state_s', 'city', 'growth', 'revenue', 'industry',
'zipcode', 'founded', 'raw_revenue', 'yrs_on_list', 'editorsPick', 'ifc_business_model']
inc5000_companies = inc5000_companies[cols_to_keep]
inc5000_companies.columns = ['Rank', 'Company', 'Employees', 'Employees Last Year', 'CEO Gender',
'Website', 'State', 'City', '% Growth (3yr)', 'Revenue Bucket', 'Industry',
'Zipcode', 'Founded', 'Revenue', 'Years on List', 'Tags', 'Description']
st.write(f'New Columns: {inc5000_companies.columns}')
st.write("With our data cleaned and ready, we can begin our deep dive into the world of fast-growing companies, uncovering patterns and trends that shape these businesses.")

#------------------------------------------
st.header("Analyzing the data")
st.write("Let's start by examining various distributions in the dataset, such as CEO genders, states, industries, and other factors among these rapidly growing companies.")
st.write("Let's look at some of the distributions presented in the Inc. 5000")

columns_for_value_counts = ['State', 'Industry', 'Founded', 'Years on List']
column_selected = st.selectbox(label = 'Select a column to see the distribution.',
                               options = columns_for_value_counts, index=3)

value_counts = inc5000_companies[column_selected].value_counts().reset_index()
value_counts.columns = ['value', column_selected]
st.dataframe(value_counts)

# Show a histogram of the values
st.plotly_chart(px.histogram(inc5000_companies,
                             x = column_selected,
                             title = f"Inc. 5000 representation by {column_selected}"))

# Revenue Analysis
st.header("Let's take a look at how much revenue these organizations have brought in last year")
st.write("Below you'll see some basic descriptive statistics of the revenue made by these companies")
st.write(inc5000_companies['Revenue'].describe().round(2))
st.write("""Definitely a massive difference between the median and the mean as well as a massive standard deviation.
         Outliers are definitely playing a huge role here (As we can expect
         among even the most rapidly growing startups)""")

st.write("Let's take a look at some of the top ranked companies on the list")
sample_col1, sample_col2 = st.columns(2)
with sample_col1:
    st.write("Top 10 companies by 3 year Growth")
    st.write(inc5000_companies.sort_values(by='% Growth (3yr)', ascending=False).head(10))
with sample_col2:
    st.write("Top 10 companies by Revenue")
    st.write(inc5000_companies.sort_values(by='Revenue', ascending=False).head(10))

st.write("Let's take a deeper dive into the revenue reportings from these companies")
options = ["All", "Bottom 25%", "Middle 25-75%", "Top 25%", "Top 10%"]
option_choice = st.selectbox("Which group should we look at?", options=options)
if option_choice == 'All':
    st.write('As we can see in both the descriptive stats and the chart below, the outliers are in a league of their own.')

q25, q50, q75, q90 = inc5000_companies['Revenue'].quantile([0.25, 0.5, 0.75, .9])

if option_choice == "All":
    mask = inc5000_companies['Revenue'] > 0
elif option_choice == 'Bottom 25%':
    mask = inc5000_companies['Revenue'] < q25
elif option_choice == 'Middle 25-75%':
    mask = inc5000_companies['Revenue'].between(q25, q75)
elif option_choice == 'Top 25%':
    mask = inc5000_companies['Revenue'] > q75
else:
    mask = inc5000_companies['Revenue'] > q90

st.plotly_chart(px.histogram(inc5000_companies[mask],
                                  x="Revenue",
                                  title=f'Distribution of Revenue ({option_choice})'))
st.plotly_chart(px.box(inc5000_companies,
                          x = 'Revenue',
                          title = 'Distribution of Total Revenue by Company', log_x=True))

st.header("Let's take a peek at how much each US state is represented. (the Inc. 5000 highlights American companies, so they're all based in a state)")
# Groupby state and figure out Company count and Total Revenue
state_summary = inc5000_companies.groupby('State').agg({'Revenue': 'sum', 'Company': 'count'})
state_summary = state_summary.rename(columns={'Revenue': 'Total Revenue', 'Company': 'Number of Companies'}).reset_index()
st.write(state_summary)


total_revenue = state_summary['Total Revenue'].sum()

# Define a Plotly Express choropleth map
fig = px.choropleth(
    state_summary,
    locations='State',
    color='Total Revenue',
    hover_data=['Number of Companies'],
    scope='usa',
    locationmode='USA-states',
    color_continuous_scale='Mint',
    range_color=(0, state_summary['Total Revenue'].max()),
    labels={'Total Revenue':'Total Revenue ($)', 'Number of Companies':'Number of Companies',},
    title= "Total revenue brought in by Inc. 5000 companies, by state")

# Display the figure in Streamlit
st.plotly_chart(fig)
st.write("""It's clear here that California,
         the most well known for startups,
         is an outlier among all states for most revenue.
         Texas trails behind,
         then Ohio, Virgnia, Florida, and Arizona all end up in a similar place""")

# Get top cities by revenue
top_n_cities = st.slider("How many of the top cities should we look at?",
                         min_value=5, max_value=50, value=25)
all_cities = inc5000_companies.groupby(["State", 'City'], as_index=False).agg({'Revenue': ['sum', 'count']})
all_cities.columns = ['State', 'City', 'Total Revenue ($)', 'Number of Companies']
st.dataframe(all_cities)
top_cities_list = all_cities.sort_values(by='Total Revenue ($)', ascending=False).head(top_n_cities)['City'].tolist()
top_cities = all_cities[all_cities['City'].isin(top_cities_list)]

# Create the bar chart using Plotly Express
city_chart = px.bar(top_cities, x='City', color = 'State', y='Total Revenue ($)',
                    hover_data=['Number of Companies'],
                    title=f'Total Revenue by City (Top {top_n_cities})')
st.plotly_chart(city_chart)

st.write("""Like the states, there are a few cities that tower above the rest.
         This is often due to either many companies (like San Fransisco) in an area or one or two outlier
         companies (like Eighty Four, PA)""")

#Looking into industries
st.header("Exploring industries")
st.write("""Knowing which companies are thriving can be helpful if you're looking for inspiration,
a job, or case studies. However, if you're interested in seeing what markets are moving, it's important to look at which industries are thriving.""")

# Create a vertical stacked bar chart with Industry as the y-axis and Company as the stacked bars
industry_col1, industry_col2 = st.columns(2)
industry_chart = px.bar(inc5000_companies,
                        y='Industry',
                        x='Revenue',
                        title='Total Revenue by Industry',
                        hover_name = 'Company')
with industry_col1:
    st.plotly_chart(industry_chart)


# Group the data by Industry and rank the companies within each industry based on their revenue
inc5000_companies['Rank'] = inc5000_companies.groupby('Industry')['Revenue'].rank(ascending=False)

# Filter the data to only include the top 10 companies in each industry
top_companies = inc5000_companies[inc5000_companies['Rank'] <= 10]
color_map = {company: color for company, color in zip(top_companies['Company'].unique(), px.colors.qualitative.Dark24)}
# Create a horizontal stacked bar chart with Industry as the x-axis and Company as the stacked bars
industry_chart = px.bar(top_companies,
                        x='Industry',
                        y='Revenue',
                        color='Company',
                        title='Top 10 Companies per Industry by Revenue',
                        color_discrete_map=color_map)
industry_chart.update_layout(barmode='stack', xaxis_tickangle=-45, showlegend=False)
st.plotly_chart(industry_chart)
st.write("Exploring the industries these companies belong to, we notice that some industries have a few standout companies with massive revenue, while others have a more balanced distribution.")
st.write("We have some companies giving their industries a substantial boost in revenue, particularly in HR, Logistics and Security Services, enabling them to tower over other industries")

# Create a vertical stacked bar chart with Industry as the y-axis and Company as the stacked bars
outliers_removed = inc5000_companies.Revenue < q90
industry_chart = px.bar(inc5000_companies.loc[outliers_removed],
                        y='Industry',
                        x='Revenue',
                        title='Total Revenue by Industry - 90th percentile companies removed',
                        orientation='h',
                        hover_name ='Company',
                        color_discrete_map=color_map)
industry_chart.update_layout(barmode='stack', height=1000, font=dict(size=8))
st.plotly_chart(industry_chart)
st.write("Removing the top 10% of companies by revenue gives us a clearer view of the overall landscape, revealing that many industries have a more even playing field without the outliers.")
st.write(f"When we eliminate the 90th percentile companies (our outliers), it appears many more industries are close to one another. Notably, software leads the pack followed by a variety of business services across HR, Finance, Logistics, and Marketing.")

st.header("Looking into employee count by companies and how much revenue is brought in")
st.write("Next, let's analyze the relationship between employee count and revenue generated by these companies.")
inc5000_companies['Revenue per Employee'] = inc5000_companies.Revenue / inc5000_companies.Employees

employee_col1, employee_col2, employee_col3 = st.columns(3)
with employee_col1:
    st.dataframe(inc5000_companies['Revenue per Employee'].describe())
with employee_col2:
    st.plotly_chart(px.box(inc5000_companies, x='Revenue per Employee', hover_name = 'Company',hover_data = ['Employees', 'Revenue']))

st.header("Analyzing company growth")
st.write("""Companies don't get on the Inc. 5000 list because they had high revenue.
           They're on the list beceause they've grown fast. Let's look at their three year growth""")

st.dataframe(inc5000_companies['% Growth (3yr)'].describe())
st.plotly_chart(px.histogram(inc5000_companies.loc[outliers_removed], '% Growth (3yr)', title = '3 year growth (%) - Outliers Removed'))
st.plotly_chart(px.box(inc5000_companies.loc[outliers_removed], '% Growth (3yr)', title = '3 year growth (%) - Outliers Removed'))



current_year = datetime.now().year
inc5000_companies['Company Age'] = current_year - inc5000_companies['Founded']
import plotly.express as px

fig = px.scatter(inc5000_companies, x='Company Age', y='% Growth (3yr)', title='Company Age vs 3-Year Growth', hover_name="Company")
st.plotly_chart(fig)
st.write("""While there is representation from companies ranging in age from less
        than five years old to over 105 years old, we see the highest growing companies are new.
         This maskes sense, as scaling a new organization is much more likely than scaling an organization
         that has already established itself in a marketplace""")

st.write("Let's create some buckets for company ages")
bucket_list = ['0-5 years', '5-10 years', '10-20 years',
               '20-50 years', '50-75 years', '75+ years']
st.write(f"Here will be our age buckets: {bucket_list}")
def bucket_ages(x):
    if x <= 5:
        x = '0-5 years'
    elif x <= 10:
        x = '5-10 years'
    elif x <= 20:
        x = '10-20 years'
    elif x <= 50:
        x = '20-50 years'
    elif x <= 75:
        x = '50-75 years'
    else:
        x = '75+ years'
    return x

inc5000_companies['Age Bucket'] = inc5000_companies['Company Age'].apply(bucket_ages)
histogram = px.histogram(inc5000_companies,
                         x='Age Bucket',
                         title="Histogram of Company Age Buckets",
                         category_orders={'Age Bucket': bucket_list})
st.plotly_chart(histogram)
st.header("Let's utilize NLP to understand business descriptions")
dropdown = st.expander('Show Details of what we are doing here')
with dropdown:
        st.write("""TF-IDF stands for "Term Frequency-Inverse Document Frequency",
which is a numerical statistic that helps us understand the
importance of words across a document. In this code, we're using TF-IDF
 to find pairs of companies that have similar descriptions.
To do this, we first clean the data by removing duplicate companies and
null descriptions. We then use the TfidfVectorizer class from scikit-learn
to create a matrix of TF-IDF values for each description. This matrix represents
the importance of each word in each description. We then use cosine similarity
to find pairs of descriptions that are similar to each other based on their TF-IDF values.
Finally, we display a dataframe with pairs of companies that have similar descriptions.
Overall, this process helps us to better understand the business descriptions of
these companies and identify potential competitors or similar companies.""")
         

#Drop Duplicates and get rid of null descriptions
inc5000_companies.drop_duplicates(subset='Company', inplace=True)
inc5000_companies.drop_duplicates(subset='Description', inplace=True)
inc5000_companies.dropna(subset=["Description"], inplace=True)

@st.cache_data
def get_similar_descriptions():
    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words="english")
    # Fit vectorizer on all descriptions and eliminate stopwords
    vectorizer.fit(inc5000_companies["Description"])
    # Transform each description into a vector
    doc_vectors = vectorizer.transform(inc5000_companies["Description"])
    # Compute the pairwise cosine similarity between all descriptions
    similarity_matrix = cosine_similarity(doc_vectors)

    # Find pairs of descriptions with high similarity
    threshold = 0.5
    similar_pairs = []
    for i in range(similarity_matrix.shape[0]):
        for j in range(i+1, similarity_matrix.shape[1]):
            if similarity_matrix[i,j] > threshold:
                similarity_score = similarity_matrix[i,j]
                company1 = inc5000_companies.iloc[i]["Company"]
                company2 = inc5000_companies.iloc[j]["Company"]
                description1 = inc5000_companies.iloc[i]["Description"]
                description2 = inc5000_companies.iloc[j]["Description"]
                similar_pairs.append((company1, company2, round(similarity_score * 100, 1), description1, description2))
    similar_descriptions = pd.DataFrame(similar_pairs, columns=["Company A", "Company B", "Similarity (%)", "Description A", "Description B"])
    return similarity_matrix, similar_descriptions, threshold
#Call the above function
similarity_matrix, similar_descriptions, threshold = get_similar_descriptions()
st.dataframe(similar_descriptions)

st.write("""There are a lot of connections here among companies with similar descriptions.
         These organizations are likely competitors, and this information
         opens the door for a much more detailed analysis.""")

# Initialize empty dictionary to store similar companies for each company
similar_companies_dict = {}
# Iterate over each row in similarity_matrix
for i, row in enumerate(similarity_matrix):
    # Get the name of the company for this row
    company1 = inc5000_companies.iloc[i]["Company"]
    # Find the indices of companies with high similarity to this company
    similar_indices = np.where(row > threshold)[0]
    # Iterate over the similar indices
    for j in similar_indices:
        # Skip self-similarity
        if i == j:
            continue
        # Get the name of the similar company
        company2 = inc5000_companies.iloc[j]["Company"]
        # Add company2 to the similar companies dictionary of company1
        if company1 not in similar_companies_dict:
            similar_companies_dict[company1] = {}
        similar_companies_dict[company1][company2] = round(row[j] * 100, 1)

# Create a new dataframe with two columns: Company and Similar Companies
company_list = []
similar_companies_list = []
for company, similar_companies_dict in similar_companies_dict.items():
    company_list.append(company)
    similar_companies_list.append(similar_companies_dict)
st.write('Below we have a more concise dataframe which holds the companies with similar descriptions to others as well as all companies the companies they are similar to and how similar they are to those companies')
similar_companies_df = pd.DataFrame({"Company": company_list, "Similar Companies": similar_companies_list})
st.dataframe(similar_companies_df)

st.write("""There's a lot more to uncover here between the companies' descriptions,
our competitor detection, and even looking at companies' performance over time once
the next report comes out in 2023. I hope this exploration sparked some interest
in you towards fast-growing companies, and stay tuned for deeper dives into this data
once the next report is published!""")