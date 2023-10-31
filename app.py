from plotly import figure_factory
import plotly.graph_objects as go
import streamlit as st
import preprocessor
import helper
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Define custom CSS to set the background color
custom_css = f"""
    <style>
        .stApp {{
            background-color: '##ffffff';
        }}
    </style>
"""

# Apply the custom CSS to the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

df = pd.read_csv('athlete_events1.csv')
region_df = pd.read_csv('noc_regions.csv')

df = preprocessor.preprocess(df, region_df)

st.sidebar.title("PODIUM PROJECTIONS")

st.sidebar.image('olypics.jpg')
user_menu = st.sidebar.radio(
    'Select an Option',
    ('Medal Tally', 'Overall Analysis', 'Country-wise Analysis', 'Athlete-wise Analysis', 'Medal Prediction', 'Sport Prediction'
     , 'Physical Traits and Sporting Triumph')
)

if user_menu == 'Medal Tally':

    st.sidebar.header("Medal Tally")
    years, country = helper.country_year_list(df)

    selected_year = st.sidebar.selectbox("Select Year", years)
    selected_country = st.sidebar.selectbox("Select Country", country)

    medal_tally = helper.fetch_medal_tally(df, selected_year, selected_country)
    if (st.button("Medal Tally")):
        if selected_year == 'Overall' and selected_country == 'Overall':
            st.title("Overall Tally")
        if selected_year != 'Overall' and selected_country == 'Overall':
            st.title("Medal Tally in " + str(selected_year) + " Olympics")
        if selected_year == 'Overall' and selected_country != 'Overall':
            st.title(selected_country + " overall performance")
        if selected_year != 'Overall' and selected_country != 'Overall':
            st.title(selected_country + " performance in " + str(selected_year) + " Olympics")
        st.table(medal_tally)

if user_menu == 'Overall Analysis':
    editions = df['Year'].unique().shape[0] - 1
    cities = df['City'].unique().shape[0]
    sports = df['Sport'].unique().shape[0]
    events = df['Event'].unique().shape[0]
    athletes = df['Name'].unique().shape[0]
    nations = df['region'].unique().shape[0]

    st.title("Top Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Editions")
        st.title(editions)
    with col2:
        st.header("Hosts")
        st.title(cities)
    with col3:
        st.header("Sports")
        st.title(sports)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Events")
        st.title(events)
    with col2:
        st.header("Nations")
        st.title(nations)
    with col3:
        st.header("Athletes")
        st.title(athletes)

    st.title("Most successful Athletes")
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    selected_sport = st.selectbox('Select a Sport', sport_list)
    x = helper.most_successful(df, selected_sport)
    # Assuming 'x' is your DataFrame
    x = x.drop(columns=['count'])

    # Display the DataFrame
    st.table(x)

if user_menu == 'Country-wise Analysis':
    # st.title('Country-wise Analysis')
    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()

    selected_country = st.sidebar.selectbox('Select a country', country_list)

    country_df = helper.yearwise_medal_tally(df, selected_country)
    fig = px.line(country_df, x="Year", y="Medal")
    st.title(selected_country + "'s Medal tally over the years")
    st.plotly_chart(fig)
    try:
        st.title(selected_country + " excels in")
        pt = helper.country_event_heatmap(df, selected_country)
        fig, ax = plt.subplots(figsize=(20, 20))
        ax = sns.heatmap(pt, annot=True)
        st.pyplot(fig)
    except:
        st.write("no data")

if user_menu == 'Athlete-wise Analysis':
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()

    fig1 = plotly.figure_factory.create_distplot([x1, x2, x3, x4],
                                                 ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'],
                                                 show_hist=False, show_rug=False)

    fig1.update_layout(autosize=False, width=1000, height=600)
    st.title("Distribution of age")
    st.plotly_chart(fig1)

    x = []
    name = []
    famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                     'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                     'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                     'Water polo', 'Hockey', 'Rowing', 'Fencing',
                     'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery',
                     'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Rugby Sevens',
                     'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']
    # Create a subplot with a row for each sport
    fig = go.Figure()
    for sport in famous_sports:
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        temp_age_data = temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna()
        fig.add_trace(go.Histogram(x=temp_age_data, name=sport))

    # Update the layout of the figure
    fig.update_layout(barmode='overlay', title="Distribution of Age wrt Sports(Gold Medalist)")
    fig.update_traces(opacity=0.75)  # Adjust opacity for overlay effect
    fig.update_layout(autosize=False, width=1000, height=600)
    st.title("Distribution of age")
    st.plotly_chart(fig)

    sports_list = df['Sport'].unique()
    sports_list.sort()

    # Add a Streamlit selectbox for sport selection
    selected_sport = st.selectbox("Select a Sport", sports_list)

    if st.button("Debug Selected Sport"):
        st.write("Selected Sport:", selected_sport)  # Debugging output

    # Check if 'Overall' is selected
    if selected_sport == 'Overall':
        st.title("Overall Summary")
        # Add code for displaying an overall summary

    else:
        # Filter the DataFrame based on the selected sport
        temp_df = helper.weight_v_height(df, selected_sport)

        # Debugging output
        if st.button("Debug Filtered Data"):
            athlete_df = df.drop_duplicates(subset=['Name', 'region'])
            athlete_df['Medal'].fillna('No Medal', inplace=True)
            st.write(athlete_df)
            st.write("Filtered Data:")
            st.write(temp_df)

        # Create a scatterplot with hue and style
        fig, ax = plt.subplots()
        temp_df_long = pd.melt(temp_df, id_vars=['Weight', 'Height'], value_vars=['Medal', 'Sex'], var_name='Variable')

        ax = sns.scatterplot(data=temp_df_long, x='Weight', y='Height', hue='Variable', style='Variable', s=100)
        st.title("Height vs Weight")
        st.pyplot(fig)

        st.title("TOP ATHLETES")
        top_athletes = df.groupby('Name')['Medal'].count().sort_values(ascending=False).head(10)
        st.write(top_athletes)

if user_menu == 'Medal Prediction':
    st.title('Medal Prediction')

    df = df.dropna()

    # Splitting data into features (X) and target (y)
    X = df[['Age', 'Height', 'Weight']]
    y = df['Medal']

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.009, random_state=42)

    # Creating and training the random forest classifier
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Get user input for test data
    st.write("Enter test data:")
    user_age = st.slider('Age', min_value=13, max_value=45, step=1)
    user_height = st.slider('Height (cm)', min_value=100, max_value=220, step=1)
    user_weight = st.slider('Weight (kg)', min_value=30, max_value=200, step=1)

    # Create a DataFrame with the user's input
    user_data = pd.DataFrame({
        'Age': [user_age],
        'Height': [user_height],
        'Weight': [user_weight]
    })

    # Make predictions for the user's input data
    user_predictions = classifier.predict(user_data)

    # Display the prediction result
    st.write("User Input Data:")
    st.write(user_data)

    # You can also provide probability estimates for positive and negative outcomes
    class_probabilities = classifier.predict_proba(user_data)
    st.write(f"Probability of Winning a Medal: {class_probabilities[0][1]:.2f}")
    st.write(f"Probability of Not Winning a Medal: {class_probabilities[0][0]:.2f}")
    if(class_probabilities[0][1]>class_probabilities[0][0]):
        st.write("Higher probability to earn a medal!!!")
        if st.button('Predict Medal'):
            # Create a DataFrame with the input data
            input_data = pd.DataFrame({'Age': [user_age], 'Height': [user_height], 'Weight': [user_weight]})

            # Make a prediction using the trained classifier
            prediction = classifier.predict(input_data)

            # Display the predicted medal
            st.write(f'Predicted Medal: {prediction[0]}')
    else:
        st.write("Lesser probability to earn a medal!!! ")

    # Calculating accuracy (you may want to display this based on your dataset)
    accuracy = accuracy_score(y_test, classifier.predict(X_test))
    st.write(f'Accuracy on Test Data: {accuracy * 100:.2f}%')

    from sklearn.model_selection import cross_val_score

    # Perform cross-validation to get a more reliable accuracy estimate
    cv_scores = cross_val_score(classifier, X, y, cv=5)
    st.write(f'Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%')

    # Define a parameter grid to search through
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    # Perform Grid Search to find the best combination of hyperparameters
    grid_search = GridSearchCV(classifier, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and retrain the model
    best_params = grid_search.best_params_
    classifier = RandomForestClassifier(random_state=42, **best_params)
    classifier.fit(X_train, y_train)

    # Display the best parameters
    st.write(f'Best Hyperparameters: {best_params}')

    # Evaluate the model with cross-validation again after hyperparameter tuning
    cv_scores = cross_val_score(classifier, X, y, cv=5)
    st.write(f'Cross-Validation Accuracy after Hyperparameter Tuning: {cv_scores.mean() * 100:.2f}%')


    # Evaluate the model using classification report and confusion matrix
    y_pred = classifier.predict(X_test)
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    st.write(f'Classification Report:\n{classification_rep}')
    st.write(f'Confusion Matrix:\n{conf_matrix}')


if user_menu == 'Sport Prediction':
    st.title('Sport Prediction')
    famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                     'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                     'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                     'Water polo', 'Hockey', 'Rowing', 'Fencing',
                     'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery',
                     'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Rugby Sevens',
                     'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']

    df = df.dropna()

    # Assuming you have a dataset that includes 'Sport' as a target variable
    X = df[['Age', 'Height', 'Weight']]
    y = df['Sport']

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.009, random_state=42)

    # Creating and training the random forest classifier
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Get user input for test data
    st.write("Enter test data:")
    user_age = st.slider('Age', min_value=13, max_value=45, step=1)
    user_height = st.slider('Height (cm)', min_value=100, max_value=220, step=1)
    user_weight = st.slider('Weight (kg)', min_value=30, max_value=200, step=1)

    # Create a DataFrame with the user's input
    user_data = pd.DataFrame({
        'Age': [user_age],
        'Height': [user_height],
        'Weight': [user_weight]
    })

    # Make predictions for the user's input data
    predicted_sport = classifier.predict(user_data)

    # Display the predicted sport
    st.write("User Input Data:")
    st.write(user_data)
    st.write(f"Predicted Sport: {predicted_sport[0]}")

    # You can also display the top N sports with highest probabilities
    top_sports = classifier.predict_proba(user_data).argsort()[0][::-1][:5]  # Top 5 sports
    top_sport_names = [famous_sports[i] for i in top_sports]

    accuracy = accuracy_score(y_test, classifier.predict(X_test))
    st.write(f'Accuracy on Test Data: {accuracy * 100:.2f}%')



if user_menu == 'Physical Traits and Sporting Triumph':
    data = df.dropna(subset=['Medal'])

    colors = {'Gold': 'yellow', 'Silver': 'silver', 'Bronze': 'brown', np.nan: 'red'}
    labels = {'Gold': 'Gold', 'Silver': 'Silver', 'Bronze': 'Bronze', np.nan: 'No medal'}

    # Create a Streamlit app
    st.title('PHYSIQUE VS MEDAL')

    # Create a Matplotlib figure and axes
    fig, ax = plt.subplots()

    for medal, color in colors.items():
        subset = data[data['Medal'] == medal]
        ax.scatter(subset['Height'], subset['Weight'], label=labels[medal], c=color)

    # Set labels and title
    ax.set_xlabel('Height (cm)')
    ax.set_ylabel('Weight (kg)')
    ax.set_title('Scatter Plot with Labels')

    # Add a legend
    ax.legend()

    # Display the scatter plot using st.pyplot
    st.pyplot(fig)
