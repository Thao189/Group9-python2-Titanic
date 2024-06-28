import streamlit as st
import pandas as pd
import base64
import pickle
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from streamlit_extras.let_it_rain import rain
# Function to set the cover page background
def set_background(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/png;base64,{encoded}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center center;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Function to set the side bar background
def set_sidebar_background(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    sidebar_bg_img = f"""
    <style>
    [data-testid="stSidebar"] > div:first-child {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: 85%;
        background-position: middle;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(sidebar_bg_img, unsafe_allow_html=True)

# Set the initial sidebar background image
set_sidebar_background('3.png')

# Function to set main page background on button click
def set_background2(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    sidebar2_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(sidebar2_bg_img, unsafe_allow_html=True)
# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'cover'

# Set the main page background based on the session state
if st.session_state.page == 'cover':
    set_background('titanic_3.png')
else:
    set_background2('2.png')

# Function to hide the sidebar
def hide_sidebar():
    hide_sidebar_style = """
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        </style>
    """
    st.markdown(hide_sidebar_style, unsafe_allow_html=True)

# Function to display the introduction page
def cover_page():
    st.markdown("<h1 style='text-align: center; color: white;'>TITANIC DATA</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; color: white;'>Introduction</h2>", unsafe_allow_html=True)
    st.markdown("""
        <p style='color:white'>
       Welcome to our exploration trip on the Titanic dataset based on the well-known true Titanic shipwreck in 1992. Here you find a lot of necessary information you need to know about the true history of the previous decades, ranging from the survival ability and distribution of their age, gender, ticket class, and the station they are traveling to. The aim of this report is to delve into applying Project 2 in Python 2 from the Business IT Course . Moreover, our data will offer you descriptive insights about this crucial historical event and be more empathetic with it. Our report belongs to Group 9 on Thursday morning guided and supported positively by Mr. Do Duc Tan. Finally, we are curious about the crucial event in the human trove of knowledge about history and we will highlight it in detail. 
        <p style='color:white'>
       Particularly, the sinking of the Titanic is renowned as one of the most infamous disasters in history. On April 15, 1912, during its voyage, the supposedly unsinkable RMS Titanic sank after hitting an iceberg. Tragically, due to insufficient lifeboats, 1502 out of 2224 passengers and crew perished. Survival appeared to be influenced by certain factors, representing that some demographics were more likely to survive than others.
        </p>
    """, unsafe_allow_html=True)
    if st.button("Let's get started"):
        st.session_state.page = 'main'
        st.experimental_rerun()
# Function to inject dark theme CSS
def set_dark_theme():
    dark_theme_css = """
    <style>
    body {
        color: white;
        background-color: #1e1e1e;
    }
    .css-1aumxhk, .css-15tx938, .css-1kyxreq, .css-1e3in70 {
        background-color: #333 !important;
    }
    .css-1aumxhk .stButton button {
        background-color: #444;
        color: white;
    }
    </style>
    """
    st.markdown(dark_theme_css, unsafe_allow_html=True)
    
# Load the Titanic dataset from the workspace directory
@st.cache_data  # Cache the dataset for improved performance
def load_data():
    Titanic = pd.read_csv('Titanic.csv')
    return Titanic
# Function to display the main app content
def main_app():
    st.title("Hi👋 we're from group 9 class Business Class🌷֒")
    st.text('This is a web app to allow exploration of Titanic Survival🚢')

    # Load the dataset if it's not already loaded
    Titanic = load_data()
    # Sidebar setup
    st.sidebar.title('About This App✨')
    # Tabs setup
    tabs = st.sidebar.radio('Select what you want to display👇:', ['🏡Home', '🔎Data Explorer', '📊Features Distribution', '🌊Titanic Survival Prediction'])
    
    if tabs == '🏡Home':
        st.header("This is our Dataset📚")
        st.dataframe(Titanic)
        rain(
        emoji="🛥️",
        font_size=50,
        falling_speed=5,
        animation_length="2",
        )
    elif tabs == '🔎Data Explorer':
        st.subheader("Explore Dataset")
        eda(Titanic)
    elif tabs == '📊Features Distribution':
        st.subheader('Titanic Dataset Analysis')
        visualize(Titanic)
    elif tabs == '🌊Titanic Survival Prediction':
        st.header('Would you have survived the Titanic?🚢')
        predict_survival()

# Function for Data Explorer
def eda(df):
        # Add other column descriptions as needed
    explore_dataset_option = st.checkbox("🔎Explore Dataset")

    if explore_dataset_option:
        with st.expander("Explore Dataset Options", expanded=True):
            show_dataset_summary_option = st.checkbox("Show Dataset Summary")
            if show_dataset_summary_option:
                st.write(df.describe())
            show_dataset = st.checkbox("Show Dataset")
            if show_dataset:
                number = st.number_input("Number of rows to view", min_value=1, value=5)
                st.dataframe(df.head(number))

            show_columns_option = st.checkbox("Show Columns Names")
            if show_columns_option:
                st.write(df.columns)

            show_shape_option = st.checkbox("Show Shape of Dataset")
            if show_shape_option:
                st.write(df.shape)
                data_dim = st.radio("Show Dimension by ", ("Rows", "Columns"))
                if data_dim == "Columns":
                    st.text("Number of Columns")
                    st.write(df.shape[1])
                elif data_dim == "Rows":
                    st.text("Number of Rows")
                    st.write(df.shape[0])
                else:
                    st.write(df.shape)

            select_columns_option = st.checkbox("Select Column to show")
            if select_columns_option:
                all_columns = df.columns.tolist()
                selected_columns = st.multiselect("Select Columns", all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)

            show_value_counts_option = st.checkbox("Show Value Counts")
            if show_value_counts_option:
                all_columns = df.columns.tolist()
                selected_columns = st.selectbox("Select Column", all_columns)
                st.write(df[selected_columns].value_counts())

            show_data_types_option = st.checkbox("Show Data types")
            if show_data_types_option:
                st.text("Data Types")
                st.write(df.dtypes)

            show_summary_option = st.checkbox("Show Summary")
            if show_summary_option:
                st.text("Summary")
                st.write(df.describe().T)

            show_raw_data_option = st.checkbox('Show Raw Data')
            if show_raw_data_option:
                raw_data_rows = st.number_input("Number of Rows for Raw Data", min_value=1, value=5)
                raw_data_selection = df.head(raw_data_rows)
                selected_columns = st.multiselect("Select Columns", df.columns.tolist(), default=df.columns.tolist())
                new_df = raw_data_selection[selected_columns]
                st.dataframe(new_df)

            # Checkbox to show variable meanings
            show_variable_meanings_option = st.checkbox('Show Variable Meanings')
            if show_variable_meanings_option:
                st.text("Variable Meanings")
                st.markdown("""
                # Explain the meaning of features in Titanic Dataset🧐:
                - **Age:** Age of the passenger
                - **Parch:** Number of parents/children aboard the Titanic
                - **SibSp:** Number of siblings/spouses aboard the Titanic
                - **Survived:** Survival (0 = No, 1 = Yes)
                - **Pclass:** Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
                - **Name:** Name of the passenger
                - **Sex:** Sex of the passenger
                - **Ticket:** Ticket number
                - **Fare:** Fare paid for the ticket
                - **Cabin:** Cabin number
                - **Embarked:** Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
                """)
# Function for Features Distribution
def visualize(df):
    button_labels = [
        'Survival Distribution', 
        'Age Distribution', 
        'Family Size Distribution',
        'Class Distribution'
    ]

    selected_chart = st.radio('Select Chart', button_labels, index=0)

    if selected_chart == 'Survival Distribution':
        visualize_survival_distribution(df)
    elif selected_chart == 'Age Distribution':
        visualize_age_distribution(df)
    elif selected_chart == 'Family Size Distribution':
        visualize_family_size_distribution(df)
    elif selected_chart == 'Class Distribution':
        visualize_class_distribution(df)

def visualize_survival_distribution(df):
    st.subheader('Survival Distribution')

    df['Survived'] = df['Survived'].map({0: 'Not Survived', 1: 'Survived'})
    df['Embarked'] = df['Embarked'].fillna('Unknown')
    df['Embarked'] = df['Embarked'].map({'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown', 'Unknown': 'Unknown'})

    fig_survival = px.sunburst(df, path=['Survived', 'Sex', 'Embarked'],
                               color='Survived', color_discrete_map={'Survived': '#1f77b4', 'Not Survived': '#d62728'})
    
    fig_survival.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
    )

    # Set overall app background
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f2f6; /* Adjust as per your preference */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.plotly_chart(fig_survival)
    st.write('💬The sunburst chart visualizes the survival distribution of Titanic passengers, segmented by gender and starting places. Males exhibit a higher mortality rate, particularly those coming from Southampton. In contrast, females demonstrate a higher survival rate, especially those coming from Southampton.')

def visualize_age_distribution(df):
    st.subheader('Age Distribution')

    age_range = st.slider('Select Age Range', 
                          min_value=int(df['Age'].min()), 
                          max_value=int(df['Age'].max()), 
                          value=(int(df['Age'].min()), int(df['Age'].max())), step=20)

    fig_age = px.histogram(df, 
                           x='Age', 
                           range_x=age_range,
                           nbins=20,
                           title='Age Distribution',
                           labels={'Age': 'Age Count'},
                           color_discrete_sequence=['skyblue'])
    
    fig_age.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
    )
    # Set overall app background
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f2f6; /* Adjust as per your preference */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.plotly_chart(fig_age)
    st.write('💬This histogram graph represents age distribution with a prominent peak around 20 to 30 years old. Besides that, in this journey, most people come from the adolescents to middle age.')
def visualize_family_size_distribution(df):
    st.subheader('Family Size Distribution')

    # Calculating family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Categorizing family size
    df['FamilyCategory'] = df['FamilySize'].apply(lambda x: 'Alone' if x == 1 else ('Small Family' if x <= 4 else 'Large Family'))

    # Sidebar checkboxes for filtering by family size categories
    st.sidebar.subheader('Filter by Family Size')
    alone_checkbox = st.sidebar.checkbox('Alone', value=True)
    small_family_checkbox = st.sidebar.checkbox('Small Family', value=True)
    large_family_checkbox = st.sidebar.checkbox('Large Family', value=True)

    # Filter the dataframe based on selected checkboxes
    selected_categories = []
    if alone_checkbox:
        selected_categories.append('Alone')
    if small_family_checkbox:
        selected_categories.append('Small Family')
    if large_family_checkbox:
        selected_categories.append('Large Family')

    filtered_df = df[df['FamilyCategory'].isin(selected_categories)]

    # Counting family sizes in the filtered data
    family_size_counts = filtered_df['FamilySize'].value_counts().sort_index().reset_index()
    family_size_counts.columns = ['FamilySize', 'Count']

     # Creating a bar plot
    fig_family_size = px.bar(family_size_counts, x='FamilySize', y='Count',
                             title='Family Size Distribution',
                             labels={'FamilySize': 'Family Size', 'Count': 'Count'},
                             color='FamilySize', color_continuous_scale=px.colors.sequential.Blues)
    
    fig_family_size.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
    )

    # Set overall app background
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f2f6; /* Adjust as per your preference */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Displaying the plotly chart
    st.plotly_chart(fig_family_size)
    st.write('💬The bar chart illustrates the number of family members including alone, small houses (under 4 members) and large families affecting the people’s survival proportions. It is clear that there is a tendency for one person to choose traveling rather than the large families.')
def visualize_class_distribution(df):
    # Add checkboxes for filtering by Sex
    selected_sex = st.multiselect("Select Sex", df['Sex'].unique(), default=df['Sex'].unique())
    
    # Filter the dataframe based on selected Sex
    filtered_df = df[df['Sex'].isin(selected_sex)]
    
    # Mutate the 'Survived' and 'Pclass' columns
    filtered_df['Survived'] = filtered_df['Survived'].map({0: 'Death', 1: 'Survived'})
    filtered_df['Pclass'] = filtered_df['Pclass'].map({1: '1st', 2: '2nd', 3: '3rd'})
    
    # Ensure Pclass is ordered
    filtered_df['Pclass'] = pd.Categorical(filtered_df['Pclass'], categories=["1st", "2nd", "3rd"], ordered=True)
    
    # Create the interactive facet grid boxplot using Plotly
    fig = px.box(filtered_df, x='Pclass', y='Age', color='Pclass',
                 category_orders={"Pclass": ["1st", "2nd", "3rd"]},
                 labels={"Pclass": "Ticket Class", "Age": "Passenger Age"},
                 height=600, width=1000)
    
    fig.update_layout(
        title={
            'text': f'Age Distribution by Ticket Class',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    # Remove legend
    fig.update_traces(showlegend=False)

    # Set overall app background
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f2f6; /* Adjust as per your preference */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.plotly_chart(fig)
    st.write('💬The box plot illustrates the age distribution and the gender of passengers across three ticket classes. First-class passengers have the widest age range and higher median. Second-class passengers show a high second median age.Third-class passengers, with a lowest median age.')

# Titanic Prediction Function
def predict_survival():
    # Load training data and preprocess
    train = pd.read_csv('Titanic.csv')
    notebook_config = {  
        'random_state': 12345,  # for the GradientBoostingClassifier
        'n_jobs': 1,            # for the cross_val_score
        'cv': 10                # for the cross_val_score
    }

    train["Age"] = train["Age"].fillna(train["Age"].mean())
    train["Embarked"] = train["Embarked"].fillna("S")
    train = train.drop(columns=['PassengerId', 'Name', 'Cabin', 'Ticket'])
    train.loc[train['Fare'] > 300, 'Fare'] = 300
    train['num_relatives'] = train['SibSp'] + train['Parch']

    train['Pclass'] = train['Pclass'].astype(int)
    train['Sex'] = train['Sex'].map({'male': 0, 'female': 1}).astype(int)
    train['Embarked'] = train['Embarked'].map({'S': 'S', 'C': 'C', 'Q': 'Q'}).astype(str)

    # Preprocessing pipelines
    age_pipe = Pipeline(steps=[
        ('age_imp', SimpleImputer(strategy='median')),
        ('age_scale', MinMaxScaler())
    ])
    fare_pipe = Pipeline(steps=[
        ('fare_imp', SimpleImputer(strategy='mean')),
        ('fare_scale', MinMaxScaler())
    ])
    embarked_pipe = Pipeline(steps=[
        ('embarked_imp', SimpleImputer(strategy='most_frequent')),
        ('embarked_onehot', OneHotEncoder(drop=None))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('age_pipe', age_pipe, ['Age']),
            ('fare_pipe', fare_pipe, ['Fare']),
            ('embarked_pipe', embarked_pipe, ['Embarked']),
            ('minmax_scaler', MinMaxScaler(), ['SibSp', 'Parch', 'num_relatives']),
            ('pclass_onehot', OneHotEncoder(drop=None), ['Pclass']),
            ('sex_onehot', OneHotEncoder(drop='first'), ['Sex'])
        ]
    )

    # Gradient Boosting Classifier
    y_train = train['Survived'].values
    x_train = train.drop(columns=['Survived'])

    grad_boost = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('grad_boost', GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=notebook_config['random_state']
        ))
    ])

    grad_boost_acc = cross_val_score(
        estimator=grad_boost,
        X=x_train,
        y=y_train,
        scoring='accuracy',
        cv=notebook_config['cv'],
        n_jobs=notebook_config['n_jobs']
    )

    print('best GradientBoosting acc (mean) =', round(np.mean(grad_boost_acc), 2))
    print('best GradientBoosting acc (std)  =', round(np.std(grad_boost_acc), 2))

    grad_boost.fit(x_train, y_train)

    # Save the model
    with open("grad_boost.pkl", "wb") as f:
        pickle.dump(grad_boost, f)

    # Define Streamlit functions
    def get_user_data() -> pd.DataFrame:
        Titanic = {}

        Titanic['Age'] = st.slider('Enter Age:', min_value=0, max_value=80, value=20, step=1)
        Titanic['Fare'] = st.slider('How much did your ticket cost you? (in 1912$):', min_value=0, max_value=500, value=80, step=1)
        Titanic['SibSp'] = st.slider('Number of siblings and spouses aboard:', min_value=0, max_value=15, value=3, step=1)
        Titanic['Parch'] = st.slider('Number of parents and children aboard:', min_value=0, max_value=15, value=3, step=1)

        col1, col2, col3 = st.columns(3)
        Titanic['Pclass'] = col1.radio('Ticket class:', options=['1st', '2nd', '3rd'])
        Titanic['Sex'] = col2.radio('Sex:', options=['Man', 'Woman'])
        Titanic['Embarked'] = col3.radio('Port of Embarkation:', options=['Cherbourg', 'Queenstown', 'Southampton'], index=2)

        # Convert inputs to the same format as training data
        Titanic['Sex'] = 0 if Titanic['Sex'] == 'Man' else 1
        Titanic['Pclass'] = {'1st': 1, '2nd': 2, '3rd': 3}[Titanic['Pclass']]
        Titanic['Embarked'] = {'Cherbourg': 'C', 'Queenstown': 'Q', 'Southampton': 'S'}[Titanic['Embarked']]
        Titanic['num_relatives'] = Titanic['SibSp'] + Titanic['Parch']

        df = pd.DataFrame([Titanic])

        return df

    @st.cache_resource
    def load_model(model_file_path: str):
        with st.spinner("Loading model..."):
            with open(model_file_path, 'rb') as file:
                model = pickle.load(file)
        return model

    # Main function to run prediction
    def main():
        df_user_data = get_user_data()

        model = load_model(model_file_path="grad_boost.pkl")
        prob = model.predict_proba(df_user_data)[0][1]
        prob = int(prob * 100)

        emojis = ["😕", "🙃", "🙂", "😀"]
        state = min(prob // 25, 3)

        st.write('')
        st.title(f'{prob}% chance to survive! {emojis[state]}')
        if state == 0:
            st.error("Good luck next time, you will be next Jack! ☠️")
            st.image('image_copy_7.png')
        elif state == 1:
            st.warning("Hey... I hope you know how to swim, maybe you have to do it! 🏊‍♂️")
        elif state == 2:
            st.info("Well done! You are on the right track, but don't get lost! 💪")
        else:
            st.success('Congratulations! You can rest assured, you will be fine! 🎉')
            st.image('image_copy_6.png')

        if st.button("Facts"):
            st.markdown("""
            # Insider Survival Facts Based on the dataset from Kaggle:
        
            - **Overall Survival Rate:**
                - Only about 38.4% of passengers survived in this accident.
        
            - **Survival Rate by Gender:**
                - **Females:** 74.2% survival rate
                - **Males:** 18.9% survival rate
        
            - **Survival Rate by Ticket Class:**
                - **1st Class:** 62.96% survival rate
                - **2nd Class:** 47.28% survival rate
                - **3rd Class:** 24.24% survival rate
        
            - **Survival Rate by Age Group:**
                - **Children (0-12 years):** Approximately 58.33% survival rate
                - **Teenagers (13-19 years):** Approximately 38.46% survival rate
                - **Adults (20-50 years):** Approximately 35.80% survival rate
                - **Seniors (50+ years):** Approximately 28.33% survival rate
        
            - **Survival Rate by Family Size:**
                - **Alone (0 relatives):** 30.13% survival rate
                - **Small Family (1-2 relatives):** 52.98% survival rate
                - **Large Family (3+ relatives):** 33.33% survival rate
            """)

    if __name__ == '__main__':
        main()
# Function to run the app
if st.session_state.page == 'cover':
    cover_page()
else:
    main_app()
