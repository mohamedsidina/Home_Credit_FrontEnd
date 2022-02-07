import streamlit as st
import pandas as pd
from downcast import reduce
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_lottie import st_lottie
import shap
import requests
import json
import numpy as np
import streamlit_authenticator as stauth

# Use the full page instead of a narrow central column
st.set_page_config(
    page_title="Home Credit Simulator", page_icon=":house:", layout="wide"
)


# API ul + endpoint
api_url = "https://oc-project-07-api.herokuapp.com/predict"  # "http://127.0.0.1:8000/predict" local api adresse

# ======================================= Part n°1 : User login Information and Interface =================================================#

# Define user information (name, username, password)
names = ["Mohamed Sidina", "Hamza"]
usernames = ["mohamed.sidina", "hamtaj19"]
passwords = ["1234", "5678"]

# Convert the plain text passwords to hashed passwords
hashed_passwords = stauth.hasher(passwords).generate()

# Create an authentication object
authenticator = stauth.authenticate(
    names,
    usernames,
    hashed_passwords,
    "some_cookie_name",
    "some_signature_key",
    cookie_expiry_days=30,
)

# Render the login module as follows
name, authentication_status = authenticator.login("Login", "main")
# Verify name and authentication status
if st.session_state["authentication_status"]:
    st.write("Welcome *%s*" % (st.session_state["name"]))

elif st.session_state["authentication_status"] == False:
    st.error("Username/password is incorrect")
    st.stop()

elif st.session_state["authentication_status"] == None:
    st.warning("Please enter your username and password")
    st.stop()

# ======================================= Part n°2 : Cach Functions =======================================================================#

# Load animations
@st.experimental_memo
def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


# Page n°1 animation : Credit Home animation
lottie_home_credit = load_lottie_file("./images/welcome-new-user.json")
# lottie_sidebar_background = load_lottie_file("https://assets7.lottiefiles.com/packages/lf20_dwcrrhkn.json")

# Page n°2 animation : credit request rejected
credit_rejected = load_lottie_file("./images/try-again.json")

# Page n°2 animation : credit request approved
credit_approved = load_lottie_file("./images/congrats.json")

# Function to load database with cleaned data and customers clusters
@st.experimental_memo
def get_cleaned_data(data_path: str):
    client_database = pd.read_csv(data_path)
    client_database = reduce(client_database)

    return client_database


# Path to client database
data_path = "./data_cleaned/application_train_cleaned_with_clusters.csv"

# Load database with cleaned data and customers clusters
with st.spinner(text="Loading Application In progress"):
    application_train = get_cleaned_data(data_path)

# Function to retunr index value
def get_customer_index(customer_data_value, client_data_base, feature, top_n):
    """
    This function retuns the index number associate to the give customer value and feature.

    Input :
        - customer_data_value : The given value of the feature for the simulated customer.
        - client_database : Home Credit trainning dataframe.
        - feature : Name of the feature choosen in a string type.
        - top_n : number of top values to be showen on the dashboard.
    """

    # Calculate the top values
    top_values = (
        client_data_base[feature]
        .value_counts(normalize=True)
        .head(top_n)
        .index.tolist()
    )

    # Find the index value
    if customer_data_value == top_values[0]:
        index = 0
    elif customer_data_value == top_values[1]:
        index = 1
    elif customer_data_value == top_values[2]:
        index = 2
    elif customer_data_value == top_values[3]:
        index = 3
    elif customer_data_value == top_values[4]:
        index = 4

    # Return the index value
    return index


# ======================================= Part n°3 : Application sidebar ==================================================================#

# Load Home Credit Logo at top corner of the sidebar
st.sidebar.image("./images/Home-Credit-logo.png", use_column_width=True)

# Create a page dropdown to navigate between Application pages
st.sidebar.title("Choose page")
page = st.sidebar.selectbox(
    "", ["Application presentation", "Predict customer's credit class", "Dashboard"]
)

# ======================================= Part n°4 : Application presentation =============================================================#

if page == "Application presentation":

    # Display  Section n°1 : Explain why application was created.
    # Create a container to hold this page components
    app_presentation = st.container()

    with app_presentation:
        st.header("Welcome to Home Credit Simulator")

        # Display more details about project on the right side of this page
        st.markdown("- **This application predicts client's repayment abilities.**")
        st.markdown(
            "- It also **gives** more **transparency** on the **AI prediction** :"
        )
        st.markdown(
            "- **An explantion** of the **prediction** made by the AI model **is shared**."
        )
        st.markdown(
            "  - **A comparison** between the **client data** vs all Home Credit Group database **is shared** given by a dashboard."
        )
        #             st.markdown("##")
        #             st.markdown("##")
        # Display Home credit animation on the left side of this page
        st_lottie(
            lottie_home_credit,
            speed=0.5,
            reverse=False,
            loop=True,
            quality="high",
            height=900,
            width=700,
        )

# ======================================= Part n°5 : Predict customer's credit class ======================================================#

elif page == "Predict customer's credit class":

    # Display Section n°2 :Load customer data, predict credit class, show local & global AI model explainability.
    # Create a container to hold this page components
    customer_class_prediction = st.container()

    with customer_class_prediction:
        st.header("Predict Customer Home Credit Class")
        # Add a seperation line
        st.markdown("------------------")
        st.markdown("**Predict client credit class :**")

        # Add sidebar button to upload customer csv file
        st.sidebar.title("Data Uploader")
        customer_data_file = st.sidebar.file_uploader(
            label="Please, use this button to load customer data",
            type=["csv"],
            accept_multiple_files=False,
            key=None,
            help=None,
            on_change=None,
            args=None,
            kwargs=None,
        )

        if st.button("Predict"):
            predict_class = True
        else:
            predict_class = False

        if st.button("Reset"):
            predict_class = False
        else:
            pass

        if predict_class == True:

            # Upload customer file and downcast it to reduce memory used.
            if customer_data_file is not None:
                customer_data = pd.read_csv(customer_data_file)
                customer_data = reduce(customer_data)
                customer_data.to_csv("data_cleaned/customer_data.csv", index=False)
                
                # If customer data has more than one row print an error message and stop application
                if customer_data.shape[0] > 1:
                    st.error(
                        "This file contains more than one customer data.\n\tPlease load an csv file with only one customer data"
                    )
                    st.stop()

                # If customer data has one row, load data and print a success message
                else:
                    with st.spinner(text="Loading customer data"):
                        time.sleep(2)
                        st.success("Customer data loaded successfully")
                        st.table(
                            customer_data[
                                [
                                    "CODE_GENDER",
                                    "DAYS_BIRTH",
                                    "AMT_INCOME_TOTAL",
                                    "AMT_CREDIT",
                                    "OCCUPATION_TYPE",
                                    "NAME_EDUCATION_TYPE",
                                    "NAME_FAMILY_STATUS",
                                    "CNT_FAM_MEMBERS",
                                ]
                            ]
                        )

            # Seperate this page into 3 columns
            left_column, middle_column, right_column = st.columns([1, 5, 1])

            # Call API to get predictions
            response = requests.post(
                url=api_url, json=customer_data.loc[customer_data.index[0]].to_dict()
            )

            # Probaility_score
            probability_score = round(response.json()["probabilty"], 2)

            # Credit Class
            credit_class = response.json()["credit_class"]

            # Transormed client data
            transformed_data = pd.DataFrame.from_dict(
                response.json()["transformed_data"]
            )
            
                        
            # Shap local explination
            base_value = np.asarray(json.loads(response.json()["shap_base_value"]))
            shap_local_values = np.asarray(json.loads(response.json()["shap_local_values"]))
        
            # Customer cluster
            customer_cluster = response.json()["client_cluster"]
            pd.DataFrame(data=pd.Series(response.json()["client_cluster"]), 
                         columns=["cluster"]
                        ).to_csv("data_cleaned/client_cluster.csv", index=False)
        
            # Display prediction probability and credit class
            with left_column:
                st.markdown("##")
                st.markdown("##")
                st.metric("Score : ", probability_score)

            with middle_column:
                if credit_class == "crédit accordé":
                    st.markdown("##")
                    st.markdown("##")
                    st.success(
                        "Your credit request is pre-approved. One of our agents will contact you shorty to move on with the process."
                    )
                else:
                    st.markdown("##")
                    st.markdown("##")
                    st.warning(
                        """
                            Sorry!! Your request has been disapproved. Please do try again when your situation evolves.\n
                            """
                    )

            with right_column:
                if credit_class == "crédit accordé":
                    st_lottie(
                        credit_approved,
                        speed=0.5,
                        reverse=False,
                        loop=True,
                        quality="high",
                        height=180,
                        width=170,
                    )

                else:
                    st_lottie(
                        credit_rejected,
                        speed=0.5,
                        reverse=False,
                        loop=True,
                        quality="high",
                        height=180,
                        width=170,
                    )

        # Add a seperation line
        st.markdown("------------------")
        st.markdown("**Local Explanation customer prediction :**")
    
        # Disable streamlit warning
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        if predict_class == True:
            
            # Plot local Explanation
            st.pyplot(
                fig=shap.force_plot(
                    base_value=base_value,
                    shap_values=shap_local_values ,
                    features=transformed_data,
                    figsize=(18, 4),
                    text_rotation=70,
                    matplotlib=True
                )
            )

        else:

            st.empty()
        

        # Add a seperation line
        st.markdown("------------------")
        st.markdown("**Global Explanation - Feature Importance :**")

        if predict_class == True:

            # Display golbal shap explainer values
            st.image("./images/global_shap.png", width=None, use_column_width=False)

        else:

            st.empty()


# ======================================= Part n°6 : Customer Dashboard ===================================================================#

elif page == "Dashboard":

    # Display Section n°3 : Dashboard to compar customer data with client database.
    # Create a container to hold this page components
    customer_dashboard = st.container()

    with customer_dashboard:

        # Add header to this section
        st.header("Customer Dashboard")

        # Load Simulated customer data
        customer_data = pd.read_csv("data_cleaned/customer_data.csv")
        
        # Load Simulated customer cluster
        customer_cluster = pd.read_csv("./data_cleaned/client_cluster.csv")
        customer_cluster = customer_cluster.loc[0, "cluster"]

        # Button to display customer data
        if st.button("Display"):
            show_client_data = True
        else:
            show_client_data = False

        # Button to reset customer data
        if st.button("Rest"):
            show_client_data = False

        # Define opacity and orientation
        opacity = 0.8
        orientation = "h"
        boxplot_fill_in_color = "#FAFAFA"

        if show_client_data == True:
            st.markdown(
                """
                    * <font color='#000000'><b>Legend  :point_right:  </b></font>
                    <font color='#0097A7'><b> ------ Simulated Customer data</b></font>
                    """,
                unsafe_allow_html=True,
            )
        else:
            pass
        # Create a figure to home the dashboard
        dashboard = make_subplots(
            rows=4,
            cols=2,
            specs=[
                [{"type": "bar"}, {"type": "box"}],
                [{"type": "box"}, {"type": "box"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "box"}],
            ],
            horizontal_spacing=0.1,
            vertical_spacing=0.1,
            column_widths=[2, 2],
            #                 row_heights=[4,4,4,4],
            subplot_titles=[
                "<b>Gender (%)</b>",
                "<b>Age (years)</b>",
                "<b> Income ($K)</b>",
                "<b> Credit Amount ($M)</b>",
                "<b> Top 5 Job title (%)</b>",
                "<b> Top 3 Education Level (%)</b>",
                "<b> Top 5 Family Status (%)</b>",
                "<b> Family members</b>",
            ],
        )

        # Calculate X and Y for Gender plot
        x_code_gender = (
            application_train.loc[
                application_train["clusters"] == customer_cluster, "CODE_GENDER"
            ]
            .value_counts(normalize=True)
            .drop("XNA")
            .index
        )
        y_code_gender = (
            application_train.loc[
                application_train["clusters"] == customer_cluster, "CODE_GENDER"
            ]
            .value_counts(normalize=True)
            .drop("XNA")
            .values.round(2)
        )

        # Simualted customer Gender data
        if show_client_data == True:
            age_index = get_customer_index(
                customer_data_value=customer_data[
                    "CODE_GENDER"
                ].values,  ###### Add Customer Simulated Data HERE ######
                client_data_base=application_train,
                feature="CODE_GENDER",
                top_n=2,
            )

            age_plot_colors = [
                boxplot_fill_in_color,
            ] * 2
            age_plot_colors[age_index] = "#0097A7"

        else:  # simulated customer data not loaded yet
            age_plot_colors = [
                "#5E35B1",
                "#3F51B5",
            ]  # "#42A5F5", "#4FC3F7", "#80DEEA"]

        # Add Gender plot to subplot (1, 1)
        dashboard.add_trace(
            go.Bar(
                x=x_code_gender,
                y=y_code_gender * 100,
                opacity=opacity,
                text=y_code_gender * 100,
                textposition="inside",
                insidetextanchor="middle",
                insidetextfont={
                    "family": "Open Sans",
                    "size": 40,
                },
                marker_color=age_plot_colors,
            ),
            row=1,
            col=1,
        )

        # Calculate X for Age plot
        customer_age_data = round(
            (
                application_train.loc[
                    application_train["clusters"] == customer_cluster, "DAYS_BIRTH"
                ]
                * -1
            )
            / 360,
            2,
        )

        # Subplot (1, 2) : Age plot
        dashboard.add_trace(
            go.Box(
                x=customer_age_data,
                fillcolor=boxplot_fill_in_color,
                line={
                    "color": "#7D3C98",
                    "width": 2,
                },
                opacity=opacity,
                orientation=orientation,
            ),
            row=1,
            col=2,
        )

        # Calculate X for Income plot
        customer_Income_data = round(
            application_train.loc[
                application_train["clusters"] == customer_cluster, "AMT_INCOME_TOTAL"
            ],
            2,
        )

        # Add Income plot to subplot (2, 1)
        dashboard.add_trace(
            go.Box(
                x=customer_Income_data,
                fillcolor=boxplot_fill_in_color,
                line={
                    "color": "#7D3C98",
                    "width": 2,
                },
                opacity=opacity,
                orientation=orientation,
            ),
            row=2,
            col=1,
        )

        # Subplot (2, 2) : Credit Amont plot
        dashboard.add_trace(
            go.Box(
                x=application_train.loc[
                    application_train["clusters"] == customer_cluster, "AMT_CREDIT"
                ],
                fillcolor=boxplot_fill_in_color,
                line={
                    "color": "#7D3C98",
                    "width": 2,
                },
                opacity=opacity,
                orientation=orientation,
            ),
            row=2,
            col=2,
        )

        # Calculate X and Y for Job plot
        x_job = (
            application_train.loc[
                application_train["clusters"] == customer_cluster, "OCCUPATION_TYPE"
            ]
            .value_counts(normalize=True)
            .head(5)
            .index
        )
        y_job = (
            application_train.loc[
                application_train["clusters"] == customer_cluster, "OCCUPATION_TYPE"
            ]
            .value_counts(normalize=True)
            .head(5)
            .values.round(2)
        )

        # Simualted customer Job data
        if show_client_data == True:
            job_index = get_customer_index(
                customer_data_value=customer_data[
                    "OCCUPATION_TYPE"
                ].values,  ###### Add Customer Simulated Data HERE ######
                client_data_base=application_train,
                feature="OCCUPATION_TYPE",
                top_n=5,
            )

            job_plot_colors = [
                boxplot_fill_in_color,
            ] * 5
            job_plot_colors[job_index] = "#0097A7"

        else:  # simulated customer data not loaded yet
            job_plot_colors = ["#5E35B1", "#3F51B5", "#42A5F5", "#4FC3F7", "#80DEEA"]

        # Subplot (3, 1) : Job plot
        dashboard.add_trace(
            go.Bar(
                x=x_job,
                y=y_job,
                opacity=opacity,
                text=y_job * 100,
                textposition="inside",
                insidetextanchor="middle",
                insidetextfont={
                    "family": "Open Sans",
                    "size": 40,
                },
                marker_color=job_plot_colors,
            ),
            row=3,
            col=1,
        )

        # Calculate X and Y for Education Level plot
        x_education = (
            application_train.loc[
                application_train["clusters"] == customer_cluster, "NAME_EDUCATION_TYPE"
            ]
            .value_counts(normalize=True)
            .head(3)
            .index
        )
        y_education = (
            application_train.loc[
                application_train["clusters"] == customer_cluster, "NAME_EDUCATION_TYPE"
            ]
            .value_counts(normalize=True)
            .head(3)
            .values.round(2)
        )

        # Simualted customer Education data
        if show_client_data == True:
            education_index = get_customer_index(
                customer_data_value=customer_data[
                    "NAME_EDUCATION_TYPE"
                ].values,  ###### Add Customer Simulated Data HERE ###
                client_data_base=application_train,
                feature="NAME_EDUCATION_TYPE",
                top_n=3,
            )

            education_plot_colors = [
                boxplot_fill_in_color,
            ] * 3
            education_plot_colors[education_index] = "#0097A7"

        else:  # simulated customer data not loaded yet
            education_plot_colors = [
                "#5E35B1",
                "#3F51B5",
                "#42A5F5",
            ]  # "#4FC3F7", "#80DEEA"]

        # Subplot (3, 2) : Education Level plot
        dashboard.add_trace(
            go.Bar(
                x=x_education,
                y=y_education,
                opacity=opacity,
                text=y_education * 100,
                textposition="inside",
                insidetextanchor="middle",
                insidetextfont={
                    "family": "Open Sans",
                    "size": 40,
                },
                marker_color=education_plot_colors,
            ),
            row=3,
            col=2,
        )

        # Calculate X and Y for Family Status plot
        x_family_status = (
            application_train.loc[
                application_train["clusters"] == customer_cluster, "NAME_FAMILY_STATUS"
            ]
            .value_counts(normalize=True)
            .head(5)
            .index
        )
        y_family_status = (
            application_train.loc[
                application_train["clusters"] == customer_cluster, "NAME_FAMILY_STATUS"
            ]
            .value_counts(normalize=True)
            .head(5)
            .values.round(2)
        )

        # Simualted customer Family Status data
        if show_client_data == True:
            status_index = get_customer_index(
                customer_data_value=customer_data[
                    "NAME_FAMILY_STATUS"
                ].values,  ###### Add Customer Simulated Data HERE ######
                client_data_base=application_train,
                feature="NAME_FAMILY_STATUS",
                top_n=5,
            )

            status_plot_colors = [
                boxplot_fill_in_color,
            ] * 5
            status_plot_colors[status_index] = "#0097A7"

        else:  # simulated customer data not loaded yet
            status_plot_colors = ["#5E35B1", "#3F51B5", "#42A5F5", "#4FC3F7", "#80DEEA"]

        # Subplot (4, 1) : Family Status plot
        dashboard.add_trace(
            go.Bar(
                x=x_family_status,
                y=y_family_status,
                opacity=opacity,
                text=y_family_status * 100,
                textposition="inside",
                insidetextanchor="middle",
                insidetextfont={
                    "family": "Open Sans",
                    "size": 40,
                },
                marker_color=status_plot_colors,
            ),
            row=4,
            col=1,
        )

        # Calculate X and Y for Family members plot
        x_family_members = (
            application_train.loc[
                application_train["clusters"] == customer_cluster, "CNT_FAM_MEMBERS"
            ]
            .astype("float32")
            .head(100)
        )

        # subplot (4, 2) : Family Members plot
        dashboard.add_trace(
            go.Box(
                x=x_family_members,
                fillcolor=boxplot_fill_in_color,
                line={
                    "color": "#7D3C98",
                    "width": 2,
                },
                opacity=opacity,
                orientation=orientation,
            ),
            row=4,
            col=2,
        )

        # Customize dashboard
        dashboard.update_layout(
            title_font_family="sans serif",
            title_font_color="#17202A",
            title_font_size=24,
            plot_bgcolor="#FFFFFF",
            xaxis_rangeselector_font_size=1,
            yaxis_showticklabels=False,  # Gender plot
            yaxis2_showticklabels=False,  # Age plot
            yaxis3_showticklabels=False,  # Income plot
            yaxis4_showticklabels=False,  # Credit Amount plot
            yaxis5_showticklabels=False,  # Job title plot
            yaxis6_showticklabels=False,  # Education Level plot
            yaxis7_showticklabels=False,  # Family Status plot
            yaxis8_showticklabels=False,  # Family Members plot
            showlegend=False,
            width=900,
            height=2000,
        )

        # Fix Subplots titles positions
        dashboard.layout.annotations[0].update(x=0.07, font_size=25)
        dashboard.layout.annotations[1].update(x=0.65, font_size=25)
        dashboard.layout.annotations[2].update(x=0.07, font_size=25)
        dashboard.layout.annotations[3].update(x=0.65, font_size=25)
        dashboard.layout.annotations[4].update(x=0.07, font_size=25)
        dashboard.layout.annotations[5].update(x=0.67, font_size=25)
        dashboard.layout.annotations[6].update(x=0.07, font_size=25)
        dashboard.layout.annotations[7].update(x=0.65, font_size=25)
        dashboard.update_xaxes(
            tickfont_size=18, ticks="outside", ticklen=6, tickwidth=1
        )

        # subplot (2, 1) : Add simulated customer data to Age plot
        if show_client_data == True:
            x2 = round(
                (customer_data.loc[0, "DAYS_BIRTH"] * -1) / 360, 2
            )  ########## Add Customer Simulated Data HERE ######

            # Simulated customer data
            dashboard.add_shape(
                type="line",
                x0=x2,  # customer data here
                y0=-0.4,
                x1=x2,  # customer data here
                y1=0.4,
                line=dict(
                    color="#0097A7",
                    width=4,
                    dash="dot",
                ),
                row=1,
                col=2,
            )

            # subplot (2, 1) : Add simulated customer data to Income plot
            x3 = round(
                customer_data.loc[0, "AMT_INCOME_TOTAL"], 2
            )  ########## Add Customer Simulated Data HERE ##########

            # Simulated customer data
            dashboard.add_shape(
                type="line",
                x0=x3,  # customer data here
                y0=-0.4,
                x1=x3,  # customer data here
                y1=0.4,
                line=dict(
                    color="#0097A7",
                    width=4,
                    dash="dot",
                ),
                row=2,
                col=1,
            )

            # subplot (2, 2) : Add simulated customer data to Credit Amont plot
            x4 = round(
                customer_data.loc[0, "AMT_CREDIT"], 2
            )  ########## Add Customer Simulated Data HERE ##########

            # Simulated customer data
            dashboard.add_shape(
                type="line",
                x0=x4,  # customer data here
                y0=-0.4,
                x1=x4,  # customer data here
                y1=0.4,
                line=dict(
                    color="#0097A7",
                    width=4,
                    dash="dot",
                ),
                row=2,
                col=2,
            )

            # subplot (4, 2) : Add simulated customer data to Credit Family members plot
            x10 = round(
                customer_data.loc[0, "CNT_FAM_MEMBERS"], 2
            )  ########## Add Customer Simulated Data HERE ##########

            # Simulated customer data
            dashboard.add_shape(
                type="line",
                x0=x10,  # customer data here
                y0=-0.4,
                x1=x10,  # customer data here
                y1=0.4,
                line=dict(
                    color="#0097A7",
                    width=4,
                    dash="dot",
                ),
                row=4,
                col=2,
            )

        else:  # simulated customer data not loaded yet
            pass

        # Display family members plot
        st.plotly_chart(dashboard, use_container_width=True)


# Customize Side bar
# Add information about Project
st.sidebar.title("Contribute")
st.sidebar.info(
    "This an open source project, that I developed during my training at OpenClassRooms.\n\t"
    "If you have any comments or questions, please feel free to reach me "
    "[here's my github](https://github.com/mohamedsidina/OC_Project_07/issues) and "
    "[and my email](mohamed.sidina@gmail.com) "
)

# Add my contact information.
st.sidebar.title("About")
st.sidebar.info(
    """
    This app is maintained by **Mohamed Sidina**.\n\tYou can learn more about me on my [linkedin profile](https://www.linkedin.com/in/mohamed-sidina-sid-ahmed-6701935a/)
    """
)

# Customize Streamlit style
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)