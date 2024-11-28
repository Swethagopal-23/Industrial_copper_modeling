# importing libraries
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pickle 

# Title and introduction
st.title("Industrial Copper Modelling")

# Sidebar - Navigation
st.sidebar.title("Copper Price Prediction Tool")
menu = st.sidebar.radio("Navigation", ["Home", "Get Prediction", "About"])

# Home Section
if menu == "Home":
    st.title("Copper Price and Order Status Prediction")
    st.write("""
        This application helps users predict:
        - Selling price of copper products based on various features.
        - Order status (Won/Lost) based on price and other details.
    """)
    st.image("https://i.pinimg.com/736x/f2/c3/5f/f2c35fb6902a06bc220a65505176ebaa.jpg", use_column_width=True)  

# Prediction Section
elif menu == "Get Prediction":

    st.title("Make Predictions")

    # Select Prediction Type
    prediction_type = st.radio("Choose what to predict", ["Selling Price", "Order Status"])


    #user input values 
    class option():
    
        country_values=[ 25.,  26.,  27.,  28.,  30.,  32.,  38.,  39.,  40.,  77.,  78., 79.,  80.,  84.,  89., 107., 113.]

        status_values=['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM','Wonderful', 'Revised',
                'Offered', 'Offerable']

        status_encoded = {'Lost':0, 'Won':1, 'Draft':2, 'To be approved':3, 'Not lost for AM':4,'Wonderful':5, 'Revised':6,
                        'Offered':7, 'Offerable':8}
        
        item_type_values=['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']

        item_type_encoded = {'W':5.0, 'WI':6.0, 'S':3.0, 'Others':1.0, 'PL':2.0, 'IPL':0.0, 'SLAWR':4.0}

        application_values=[2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 27.0, 28.0, 29.0, 38.0, 39.0, 40.0,
                    41.0, 42.0, 56.0, 58.0, 59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]
        
        product_ref_values=[611728, 611733, 611993, 628112, 628117, 628377, 640400, 640405, 640665, 164141591, 164336407,
                    164337175, 929423819, 1282007633, 1332077137, 1665572032, 1665572374, 1665584320, 1665584642,
                    1665584662, 1668701376, 1668701698, 1668701718, 1668701725, 1670798778, 1671863738, 1671876026,
                    1690738206, 1690738219, 1693867550, 1693867563, 1721130331, 1722207579]


    if prediction_type == 'Selling Price':
        st.markdown("<h5 style=color:grey>To predict the selling price of copper, please provide the following information:",unsafe_allow_html=True)
        st.write('')

        # form to get the user input 
        with st.form('prediction'):
            col1,col2=st.columns(2)
            with col1:

                item_date=st.date_input(label='Item Date',format='DD/MM/YYYY')

                country=st.selectbox(label='Country',options=option.country_values,index=None)

                item_type=st.selectbox(label='Item Type',options=option.item_type_values,index=None)

                application=st.selectbox(label='Application',options=option.application_values,index=None)

                product_ref=st.selectbox(label='Product Ref',options=option.product_ref_values,index=None)

                customer=st.number_input('Customer ID',min_value=10000)

            with col2:

                delivery_date=st.date_input(label='Delivery Date',format='DD/MM/YYYY')

                status=st.selectbox(label='Status',options=option.status_values,index=None)

                quantity=st.number_input(label='Quantity',min_value=0.1)

                width=st.number_input(label='Width',min_value=1.0)

                thickness=st.number_input(label='Thickness',min_value=0.1)

                st.markdown('<br>', unsafe_allow_html=True)
                
                button=st.form_submit_button('CLICK TO GET PREDICTION 📊',use_container_width=True)

        if button:
           
            if not all([item_date, delivery_date, country, item_type, application, product_ref,
                        customer, status, quantity, width, thickness]):
                st.error("Please fill the required fields.")

            else:
                
                # pickle model 
                with open('D:\Industrial_Copper_Modeling\Regression_model.pkl','rb') as files:
                    predict_model=pickle.load(files)

                # customize the user data to fit the feature 
                status=option.status_encoded[status]
                item_type=option.item_type_encoded[item_type]

                delivery_time_taken=abs((item_date - delivery_date).days)

                quantity_log=np.log(quantity)
                thickness_log=np.log(thickness)

                #predict the selling price with regressor model
                user_data=np.array([[customer, country, status, item_type ,application, width, product_ref,
                                    delivery_time_taken, quantity_log, thickness_log ]])
                
                pred=predict_model.predict(user_data)

                selling_price=np.exp(pred[0])

                #display the predicted selling price 
                st.subheader(f":green[Selling Price 💸 :] {selling_price:.2f}") 
                

    if prediction_type == 'Order Status':
        st.markdown("<h5 style=color:grey;>To predict the status of copper, please provide the following information:",unsafe_allow_html=True)
        st.write('')

        #form to get the user input 
        with st.form('classifier'):
            col1,col2=st.columns(2)
            with col1:

                item_date=st.date_input(label='Item Date',format='DD/MM/YYYY')

                country=st.selectbox(label='Country',options=option.country_values,index=None)

                item_type=st.selectbox(label='Item Type',options=option.item_type_values,index=None)

                application=st.selectbox(label='Application',options=option.application_values,index=None)

                product_ref=st.selectbox(label='Product Ref',options=option.product_ref_values,index=None)

                customer=st.number_input('Customer ID',min_value=10000)

            with col2:

                delivery_date=st.date_input(label='Delivery Date',format='DD/MM/YYYY')

                quantity=st.number_input(label='Quantity',min_value=0.1)

                width=st.number_input(label='Width',min_value=1.0)

                thickness=st.number_input(label='Thickness',min_value=0.1)

                selling_price=st.number_input(label='Selling Price',min_value=0.1)

                st.markdown('<br>', unsafe_allow_html=True)
                
                button=st.form_submit_button('CLICK TO GET PREDICTION 📊',use_container_width=True)

        if button:
            #check whether user fill all required fields
            if not all([item_date, delivery_date, country, item_type, application, product_ref,
                        customer,quantity, width, thickness,selling_price]):
                st.error("Please fill the required fields.")

            else:
                #pickle model
                with open('D:\Industrial_Copper_Modeling\Classifier_model.pkl','rb') as files:
                    model=pickle.load(files)

                # customize the user data to fit the feature 
                item_type=option.item_type_encoded[item_type]

                delivery_time_taken=abs((item_date - delivery_date).days)

                quantity_log=np.log(quantity)
                thickness_log=np.log(thickness)
                selling_price_log=np.log(selling_price)

                #predict the status with classifier model
                user_data=np.array([[customer, country, item_type ,application, width, product_ref,
                                    delivery_time_taken, quantity_log, thickness_log, selling_price_log ]])
                
                status=model.predict(user_data)

                #display the predicted status 
                if status==1:
                    st.subheader(f"Status of the copper :Won 👍🏼")
                else:
                    st.subheader(f"Status of the copper : Lost 👎🏼")

# About Section
elif menu == "About":
    st.title("About")
    

    # Main description section with matching color
    st.markdown("""
        <div style="background-color: #1C1C1C; padding: 15px; border-radius: 8px;">
            <p style="font-family: Helvetica, sans-serif; color: #DAF7A6; font-size: 16px; line-height: 1.6;">
                This application is designed to assist businesses in the copper industry 
                by providing insights into pricing and order likelihood. It uses machine 
                learning models trained on historical data.
            </p>
            <p style="font-family: Helvetica, sans-serif; color: #DAF7A6; font-size: 16px; line-height: 1.6;">
                With tools for data exploration, predictive modeling, and real-time analysis, 
                this platform serves as a valuable resource for decision-makers.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Developer credit section with the same color
    st.markdown("""
        <div style="background-color: #1C1C1C; padding: 10px; border-radius: 8px; margin-top: 20px;">
            <p style="font-family: Helvetica, sans-serif; color: #DAF7A6; text-align: center; font-size: 14px; margin: 0;">
                <b>Developer:</b> Swetha G
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Domain and Skills Section (Using Markdown with colors matching the suggestions)
    st.markdown("<h4 style='color: #FFC300;'>Domain:</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-family: Helvetica, sans-serif; color: #DAF7A6;'>Manufacturing</p>", unsafe_allow_html=True)

    st.markdown("<h4 style='color: #FFC300;'>Skills & Technologies:</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-family: Helvetica, sans-serif; color: #DAF7A6;'>Python, Data Preprocessing, EDA, Streamlit, Machine Learning</p>", unsafe_allow_html=True)

    st.markdown("<h4 style='color: #FFC300;'>Overview:</h4>", unsafe_allow_html=True)

    # Data Understanding Section
    st.markdown("<h5 style='color: #FF5733;'>Data Understanding</h5>", unsafe_allow_html=True)
    st.markdown("<p style='font-family: Helvetica, sans-serif; color: #DAF7A6;'>- Identifying variable types (continuous, categorical) and their distributions. Handling missing values.</p>", unsafe_allow_html=True)

    # Data Preprocessing Section
    st.markdown("<h5 style='color: #FF5733;'>Data Preprocessing</h5>", unsafe_allow_html=True)
    st.markdown("<ul style='font-family: Helvetica, sans-serif; color: #DAF7A6;'>"
                "<li>Handle missing values.</li>"
                "<li>Treat outliers with IQR or Isolation Forest.</li>"
                "<li>Address skewness using log/Box-Cox transformations.</li>"
                "<li>Encode categorical variables.</li></ul>", unsafe_allow_html=True)

    # EDA Section
    st.markdown("<h5 style='color: #FF5733;'>Exploratory Data Analysis (EDA)</h5>", unsafe_allow_html=True)
    st.markdown("<ul style='font-family: Helvetica, sans-serif; color: #DAF7A6;'>"
                "<li>Visualize outliers and skewness with Seaborn.</li>"
                "</ul>", unsafe_allow_html=True)

    # Feature Engineering Section
    st.markdown("<h5 style='color: #FF5733;'>Feature Engineering</h5>", unsafe_allow_html=True)
    st.markdown("<ul style='font-family: Helvetica, sans-serif; color: #DAF7A6;'>"
                "<li>Create new features or transform existing ones.</li>"
                "<li>Use heatmaps to identify and drop highly correlated features.</li>"
                "</ul>", unsafe_allow_html=True)

    # Model Building Section
    st.markdown("<h5 style='color: #FF5733;'>Model Building & Evaluation</h5>", unsafe_allow_html=True)
    st.markdown("<ul style='font-family: Helvetica, sans-serif; color: #DAF7A6;'>"
                "<li>Train models with tree-based classifiers (e.g., ExtraTrees, XGBoost).</li>"
                "<li>Optimize models with cross-validation and grid search.</li>"
                "<li>Evaluate using accuracy, precision, recall, F1 score, and AUC.</li>"
                "<li>Same approach for regression models.</li>"
                "</ul>", unsafe_allow_html=True)

    # GUI Section
    st.markdown("<h5 style='color: #FF5733;'>Model GUI</h5>", unsafe_allow_html=True)
    st.markdown("<ul style='font-family: Helvetica, sans-serif; color: #DAF7A6;'>"
                "<li>Created an interactive interface using Streamlit.</li>"
                "<li>Input fields for all necessary columns except the target variable (Selling_Price for regression, Status for classification).</li>"
                "<li>Apply same preprocessing and feature transformations to predict outcomes.</li>"
                "</ul>", unsafe_allow_html=True)

    # Divider
    st.markdown('<hr style="border: 2px solid #FFC300;">', unsafe_allow_html=True)

