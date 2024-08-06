# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:34:43 2024

@author: hilal
"""
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.colors 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


pio.renderers.default = "browser" 

df = pd.read_csv('DataCoSupplyChainDataset.csv', encoding = 'latin-1') 
print(df.head())
print(df.columns)


data_preparation = df.groupby(['Delivery Status'])['Order Id'].count().reset_index(name = 'Number of Orders').sort_values(by = 'Number of Orders',ascending = False) 
print(data_preparation)

# Bar Plot Visualization
fig = px.bar(data_preparation, x = 'Delivery Status', y = 'Number of Orders', color = 'Number of Orders', 
             labels = {'Delivery Status':'Delivery Status','Number of Orders':'Number of Orders'})

pio.show(fig)


# Delivery status in terms of countries 
data_delivery_status_region = df.groupby(['Delivery Status','Order Region'])['Order Id'].count().reset_index()
data_delivery_status_region = data_delivery_status_region.sort_values(by = 'Order Id', ascending = False)

# Creating bar graphic to visualize order regions 
fig2 = px.bar(data_delivery_status_region, 
              x = 'Delivery Status', y = 'Order Id', 
              color = 'Order Region',
              color_discrete_sequence = plotly.colors.sequential.Viridis,
              labels = {'Delivery Status':'Delivery Status',
                        'Order Id' : 'Number of Orders',
                        'Order Region' :'Order Region'
                  })

pio.show(fig2)

# Customer Segmentation 
customer_segment_data = df.groupby('Customer Segment')['Order Id'].size().reset_index(name ='Order Count').sort_values(by= 'Order Count', ascending = False)
# Visualization 
pie_chart = px.pie(data_frame = customer_segment_data,
                   values = 'Order Count',
                   names = 'Customer Segment',
                   title = 'Order Destribution among Customer Segments',
                   width = 600 , height = 600,
                   color_discrete_sequence = plotly.colors.sequential.Inferno)
pie_chart.show()


geo_data = df.groupby(['Order Country', 'Order City'])['Order Profit Per Order'].sum().reset_index()
# Geographic Visualisation
geo_map = px.choropleth(data_frame = geo_data, 
                        locationmode = 'country names',
                        locations = 'Order Country',
                        color = 'Order Profit Per Order',
                        hover_name = 'Order Country',
                        hover_data =['Order City'],
                        color_continuous_scale = px.colors.sequential.Magma)
                        
geo_map.show()
                        
# Comparasion According to Countries
df_countries = df.groupby('Order Country')['Sales'].sum().reset_index(name = 'Sales of Orders')
df_countries = df_countries.sort_values(by = 'Sales of Orders',ascending = False).head(20)

fig = px.bar(df_countries, x = 'Sales of Orders', y = 'Order Country' , 
             color = 'Sales of Orders' ,
             color_continuous_scale = 'Viridis',
             title = 'Top 20 Countries According to Sales',
             labels = {'Sales of Orders': 'Total Sales', 'Order Country': 'Country'}, 
             text = 'Sales of Orders')

# Adding datalabels for making better readability
fig.update_traces(texttemplate = '%{text:.2s}',textposition = 'outside' )  
#Sorting according to sales amount   
fig.update_layout(yaxis = {'categoryorder': 'total ascending'},
                  xaxis_title = 'Sales Amount',
                  yaxis_title = 'Country')       
fig.show()             

# Product Analysis
df_products = df.groupby('Product Name')['Sales'].sum().reset_index(name='Sales of Orders')

# Sorting the data by 'Sales of Orders' in descending order and selecting the top 10
df_products = df_products.sort_values(by='Sales of Orders', ascending=False).head(20)

# Creating the bar plot
fig = px.bar(df_products, x='Sales of Orders', y='Product Name',
             color='Sales of Orders', color_continuous_scale='Cividis',
             title='Top 20 Products by Sales',
             labels={'Sales of Orders': 'Total Sales', 'Product Name': 'Product'},
             text='Sales of Orders')

# Adding data labels for better readability
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

# Sorting the bars by sales amount
fig.update_layout(yaxis={'categoryorder':'total ascending'}, 
                  xaxis_title='Sales (in units)',
                  yaxis_title='Product')

# Showing the plot
fig.show()



# Yearly Analysis
data_orderdate = df[['order date (DateOrders)', 'Sales']].copy()

# Converting 'order date (DateOrders)' to datetime format
data_orderdate['order_date'] = pd.to_datetime(data_orderdate['order date (DateOrders)'])

# Extracting the quarter, month, and year from the 'order_date'
data_orderdate["Quarter"] = data_orderdate['order_date'].dt.quarter
data_orderdate["Month"] = data_orderdate['order_date'].dt.month
data_orderdate["Year"] = data_orderdate['order_date'].dt.year

# Converting the 'Year' to string for grouping
data_orderdate['YearStr'] = data_orderdate['Year'].astype(str)

# Grouping by 'YearStr' and summing up 'Sales'
df_sales_year = data_orderdate.groupby('YearStr')['Sales'].sum().reset_index(name='Sales of Orders')

# Sorting the data by 'Sales of Orders' in descending order
df_sales_year = df_sales_year.sort_values(by='Sales of Orders', ascending=False)

# Creating the bar plot
fig = px.bar(df_sales_year, x='YearStr', y='Sales of Orders',
             color='Sales of Orders', color_continuous_scale='Viridis',
             title='Total Sales by Year',
             labels={'Sales of Orders': 'Total Sales', 'YearStr': 'Year'},
             text='Sales of Orders')

# Adding data labels for better readability
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

# Adjusting the layout for better presentation
fig.update_layout(xaxis_title='Year',
                  yaxis_title='Sales (in units)',
                  xaxis={'categoryorder':'total descending'},
                  plot_bgcolor='rgba(0,0,0,0)',
                  paper_bgcolor='rgba(0,0,0,0)')

# Showing the plot
fig.show()

# Data Preprocessing(30 JULY)
print('nan.counts =' , df.isna().sum() ) 


def labelencoder_method(x):
    le = LabelEncoder()
    x = le.fit_transform(x) 
    return x

df_encoded = df.apply( labelencoder_method)
print(df_encoded.head())

plt.figure(figsize = (6,4), dpi = 85) 
df_encoded.corr(method = 'pearson')['Sales'].sort_values().plot(kind = 'bar')

# Obtaining correlations for all features 
correlations = df_encoded.corr(method = 'pearson')['Sales'].sort_values()
print('correlation analysis =',correlations)
#Filtering Nan Values 
correlations = correlations.dropna()

# Average calculation for correlations
mean_correlation = correlations.mean()
threshold = abs(mean_correlation) 
print('threshold =',threshold)

#Feature Selection with help of threshold value
selected_features = correlations[abs(correlations) >= threshold]
print('selected features =', selected_features.index)

df_selected = df_encoded[selected_features.index]
print('selected df ')
print(df_selected.head())


plt.figure(figsize = (6,4), dpi = 85) 
df_selected.corr(method = 'pearson')['Sales'].sort_values().plot(kind = 'bar')

# Creating Heatmap
fig = plt.figure(figsize = (20,10))
sns.heatmap(df_selected.corr(), annot = True , fmt = '.2f', cmap = 'magma')

def tolerance_accuracy(y_true,y_pred,tolerance = 0.1):
    return np.mean(np.abs((y_true-y_pred)/y_true)<= tolerance)


# Machine Learning
df_processed = df_selected.drop(columns =['Sales'])
df_target = df_selected['Sales']
X_train,X_test,y_train,y_test = train_test_split(df_processed,df_target,test_size = 0.2, random_state = 42 )

# Logistic Regression 
logistic_model = LogisticRegression(max_iter = 1000, random_state = 42)
#Model Training
logistic_model.fit(X_train,y_train)
# Prediction
y_pred_logistic = logistic_model.predict(X_test)

# Performance Metrics
acc_logistic = tolerance_accuracy(y_test, y_pred_logistic)
mae_logistic = mean_absolute_error(y_test, y_pred_logistic)
mse_logistic = mean_squared_error(y_test, y_pred_logistic)
rmse_logistic = np.sqrt(mse_logistic)
r2_logistic = r2_score(y_test, y_pred_logistic)


print(f"Logistic Regression - MAE: {mae_logistic:.2f}, MSE: {mse_logistic:.2f}, RMSE: {rmse_logistic:.2f}, R²: {r2_logistic:.2f}, Accuracy: {acc_logistic * 100:.2f}%")

# Decision Tree 
  
tree_classifier = DecisionTreeClassifier(random_state = 42)
#Model Training
tree_classifier.fit(X_train,y_train)
# Prediction
y_pred_tree = tree_classifier.predict(X_test)
  
# Performance Metrics
acc_tree = tolerance_accuracy(y_test, y_pred_tree)
mae_tree = mean_absolute_error(y_test, y_pred_tree)
mse_tree = mean_squared_error(y_test, y_pred_tree)
rmse_tree = np.sqrt(mse_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print(f"Decision Tree - MAE: {mae_tree:.2f}, MSE: {mse_tree:.2f}, RMSE: {rmse_tree:.2f}, R²: {r2_tree:.2f}, Accuracy: {acc_tree * 100:.2f}%") 

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_continuous = linear_model.predict(X_test)


acc_linear = tolerance_accuracy(y_test, y_pred_continuous)
mae_linear = mean_absolute_error(y_test, y_pred_continuous)
mse_linear = mean_squared_error(y_test, y_pred_continuous)
rmse_linear = np.sqrt(mse_linear)
r2_linear = r2_score(y_test, y_pred_continuous)

print(f"Linear Regression - MAE: {mae_linear:.2f}, MSE: {mse_linear:.2f}, RMSE: {rmse_linear:.2f}, R²: {r2_linear:.2f}, Accuracy: {acc_linear * 100:.2f}%")

#SVM 
svm_model = SVC(kernel='linear', random_state=42, max_iter=1000)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)


acc_svm = tolerance_accuracy(y_test, y_pred_svm)
mae_svm = mean_absolute_error(y_test, y_pred_svm)
mse_svm = mean_squared_error(y_test, y_pred_svm)
rmse_svm = np.sqrt(mse_svm)
r2_svm = r2_score(y_test, y_pred_svm)

print(f"SVM - MAE: {mae_svm:.2f}, MSE: {mse_svm:.2f}, RMSE: {rmse_svm:.2f}, R²: {r2_svm:.2f}, Accuracy: {acc_svm * 100:.2f}%")
            
#KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

acc_knn = tolerance_accuracy(y_test, y_pred_knn)
mae_knn = mean_absolute_error(y_test, y_pred_knn)
mse_knn = mean_squared_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mse_knn)
r2_knn = r2_score(y_test, y_pred_knn)

print(f"KNN - MAE: {mae_knn:.2f}, MSE: {mse_knn:.2f}, RMSE: {rmse_knn:.2f}, R²: {r2_knn:.2f}, Accuracy: {acc_knn * 100:.2f}%")

# Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
#gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

acc_gb = tolerance_accuracy(y_test, y_pred_gb)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print(f"Gradient Boosting - MAE: {mae_gb:.2f}, MSE: {mse_gb:.2f}, RMSE: {rmse_gb:.2f}, R²: {r2_gb:.2f}, Accuracy: {acc_gb * 100:.2f}%")

# Explainable AI-shap values
def explainable_shapley(x,X_train,y_train,classifier_model):
    feature_names = list(x.columns)
    
    background_data_summary = shap.sample(X_train,100)
    explainer = shap.KernelExplainer(classifier_model.predict,background_data_summary)
    #Shap values
    shap_values = explainer.shap_values(X_test)
    #Plotting shap summary 
    shap.summary_plot(shap_values[0],X_test,feature_names = feature_names)
    plt.gcf().set_size_inches(10,6)
    plt.show()
    
    shap.summary_plot(shap_values[0],X_test,feature_names = feature_names,plot_type = 'violin')
    plt.gcf().set_size_inches(10,6)
    plt.show()
    
explainable_shapley(df_processed,X_train,y_train,tree_classifier) 

    
