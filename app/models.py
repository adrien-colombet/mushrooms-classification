import streamlit as st
import convert_tables as pwc 
import seaborn as sns
import matplotlib.pyplot as plt

def app():
    # models popularity 
    st.header("Models Popularity")

    df_popularity = pwc.get_model_popularity()

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='time', y='prop', hue='name', data=df_popularity)
    plt.title('Model Popularity [source: https://paperswithcode.com/method/mobilenetv2]')

    st.pyplot(plt)

    # models performance
    st.header("Models Performance")

    df_performance = pwc.get_models_performance()

    # Get the column names except 'Model'
    columns = [col for col in df_performance.columns if col != 'Model']

    # Create a bar plot for each column
    for col in columns:
        st.bar_chart(df_performance, y = col, x='Model')