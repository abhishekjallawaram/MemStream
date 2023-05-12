pip install streamlit pandas

import streamlit as st
import pandas as pd

data = {'Method': ['PCA', 'PCA+Sc', 'FastICA', 'FastICA+Sc', 'LDA', 'LDA+Sc', 'Chi2', 'MI', 'RFE'],
        'Random Sampling': [0.104644872, 0.988277245, 0.984321238, 0.984262302, 0.978304947, 0.978304947, 0.839524973, 0.978304947, 0.983606268],
        'Stratified Sampling': [0.099560532, 0.989425408, 0.983456234, 0.982117087, 0.968949707, 0.968949707, 0.859038462, 0.968949707, 0.976288351],
        'Random Oversampling': [0.097062918, 0.982158772, 0.982230018, 0.983828316, 0.976868768, 0.976868768, 0.838996559, 0.976864848, 0.981054397],
        'SMOTE': [0.118527371, 0.871165095, 0.984827737, 0.983733281, 0.976864848, 0.976864848, 0.83907696, 0.976868768, 0.980318761]}

df = pd.DataFrame(data)
df = df.set_index('Method')

import matplotlib.pyplot as plt

def plot_bar(method):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df.columns, df.loc[method], color=['#0059b3','#47c1bf','#a3f7bf','#f2efef'])
    ax.set_title(method)
    ax.set_ylabel('AUROC')
    plt.ylim(0.0, 1.0)
    st.pyplot(fig)

def app():
    st.title('KDD Sample Data')
    st.write('Data for different methods')
    
    # Create a dropdown to select the method
    method = st.selectbox('Select Method', df.index)
    
    # Plot the bar plot for the selected method
    plot_bar(method)

if __name__ == '__main__':
    app()
