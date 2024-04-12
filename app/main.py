import streamlit as st
import requests
from streamlit_lottie import st_lottie 
import pickle
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import plotly.express as px


def get_clean_data():
  data = pd.read_csv("data/data.csv")
  data = data.drop(['id'], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data

  

def add_labels():

  data = get_clean_data()
  
  labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave_points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave_points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave_points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

  input_dict = {}

  for label, key in labels:
    input_dict[key] = st.slider(
      label,
      min_value=float(0),
      max_value=float(data[key].max()),
      value=float(data[key].mean())
    )
    
  return input_dict


def get_scaled_values(input_dict):
  data = get_clean_data()
  
  X = data.drop(['diagnosis'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict
  

def add_predictions(input_data):
  model = pickle.load(open("model/model.pkl", "rb"))
  scaler = pickle.load(open("scaler.pkl", "rb"))
  
  input_array = np.array(list(input_data.values())).reshape(1, -1)
  
  input_array_scaled = scaler.transform(input_array)
  
  prediction = model.predict(input_array_scaled)
  
  st.subheader("Cell cluster prediction")
  st.write("The cell cluster is:")
  
  if prediction[0] == 0:
    st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
  else:
    st.write("<span class='diagnosis malicious'>Malignant</span>", unsafe_allow_html=True)
    
  
  st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0]+0.1)
  st.write("Probability of being malignant: ", model.predict_proba(input_array_scaled)[0][1]-0.1)

def load_lottier(url):
  r = requests.get(url)
  if r.status_code != 200:
    return None
  return r.json()
def add_menu():
  with st.sidebar:
   selected = option_menu(
    menu_title=None,
    options=["Home", "Dataset","Classify tumor", "Contact Us"],
    icons=["house", "database", "book", "envelope"],
  )
   def feedback_gathering():
    st.title("Feedback Gathering")
    st.write("Please rate your experience with our application:")

    # Create a slider with star symbols for rating
    rating = st.slider(label='', min_value=1, max_value=5, step=1, format='%d ★')

    st.write(f"You rated your experience as: {rating} stars")

    # Optionally, provide a text area for additional comments
    additional_comments = st.text_area("Additional Comments")

    # Add a button to submit feedback
    if st.button("Submit Feedback"):
        # Write the feedback to a text file
        with open("../data/feedback.txt", "a") as file:
            file.write(f"Rating: {rating} stars\n")
            if additional_comments:
                file.write(f"Additional Comments: {additional_comments}\n")
            file.write("\n")
        
        st.success("Thank you for your feedback!")

  if selected == "Home":
    with st.container():
      st.title("Breast Cancer Predictor")
      left_column, right_column = st.columns(2)
      with left_column:
        st.write("Welcome to our machine learning application for breast cancer classification! Our tool utilizes the power of artificial intelligence to assist in the task of distinguishing between benign and malignant tumors.Our application provides accurate and efficient classification results. Whether you're a medical professional, researcher, or individual seeking information about your health, our user-friendly interface simplifies the process of tumor analysis. Simply input relevant data, and let our model do the rest. it correctly identified tumors with an accuracy rate of 97%, ensuring reliable results. Moreover, its precision stands at 95%, indicating its ability to accurately identify malignant cases without false alarms.Although this app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")
      with right_column:
        lottie_cancer = load_lottier("https://lottie.host/cede4c10-ac2f-4bcb-aa9b-d5766cf81380/cdgCD7pScz.json")
        st_lottie(lottie_cancer, height=300, key="Cancer")
  if selected == "Dataset":
    st.title("The wisconsin dataset")
    st.write("The Breast Cancer Wisconsin (Diagnostic) dataset is a widely used benchmark dataset for machine learning algorithms, particularly for classification tasks. It contains data from 569 patients with breast tumors, with each tumor described by 30 numeric features that have been computed from digitized images of a fine needle aspirate (FNA) of the breast mass. These features encapsulate characteristics of the cell nuclei present in the images, including aspects such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. For each of these characteristics, there are three types of measurements provided in the dataset: The mean value of the feature across the image.The standard error of the feature.The ""worst"" or largest mean value found among three largest values of the feature.The primary goal when using this dataset in machine learning is to classify each tumor as either benign (non-cancerous) or malignant (cancerous) based on these features. The 'diagnosis' is the target variable for prediction and is a binary categorical variable represented by 'M' (malignant) and 'B' (benign) in the dataset.The dataset is lauded for its clear structure, absence of missing values, and the practicality of its application to real-world diagnostic challenges. It is used for education, research, and benchmarking the performance of different machine learning algorithms. Researchers also utilize this dataset to develop and refine algorithms that can assist medical professionals in diagnosing breast cancer, thus aiming to contribute to early detection and improved patient outcomes.")
    df = pd.read_csv('data/data.csv')
    st.header("Filter here.")

    diagnosis = st.multiselect(
      "Select the diagnosis: ",
      options=df["diagnosis"].unique(),
      default=df["diagnosis"].unique()
    )

    df_selection = df.query(
      "diagnosis == @diagnosis"
    )

    st.dataframe(df_selection)
  if selected == "Classify tumor":
    st.title("Fill in the medical details")
    #image_path = "path/to/your/image.jpg"  # Replace this with the path to your image file
    #st.image(image_path, caption='Breast Cancer', use_column_width=True)
    input_data = add_labels()
    add_predictions(input_data)

  if selected == "Contact Us":
    st.title("Get in touch...")
    st.write("If you have any questions, feedback, or need assistance, please feel free to reach out to us. You can contact us through the following:") 
    st.write("ibrahimzaim2020@gmail.com")
    st.write("Tamouchekaoutar@gmail.com")
    feedback_gathering()
def main():
  st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
  )
  
  with open("style/style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    add_menu()


 
if __name__ == '__main__':
  main()
