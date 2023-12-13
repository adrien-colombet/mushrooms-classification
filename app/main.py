import streamlit as st
import dataset_crop, edible_order_predict

PAGES = {
    "Dataset preparation": dataset_crop,
    "Predictions": edible_order_predict
}

def main():
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page.app()

if __name__ == "__main__":
    main()
