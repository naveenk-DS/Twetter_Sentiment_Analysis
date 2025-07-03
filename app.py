import streamlit as st

st.title("Twitter Sentiment Analysis")

tweet = st.text_area("Enter a Tweet")

if st.button("Analyze"):
    # For now, just a dummy result
    st.write("Predicted Sentiment: Positive 😊")
