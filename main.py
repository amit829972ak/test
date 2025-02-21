import streamlit as st
from utils import extract_travel_info, format_travelers

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Travel Plan Extractor",
        page_icon="✈️",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stTextInput > div > div > input {
            padding: 15px;
        }
        .info-box {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("✈️ Travel Plan Information Extractor")
    
    # Instructions
    st.markdown("""
    ### How to Use
    Enter your travel plans in natural language. For example:
    > "I am planning a round trip from New York to Paris from December 15 to December 25, 2024. 
    The trip lasts for 10 days. There will be two adults, 1 kid and 1 infant. 
    We will travel by flight and stay in hotel. Our budget is $5000"
    """)

    # Input section
    travel_text = st.text_area(
        "Enter your travel plans:",
        height=150,
        placeholder="Describe your travel plans here..."
    )

    # Process button
    if st.button("Extract Travel Information", type="primary"):
        if not travel_text:
            st.error("Please enter your travel plans to proceed.")
        else:
            try:
                # Extract information
                travel_info = extract_travel_info(travel_text)
                
                # Display results in a structured format
                st.markdown("### 📋 Extracted Travel Information")
                
                # Create two columns for better layout
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.container():
                        st.markdown("#### 📍 Location Details")
                        st.markdown(f"""
                        - **From:** {travel_info['Starting Location']}
                        - **To:** {travel_info['Destination']}
                        """)
                    
                    st.markdown("#### 📅 Date Information")
                    st.markdown(f"""
                    - **Start Date:** {travel_info['Start Date']}
                    - **End Date:** {travel_info['End Date']}
                    - **Duration:** {travel_info['Trip Duration']}
                    - **Trip Type:** {travel_info['Trip Type']}
                    """)

                with col2:
                    st.markdown("#### 👥 Travelers")
                    st.markdown(f"- **Total Travelers:** {format_travelers(travel_info['Number of Travelers'])}")
                    
                    st.markdown("#### 💰 Budget & Preferences")
                    st.markdown(f"""
                    - **Budget:** {travel_info['Budget Range']}
                    - **Transportation:** {travel_info['Transportation Preferences']}
                    - **Accommodation:** {travel_info['Accommodation Preferences']}
                    """)
                
                if travel_info['Special Requirements'] != "None specified":
                    st.markdown("#### ℹ️ Special Requirements")
                    st.markdown(f"- {travel_info['Special Requirements']}")

            except Exception as e:
                st.error(f"An error occurred while processing your request: {str(e)}")
                st.markdown("Please check your input and try again.")

    # Footer
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Be specific about dates, locations, and number of travelers
    - Include budget information if available
    - Mention transportation and accommodation preferences
    - Add any special requirements or considerations
    """)

if __name__ == "__main__":
    main()
