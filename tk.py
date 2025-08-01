import streamlit as st
import spacy
import dateparser
import re
import pandas as pd
from dateparser import parse
from datetime import datetime, timedelta
from dateparser.search import search_dates
import geonamescache
from word2number import w2n
import json
import google.generativeai as genai
import traceback

# Configure the Streamlit page
st.set_page_config(
    page_title="Travel Planner Pro",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        /* Main container */
        .main {
            padding: 2rem;
        }
        
        /* Headers */
        h1 {
            color: #1E88E5;
            font-size: 3rem !important;
            text-align: center;
            margin-bottom: 2rem !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        h2 {
            color: #2196F3;
            margin-top: 2rem !important;
        }
        
        h3 {
            color: #42A5F5;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: #F8F9FA;
            padding: 0.5rem;
            border-radius: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 0 20px;
            background-color: #FFFFFF;
            border-radius: 5px;
            color: #1E88E5;
            font-weight: 600;
        }
        
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            background-color: #1E88E5;
            color: white;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #F8F9FA;
            border-radius: 10px;
            padding: 0.5rem;
        }
        
        /* Button styling */
        .stButton>button {
            width: 100%;
            height: 50px;
            background-color: #1E88E5;
            color: white;
            font-weight: 600;
            border-radius: 10px;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #1976D2;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Text area styling */
        .stTextArea textarea {
            border-radius: 10px;
            border: 2px solid #E3F2FD;
            padding: 1rem;
        }
        
        .stTextArea textarea:focus {
            border-color: #1E88E5;
            box-shadow: 0 0 0 2px rgba(30,136,229,0.2);
        }
        
        /* Card-like containers */
        .css-1r6slb0 {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        /* Download buttons */
        .stDownloadButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            padding: 0.5rem;
            border-radius: 5px;
            border: none;
            margin-top: 1rem;
        }
        
        /* Tips section */
        .tip-box {
            background-color: #E3F2FD;
            padding: 1rem;
            border-radius: 10px;
            margin-top: 2rem;
        }
        
        /* Icons and emojis */
        .icon-text {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Table styling */
        .dataframe {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .dataframe th {
            background-color: #1E88E5;
            color: white;
            padding: 1rem !important;
        }
        
        .dataframe td {
            padding: 0.75rem !important;
        }
        
        /* Progress spinner */
        .stSpinner {
            text-align: center;
            color: #1E88E5;
        }
    </style>
""", unsafe_allow_html=True)

def setup_gemini():
    import os
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "default_key")
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel('gemini-1.5-flash')

# Load spaCy model globally
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy English model not found. Please install it using: python -m spacy download en_core_web_trf")
        st.stop()

nlp = load_spacy_model()

# Load city database from geonamescache
gc = geonamescache.GeonamesCache()
cities_dict = {city["name"].lower(): city["name"] for city in gc.get_cities().values()}

# Define seasonal mappings
seasonal_mappings = {
  "summer": "06-01",
    "mid summer": "07-15",
    "end of summer": "08-25",
    "autumn": "09-15",
    "fall": "09-15",
    "monsoon": "09-10",
    "winter": "12-01",
    "early winter": "11-15",
    "late winter": "01-15",
    "spring": "04-01"  
}

def extract_details(text):
    doc = nlp(text)
    text_lower = text.lower()
    details = {
        "Starting Location": None,
        "Destination": None,
        "Start Date": None,
        "End Date": None,
        "Trip Duration": None,
        "Trip Type": None,
        "Number of Travelers": None,
        "Budget Range": None,
        "Transportation Preferences": None,
        "Accommodation Preferences": None,
        "Special Requirements": None
    }
    # Extract locations
    locations = [ent.text for ent in doc.ents if ent.label_ in {"GPE", "LOC"}]    
    common_destinations = {"goa","Goa","French countryside","goa","Maldives", "Bali", "Paris", "New York", "Los Angeles", "San Francisco", "Tokyo", "London", "Dubai", "Rome", "Bangkok"}
    # Backup regex-based location extraction
    regex_matches = re.findall(r'\b(?:from|to|visit|traveling to|heading to|going to|in|at|of|to the|toward the)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)', text)
    # Check for cities in text using geonamescache
    extracted_cities = []
    words = text.split()
    for i in range(len(words)):
        for j in range(i + 1, min(i + 4, len(words))):  # Check up to 3-word phrases
            phrase = " ".join(words[i:j+1])
            if phrase.lower() in cities_dict or phrase in common_destinations:
                extracted_cities.append(cities_dict[phrase.lower()])

    # Combine all sources and remove duplicates while preserving order
    seen = set()
    all_locations = [loc for loc in locations + regex_matches + extracted_cities if not (loc in seen or seen.add(loc))]

    # Determine starting location and destination using dependency parsing
    start_location, destination = None, None
    for token in doc:
        if token.text.lower() == "from":
            location = " ".join(w.text for w in token.subtree if w.ent_type_ in {"GPE", "LOC"})
            if location:
                start_location = location
        elif token.text.lower() in {"to", "toward"}:
            location = " ".join(w.text for w in token.subtree if w.ent_type_ in {"GPE", "LOC"})
            if location:
                destination = location

    # If dependency parsing fails, use list extraction
    if not start_location and not destination:
        if len(all_locations) > 1:
            start_location, destination = all_locations[:2]
        elif len(all_locations) == 1:
            destination = all_locations[0]

    # Construct final details dictionary
    details = {}
    if start_location:
        details["Starting Location"] = start_location
    if destination:
        details["Destination"] = destination

    # Enhanced destination extraction for complex travel patterns
    def extract_advanced_destinations(text, current_start, current_dest):
        """Handle complex patterns like 'from india to china, to japan from nepal' or 'to thailand'"""
        
        # Pattern 1: Complex multi-destination "from X to Y, to Z from W"
        complex_pattern = r'from\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)(?:,\s*to\s+([a-zA-Z\s]+?))?(?:\s+from\s+([a-zA-Z\s]+?))?'
        complex_match = re.search(complex_pattern, text, re.IGNORECASE)
        
        if complex_match:
            groups = complex_match.groups()
            start_loc = groups[0].strip().title() if groups[0] else current_start
            
            destinations = []
            for i in range(1, len(groups)):
                if groups[i]:
                    dest = groups[i].strip().title()
                    if dest and dest not in destinations:
                        destinations.append(dest)
            
            return start_loc, ", ".join(destinations) if destinations else current_dest
        
        # Pattern 2: Simple "to X" with multiple destinations
        to_matches = re.findall(r'\bto\s+([a-zA-Z\s]+?)(?:\s|$|,)', text, re.IGNORECASE)
        if to_matches:
            destinations = []
            for match in to_matches:
                cleaned = match.strip().title()
                # Remove common words that aren't locations
                cleaned = re.sub(r'\b(And|Or|The|A|An|For|Days?|Weeks?|Months?)\b', '', cleaned).strip()
                if len(cleaned) > 2 and cleaned not in destinations:
                    destinations.append(cleaned)
            if destinations:
                return current_start, ", ".join(destinations)
        
        # Pattern 3: Visit/traveling patterns
        visit_pattern = r'(?:visit|traveling to|going to)\s+([a-zA-Z\s,]+?)(?:\s+for|\s+in|\.|$)'
        visit_match = re.search(visit_pattern, text, re.IGNORECASE)
        if visit_match:
            locations_str = visit_match.group(1)
            destinations = []
            for loc in locations_str.split(','):
                cleaned = loc.strip().title()
                if len(cleaned) > 2:
                    destinations.append(cleaned)
            if destinations:
                return current_start, ", ".join(destinations)
        
        return current_start, current_dest
    
    # Apply enhanced extraction
    start_location, destination = extract_advanced_destinations(text, start_location, destination)
    
    # Update details dictionary with enhanced results
    if start_location:
        details["Starting Location"] = start_location
    if destination:
        details["Destination"] = destination

    # Extract duration
    duration_match = re.search(r'(?P<value>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[-]?\s*(?P<unit>day|days|night|nights|week|weeks|month|months)', text, re.IGNORECASE)
    duration_days = None

    if duration_match:
        unit = duration_match.group("unit").lower()
        value = duration_match.group("value").lower()

        # Convert word-based numbers to digits
        try:
            value = int(value) if value.isdigit() else w2n.word_to_num(value)
        except ValueError:
            value = 1  # Default to 1 if conversion fails
        if "week" in unit:
            duration_days = value * 7   
        elif "month" in unit:
            duration_days = value * 30
        else:
            duration_days = value
        details["Trip Duration"] = f"{duration_days} days"
    else:
        # Handle cases where the duration is mentioned without a number
        if "week" in text:
            duration_days = 7
        elif "month" in text:
            duration_days = 30
        elif "day" in text or "night" in text:
            duration_days = 1
        
        if duration_days:
            details["Trip Duration"] = f"{duration_days} days"        
    
    # Extract dates
    text_lower = text.lower()
    
    # NEW PATTERN: Handle date ranges with format "5-12th june"
    text_lower = text.lower()
    
    # Create patterns for different date formats
    
    # Pattern 1: Handle date ranges with format "from 3-13th april 2025"
    date_range_ordinal_pattern = r'from\s+(\d{1,2})(?:st|nd|rd|th)?-(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)(?:\s+(\d{4}))?'
    ordinal_match = re.search(date_range_ordinal_pattern, text, re.IGNORECASE)
    
    # Pattern 2: Handle formats like "from 22th june 2025 to 29th june 2025"
    date_to_date_pattern = r'from\s+(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)(?:\s+(\d{4}))?\s+to\s+(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)(?:\s+(\d{4}))?'
    to_date_match = re.search(date_to_date_pattern, text, re.IGNORECASE)
    
    # Pattern 3: Handle formats like "from 02-04-2025 to 29-04-2025"
    numeric_date_pattern = r'from\s+(\d{1,2})-(\d{1,2})-(\d{4})\s+to\s+(\d{1,2})-(\d{1,2})-(\d{4})'
    numeric_match = re.search(numeric_date_pattern, text, re.IGNORECASE)
    
    # Pattern 4: Handle formats like "from 12th march for two week"
    date_for_duration_pattern = r'from\s+(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)(?:\s+(\d{4}))?\s+for\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|days|week|weeks|month|months)'
    date_for_duration_match = re.search(date_for_duration_pattern, text, re.IGNORECASE)
    
    # Pattern 5: Handle formats like "for a week from 13th april"
    duration_from_date_pattern = r'for\s+(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|days|week|weeks|month|months)\s+from\s+(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)(?:\s+(\d{4}))?'
    duration_from_date_match = re.search(duration_from_date_pattern, text, re.IGNORECASE)
    
    # Pattern 6: Handle formats like "for two weeks on 3rd april"
    duration_on_date_pattern = r'for\s+(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|days|week|weeks|month|months)\s+on\s+(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)(?:\s+(\d{4}))?'
    duration_on_date_match = re.search(duration_on_date_pattern, text, re.IGNORECASE)
    
    # Pattern 7: Handle formats like "on 13th march for a week"
    on_date_for_duration_pattern = r'on\s+(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)(?:\s+(\d{4}))?\s+for\s+(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|days|week|weeks|month|months)'
    on_date_for_duration_match = re.search(on_date_for_duration_pattern, text, re.IGNORECASE)
    
    # Pattern 8: Handle formats like "for 2 weeks on 20/05/2025" or "for two weeks on 02-08-2025"
    duration_on_numeric_date_pattern = r'for\s+(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|days|week|weeks|month|months)\s+on\s+(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})'
    duration_on_numeric_date_match = re.search(duration_on_numeric_date_pattern, text, re.IGNORECASE)
    
    # Pattern 9: Handle formats like "on 05/06/2025 for two weeks" or "on 06-07-2025 for 2 weeks"
    on_numeric_date_for_duration_pattern = r'on\s+(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\s+for\s+(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|days|week|weeks|month|months)'
    on_numeric_date_for_duration_match = re.search(on_numeric_date_for_duration_pattern, text, re.IGNORECASE)
    
    # Function to convert text numbers to integers
    def convert_text_to_number(text_num):
        if text_num.lower() in ['a', 'an']:
            return 1
        try:
            return int(text_num)
        except ValueError:
            # Convert word numbers to digits
            word_to_num = {
                'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
            }
            return word_to_num.get(text_num.lower(), 1)
    
    # Function to convert unit to days
    def convert_unit_to_days(num, unit):
        if 'week' in unit:
            return num * 7
        elif 'month' in unit:
            return num * 30
        else:
            return num
    
    # Extract and process dates based on different patterns
    start_date = None
    end_date = None
    duration_value = None
    
    current_year = datetime.now().year
    
    # Try to match each pattern
    if ordinal_match:
        # Pattern 1: "from 3-13th april 2025"
        start_day = int(ordinal_match.group(1))
        end_day = int(ordinal_match.group(2))
        month_name = ordinal_match.group(3)
        year = int(ordinal_match.group(4)) if ordinal_match.group(4) else current_year
        
        try:
            start_date = datetime.strptime(f"{start_day} {month_name} {year}", "%d %B %Y")
            end_date = datetime.strptime(f"{end_day} {month_name} {year}", "%d %B %Y")
            duration_value = (end_date - start_date).days + 1
        except ValueError:
            try:
                start_date = datetime.strptime(f"{start_day} {month_name} {year}", "%d %b %Y")
                end_date = datetime.strptime(f"{end_day} {month_name} {year}", "%d %b %Y")
                duration_value = (end_date - start_date).days + 1
            except ValueError:
                pass
    
    elif to_date_match:
        # Pattern 2: "from 22th june 2025 to 29th june 2025"
        start_day = int(to_date_match.group(1))
        start_month = to_date_match.group(2)
        start_year = int(to_date_match.group(3)) if to_date_match.group(3) else current_year
        end_day = int(to_date_match.group(4))
        end_month = to_date_match.group(5)
        end_year = int(to_date_match.group(6)) if to_date_match.group(6) else current_year
        
        try:
            start_date = datetime.strptime(f"{start_day} {start_month} {start_year}", "%d %B %Y")
            end_date = datetime.strptime(f"{end_day} {end_month} {end_year}", "%d %B %Y")
            duration_value = (end_date - start_date).days + 1
        except ValueError:
            try:
                start_date = datetime.strptime(f"{start_day} {start_month} {start_year}", "%d %b %Y")
                end_date = datetime.strptime(f"{end_day} {end_month} {end_year}", "%d %b %Y")
                duration_value = (end_date - start_date).days + 1
            except ValueError:
                pass
    
    elif numeric_match:
        # Pattern 3: "from 02-04-2025 to 29-04-2025"
        start_day = int(numeric_match.group(1))
        start_month = int(numeric_match.group(2))
        start_year = int(numeric_match.group(3))
        end_day = int(numeric_match.group(4))
        end_month = int(numeric_match.group(5))
        end_year = int(numeric_match.group(6))
        
        try:
            start_date = datetime(start_year, start_month, start_day)
            end_date = datetime(end_year, end_month, end_day)
            duration_value = (end_date - start_date).days + 1
        except ValueError:
            pass
    
    elif date_for_duration_match:
        # Pattern 4: "from 12th march for two week"
        start_day = int(date_for_duration_match.group(1))
        month_name = date_for_duration_match.group(2)
        year = int(date_for_duration_match.group(3)) if date_for_duration_match.group(3) else current_year
        duration_num = convert_text_to_number(date_for_duration_match.group(4))
        duration_unit = date_for_duration_match.group(5)
        
        try:
            start_date = datetime.strptime(f"{start_day} {month_name} {year}", "%d %B %Y")
            duration_value = convert_unit_to_days(duration_num, duration_unit)
            end_date = start_date + timedelta(days=duration_value - 1)
        except ValueError:
            try:
                start_date = datetime.strptime(f"{start_day} {month_name} {year}", "%d %b %Y")
                duration_value = convert_unit_to_days(duration_num, duration_unit)
                end_date = start_date + timedelta(days=duration_value - 1)
            except ValueError:
                pass
    
    elif duration_from_date_match:
        # Pattern 5: "for a week from 13th april"
        duration_num = convert_text_to_number(duration_from_date_match.group(1))
        duration_unit = duration_from_date_match.group(2)
        start_day = int(duration_from_date_match.group(3))
        month_name = duration_from_date_match.group(4)
        year = int(duration_from_date_match.group(5)) if duration_from_date_match.group(5) else current_year
        
        try:
            start_date = datetime.strptime(f"{start_day} {month_name} {year}", "%d %B %Y")
            duration_value = convert_unit_to_days(duration_num, duration_unit)
            end_date = start_date + timedelta(days=duration_value - 1)
        except ValueError:
            try:
                start_date = datetime.strptime(f"{start_day} {month_name} {year}", "%d %b %Y")
                duration_value = convert_unit_to_days(duration_num, duration_unit)
                end_date = start_date + timedelta(days=duration_value - 1)
            except ValueError:
                pass
    
    elif duration_on_date_match:
        # Pattern 6: "for two weeks on 3rd april"
        duration_num = convert_text_to_number(duration_on_date_match.group(1))
        duration_unit = duration_on_date_match.group(2)
        start_day = int(duration_on_date_match.group(3))
        month_name = duration_on_date_match.group(4)
        year = int(duration_on_date_match.group(5)) if duration_on_date_match.group(5) else current_year
        
        try:
            start_date = datetime.strptime(f"{start_day} {month_name} {year}", "%d %B %Y")
            duration_value = convert_unit_to_days(duration_num, duration_unit)
            end_date = start_date + timedelta(days=duration_value - 1)
        except ValueError:
            try:
                start_date = datetime.strptime(f"{start_day} {month_name} {year}", "%d %b %Y")
                duration_value = convert_unit_to_days(duration_num, duration_unit)
                end_date = start_date + timedelta(days=duration_value - 1)
            except ValueError:
                pass
    
    elif on_date_for_duration_match:
        # Pattern 7: "on 13th march for a week"
        start_day = int(on_date_for_duration_match.group(1))
        month_name = on_date_for_duration_match.group(2)
        year = int(on_date_for_duration_match.group(3)) if on_date_for_duration_match.group(3) else current_year
        duration_num = convert_text_to_number(on_date_for_duration_match.group(4))
        duration_unit = on_date_for_duration_match.group(5)
        
        try:
            start_date = datetime.strptime(f"{start_day} {month_name} {year}", "%d %B %Y")
            duration_value = convert_unit_to_days(duration_num, duration_unit)
            end_date = start_date + timedelta(days=duration_value - 1)
        except ValueError:
            try:
                start_date = datetime.strptime(f"{start_day} {month_name} {year}", "%d %b %Y")
                duration_value = convert_unit_to_days(duration_num, duration_unit)
                end_date = start_date + timedelta(days=duration_value - 1)
            except ValueError:
                pass
    
    elif duration_on_numeric_date_match:
        # Pattern 8: "for 2 weeks on 20/05/2025"
        duration_num = convert_text_to_number(duration_on_numeric_date_match.group(1))
        duration_unit = duration_on_numeric_date_match.group(2)
        start_day = int(duration_on_numeric_date_match.group(3))
        start_month = int(duration_on_numeric_date_match.group(4))
        start_year = int(duration_on_numeric_date_match.group(5))
        
        try:
            start_date = datetime(start_year, start_month, start_day)
            duration_value = convert_unit_to_days(duration_num, duration_unit)
            end_date = start_date + timedelta(days=duration_value - 1)
        except ValueError:
            pass
    
    elif on_numeric_date_for_duration_match:
        # Pattern 9: "on 05/06/2025 for two weeks"
        start_day = int(on_numeric_date_for_duration_match.group(1))
        start_month = int(on_numeric_date_for_duration_match.group(2))
        start_year = int(on_numeric_date_for_duration_match.group(3))
        duration_num = convert_text_to_number(on_numeric_date_for_duration_match.group(4))
        duration_unit = on_numeric_date_for_duration_match.group(5)
        
        try:
            start_date = datetime(start_year, start_month, start_day)
            duration_value = convert_unit_to_days(duration_num, duration_unit)
            end_date = start_date + timedelta(days=duration_value - 1)
        except ValueError:
            pass
    
    # If none of the specific patterns matched, try general approaches
    if not start_date and not end_date:
        # Seasonal date matching 
        for season, default_date in seasonal_mappings.items():
            if season in text_lower:
                year = current_year
                try:
                    start_date = datetime.strptime(f"{default_date}-{year}", "%m-%d-%Y")
                    if duration_days:
                        end_date = start_date + timedelta(days=duration_days - 1)
                    else:
                        end_date = start_date + timedelta(days=6)  # Default 7-day trip
                        duration_value = 7
                    break
                except ValueError:
                    continue
        
        # If seasonal matching failed, try dateparser as last resort
        if not start_date:
            try:
                # Look for any date-like strings
                dates_found = search_dates(text)
                if dates_found:
                    first_date = dates_found[0][1]
                    start_date = first_date
                    if duration_days:
                        end_date = start_date + timedelta(days=duration_days - 1)
            except:
                pass
    
    # Set the details
    if start_date:
        details["Start Date"] = start_date.strftime("%Y-%m-%d")
    if end_date:
        details["End Date"] = end_date.strftime("%Y-%m-%d")
    if duration_value and not details.get("Trip Duration"):
        details["Trip Duration"] = f"{duration_value} days"
    
    # Extract budget
    budget_pattern = r'(?:budget|spend|cost|price|money|funds)\s*(?:is|of)?\s*(?:around|about|approximately)?\s*[\$₹€£]?(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:k|thousand|lakh|crore|million|billion)?'
    budget_match = re.search(budget_pattern, text, re.IGNORECASE)
    if budget_match:
        budget_amount = budget_match.group(1)
        # Check for currency symbols or mentions
        if "₹" in text or "rupee" in text_lower or "inr" in text_lower:
            currency = "₹"
        elif "$" in text or "dollar" in text_lower or "usd" in text_lower:
            currency = "$"
        elif "€" in text or "euro" in text_lower:
            currency = "€"
        elif "£" in text or "pound" in text_lower or "gbp" in text_lower:
            currency = "£"
        else:
            currency = "₹"  # Default to Indian Rupees
        
        # Handle scale modifiers
        if "k" in text_lower or "thousand" in text_lower:
            budget_amount = str(int(float(budget_amount) * 1000))
        elif "lakh" in text_lower:
            budget_amount = str(int(float(budget_amount) * 100000))
        elif "crore" in text_lower:
            budget_amount = str(int(float(budget_amount) * 10000000))
        elif "million" in text_lower:
            budget_amount = str(int(float(budget_amount) * 1000000))
        elif "billion" in text_lower:
            budget_amount = str(int(float(budget_amount) * 1000000000))
        
        details["Budget Range"] = f"{currency}{budget_amount}"
    
    # Extract number of travelers
    travelers_pattern = r'(?:(\d+)\s*(?:people|person|travelers?|pax|individuals?|adults?)|(?:family|group)\s*of\s*(\d+)|(?:me|I)\s*(?:and|with)\s*(\d+)\s*(?:others?|friends?|family)?)'
    travelers_match = re.search(travelers_pattern, text, re.IGNORECASE)
    if travelers_match:
        # Get the first non-None group
        num = travelers_match.group(1) or travelers_match.group(2) or travelers_match.group(3)
        if num:
            # If pattern is "me and X others", add 1 to X
            if travelers_match.group(3):
                details["Number of Travelers"] = str(int(num) + 1)
            else:
                details["Number of Travelers"] = num
    elif any(word in text_lower for word in ["alone", "solo", "myself", "just me"]):
        details["Number of Travelers"] = "1"
    elif any(word in text_lower for word in ["couple", "two of us", "me and my"]):
        details["Number of Travelers"] = "2"
    
    # Extract trip type
    if any(word in text_lower for word in ["adventure", "trekking", "hiking", "rafting", "climbing"]):
        details["Trip Type"] = "Adventure"
    elif any(word in text_lower for word in ["beach", "resort", "spa", "relax", "leisure", "vacation"]):
        details["Trip Type"] = "Leisure/Beach"
    elif any(word in text_lower for word in ["business", "work", "conference", "meeting"]):
        details["Trip Type"] = "Business"
    elif any(word in text_lower for word in ["culture", "heritage", "museum", "historical", "sightseeing"]):
        details["Trip Type"] = "Cultural/Sightseeing"
    elif any(word in text_lower for word in ["family", "kids", "children"]):
        details["Trip Type"] = "Family"
    elif any(word in text_lower for word in ["honeymoon", "romantic", "couple"]):
        details["Trip Type"] = "Romantic/Honeymoon"
    
    # Extract transportation preferences
    if any(word in text_lower for word in ["flight", "fly", "air", "plane"]):
        details["Transportation Preferences"] = "Flight"
    elif any(word in text_lower for word in ["train", "railway"]):
        details["Transportation Preferences"] = "Train"
    elif any(word in text_lower for word in ["car", "drive", "road trip"]):
        details["Transportation Preferences"] = "Car/Road Trip"
    elif any(word in text_lower for word in ["bus", "coach"]):
        details["Transportation Preferences"] = "Bus"
    
    # Extract accommodation preferences
    if any(word in text_lower for word in ["luxury", "5 star", "premium", "high-end"]):
        details["Accommodation Preferences"] = "Luxury"
    elif any(word in text_lower for word in ["budget", "cheap", "affordable", "hostel"]):
        details["Accommodation Preferences"] = "Budget"
    elif any(word in text_lower for word in ["mid-range", "3 star", "moderate"]):
        details["Accommodation Preferences"] = "Mid-range"
    elif any(word in text_lower for word in ["resort", "all-inclusive"]):
        details["Accommodation Preferences"] = "Resort"
    elif any(word in text_lower for word in ["homestay", "local", "authentic"]):
        details["Accommodation Preferences"] = "Homestay/Local"
    
    # Extract special requirements
    special_requirements = []
    if any(word in text_lower for word in ["vegetarian", "vegan", "no meat"]):
        special_requirements.append("Vegetarian/Vegan food")
    if any(word in text_lower for word in ["disability", "wheelchair", "accessible"]):
        special_requirements.append("Accessibility requirements")
    if any(word in text_lower for word in ["pets", "dog", "cat"]):
        special_requirements.append("Pet-friendly")
    if any(word in text_lower for word in ["medical", "medicine", "treatment"]):
        special_requirements.append("Medical considerations")
    if special_requirements:
        details["Special Requirements"] = ", ".join(special_requirements)
    
    return details

def generate_itinerary(details, user_input):
    try:
        # Setup Gemini
        model = setup_gemini()
        
        # Create a comprehensive prompt for the AI
        prompt = f"""
        Create a detailed travel itinerary based on the following information:
        
        **Travel Details:**
        - Destination: {details.get('Destination', 'Not specified')}
        - Starting Location: {details.get('Starting Location', 'Not specified')}
        - Duration: {details.get('Trip Duration', 'Not specified')}
        - Start Date: {details.get('Start Date', 'Not specified')}
        - End Date: {details.get('End Date', 'Not specified')}
        - Number of Travelers: {details.get('Number of Travelers', 'Not specified')}
        - Budget: {details.get('Budget Range', 'Not specified')}
        - Trip Type: {details.get('Trip Type', 'Not specified')}
        - Transportation: {details.get('Transportation Preferences', 'Not specified')}
        - Accommodation: {details.get('Accommodation Preferences', 'Not specified')}
        - Special Requirements: {details.get('Special Requirements', 'Not specified')}
        
        **Original Request:** {user_input}
        
        Please create a comprehensive travel itinerary that includes:
        
        ## 1. Trip Overview
        - Brief summary of the trip
        - Key highlights and themes
        
        ## 2. Daily Itinerary
        For each day, provide:
        - **Day X: [Location/Theme]**
        - **Morning:** Detailed morning activities with times
        - **Afternoon:** Detailed afternoon activities with times  
        - **Evening:** Detailed evening activities with times
        - **Meals:**
          - Breakfast: Specific restaurant/location recommendations
          - Lunch: Specific restaurant/location recommendations  
          - Dinner: Specific restaurant/location recommendations
        - **Accommodation:** Specific hotel/accommodation recommendations with brief description
        
        ## 3. Accommodation Details
        - Specific hotel recommendations with:
          - Hotel names and locations
          - Price ranges per night
          - Key amenities and features
          - Why each hotel fits the traveler's needs
        
        ## 4. Dining Recommendations  
        - Must-try restaurants and local cuisines
        - Food experiences and specialties
        - Price ranges and dining styles
        
        ## 5. Attractions & Activities
        - Top attractions with descriptions
        - Unique experiences and activities
        - Entry fees and timings where applicable
        
        ## 6. Budget Breakdown
        - Accommodation costs
        - Transportation costs
        - Food and dining costs
        - Activities and attractions costs
        - Shopping and miscellaneous costs
        - Total estimated cost
        
        ## 7. Essential Information
        - Transportation details (flights, local transport)
        - Weather considerations and packing suggestions
        - Local customs and etiquette
        - Emergency contacts and important numbers
        - Currency and payment methods
        - Language tips and useful phrases
        
        Please make the itinerary detailed, practical, and tailored to the specific requirements mentioned. Include specific names, locations, and realistic time estimates.
        """
        
        # Generate the itinerary
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        st.error(f"Error generating itinerary: {str(e)}")
        return None

def parse_itinerary_data(itinerary_text):
    """Parse the AI-generated itinerary into structured data for different tabs"""
    
    parsed_data = {
        "overview": "",
        "days": [],
        "accommodation": "",
        "dining": "",
        "attractions": "",
        "budget": "",
        "essential_info": "",
        "transportation": ""
    }
    
    if not itinerary_text:
        return parsed_data
    
    # Extract overview section
    overview_match = re.search(r'##\s*1\.\s*Trip Overview(.*?)(?=##\s*2\.|$)', itinerary_text, re.DOTALL | re.IGNORECASE)
    if overview_match:
        parsed_data["overview"] = overview_match.group(1).strip()
    
    # Extract accommodation section
    accommodation_match = re.search(r'##\s*3\.\s*Accommodation Details(.*?)(?=##\s*4\.|$)', itinerary_text, re.DOTALL | re.IGNORECASE)
    if accommodation_match:
        parsed_data["accommodation"] = accommodation_match.group(1).strip()
    
    # Extract dining section
    dining_match = re.search(r'##\s*4\.\s*Dining Recommendations(.*?)(?=##\s*5\.|$)', itinerary_text, re.DOTALL | re.IGNORECASE)
    if dining_match:
        parsed_data["dining"] = dining_match.group(1).strip()
    
    # Extract attractions section
    attractions_match = re.search(r'##\s*5\.\s*Attractions & Activities(.*?)(?=##\s*6\.|$)', itinerary_text, re.DOTALL | re.IGNORECASE)
    if attractions_match:
        parsed_data["attractions"] = attractions_match.group(1).strip()
    
    # Extract budget section
    budget_match = re.search(r'##\s*6\.\s*Budget Breakdown(.*?)(?=##\s*7\.|$)', itinerary_text, re.DOTALL | re.IGNORECASE)
    if budget_match:
        parsed_data["budget"] = budget_match.group(1).strip()
    
    # Extract essential info section
    essential_match = re.search(r'##\s*7\.\s*Essential Information(.*?)$', itinerary_text, re.DOTALL | re.IGNORECASE)
    if essential_match:
        parsed_data["essential_info"] = essential_match.group(1).strip()
    
    # Extract daily itinerary section
    daily_section_match = re.search(r'##\s*2\.\s*Daily Itinerary(.*?)(?=##\s*3\.|$)', itinerary_text, re.DOTALL | re.IGNORECASE)
    if daily_section_match:
        daily_content = daily_section_match.group(1).strip()
        
        # Find all day entries
        day_matches = list(re.finditer(r'\*\*Day (\d+):[^*]*\*\*', daily_content, re.IGNORECASE))
        
        for i, match in enumerate(day_matches):
            day_num = int(match.group(1))
            day_title = match.group(0).strip()
            
            # Get the content of this day (until next day or end)
            start_pos = match.end()
            end_pos = day_matches[i+1].start() if i < len(day_matches) - 1 else len(daily_content)
            day_content = daily_content[start_pos:end_pos].strip()
            
            # Initialize day data structure
            day_data = {
                "day_number": day_num,
                "title": day_title,
                "morning": "",
                "afternoon": "",
                "evening": "",
                "meals": {
                    "breakfast": "",
                    "lunch": "",
                    "dinner": ""
                },
                "accommodation": "",
                "activities": []
            }
            
            # Extract date if present
            date_match = re.search(r'(?:Date|On):\s*(\d{1,2}(?:st|nd|rd|th)?\s+\w+(?:\s+\d{4})?|\d{4}-\d{2}-\d{2}|\w+\s+\d{1,2}(?:st|nd|rd|th)?,\s*\d{4})', day_content, re.IGNORECASE)
            if date_match:
                day_data["date"] = date_match.group(1).strip()
            
            # Extract time-based activities with more detailed pattern matching
            # Morning section
            morning_match = re.search(r'(?:Morning|AM)(?:[\s\-:]+)([\s\S]*?)(?=(?:Afternoon|Lunch|PM|Evening|Dinner|Accommodation|Day|$))', day_content, re.IGNORECASE)
            if morning_match:
                day_data["morning"] = morning_match.group(1).strip()
            
            # Also check for bullet points in the Morning section
            if "* **Morning:**" in day_content or "- **Morning:**" in day_content:
                morning_bullet = re.search(r'(?:\*|\-)\s+\*\*Morning:\*\*\s+([\s\S]*?)(?=(?:\*|\-)\s+\*\*(?:Afternoon|Lunch|Evening|Dinner|Meals|Accommodation)|$)', day_content, re.IGNORECASE)
                if morning_bullet:
                    day_data["morning"] = morning_bullet.group(1).strip()
            
            # Afternoon section
            afternoon_match = re.search(r'(?:Afternoon|PM)(?:[\s\-:]+)([\s\S]*?)(?=(?:Evening|Dinner|Accommodation|Day|$))', day_content, re.IGNORECASE)
            if afternoon_match:
                day_data["afternoon"] = afternoon_match.group(1).strip()
            
            # Also check for bullet points in the Afternoon section
            if "* **Afternoon:**" in day_content or "- **Afternoon:**" in day_content:
                afternoon_bullet = re.search(r'(?:\*|\-)\s+\*\*Afternoon:\*\*\s+([\s\S]*?)(?=(?:\*|\-)\s+\*\*(?:Evening|Dinner|Meals|Accommodation)|$)', day_content, re.IGNORECASE)
                if afternoon_bullet:
                    day_data["afternoon"] = afternoon_bullet.group(1).strip()
            
            # Evening section
            evening_match = re.search(r'(?:Evening|Night)(?:[\s\-:]+)([\s\S]*?)(?=(?:Accommodation|Day|$))', day_content, re.IGNORECASE)
            if evening_match:
                day_data["evening"] = evening_match.group(1).strip()
            
            # Also check for bullet points in the Evening section
            if "* **Evening:**" in day_content or "- **Evening:**" in day_content:
                evening_bullet = re.search(r'(?:\*|\-)\s+\*\*Evening:\*\*\s+([\s\S]*?)(?=(?:\*|\-)\s+\*\*(?:Meals|Accommodation)|$)', day_content, re.IGNORECASE)
                if evening_bullet:
                    day_data["evening"] = evening_bullet.group(1).strip()
            
            # Extract meals - enhanced to capture more details
            # Check for the Meals section with bullet points
            meals_section = re.search(r'(?:\*|\-)\s+\*\*Meals:\*\*([\s\S]*?)(?=(?:\*|\-)\s+\*\*(?:Accommodation)|$)', day_content, re.IGNORECASE)
            if meals_section:
                meals_content = meals_section.group(1).strip()
                
                # Extract breakfast details
                breakfast_match = re.search(r'(?:Breakfast|breakfast):\s+([^\n]+)', meals_content, re.IGNORECASE)
                if breakfast_match:
                    day_data["meals"]["breakfast"] = breakfast_match.group(1).strip()
                
                # Extract lunch details
                lunch_match = re.search(r'(?:Lunch|lunch):\s+([^\n]+)', meals_content, re.IGNORECASE)
                if lunch_match:
                    day_data["meals"]["lunch"] = lunch_match.group(1).strip()
                
                # Extract dinner details
                dinner_match = re.search(r'(?:Dinner|dinner):\s+([^\n]+)', meals_content, re.IGNORECASE)
                if dinner_match:
                    day_data["meals"]["dinner"] = dinner_match.group(1).strip()
            else:
                # Fallback to original approach
                breakfast_match = re.search(r'(?:Breakfast)(?:[\s\-:]+)([^#\n]+)', day_content, re.IGNORECASE)
                if breakfast_match:
                    day_data["meals"]["breakfast"] = breakfast_match.group(1).strip()
                
                lunch_match = re.search(r'(?:Lunch)(?:[\s\-:]+)([^#\n]+)', day_content, re.IGNORECASE)
                if lunch_match:
                    day_data["meals"]["lunch"] = lunch_match.group(1).strip()
                
                dinner_match = re.search(r'(?:Dinner)(?:[\s\-:]+)([^#\n]+)', day_content, re.IGNORECASE)
                if dinner_match:
                    day_data["meals"]["dinner"] = dinner_match.group(1).strip()
            
            # Extract accommodation - enhanced to capture more details
            accommodation_match = re.search(r'(?:\*|\-)\s+\*\*Accommodation:\*\*\s+([\s\S]*?)(?=(?:\*|\-)\s+\*\*|$)', day_content, re.IGNORECASE)
            if accommodation_match:
                day_data["accommodation"] = accommodation_match.group(1).strip()
            else:
                # Fallback approach
                accommodation_match = re.search(r'(?:Accommodation|Stay|Hotel)(?:[\s\-:]+)([^#\n]+)', day_content, re.IGNORECASE)
                if accommodation_match:
                    day_data["accommodation"] = accommodation_match.group(1).strip()
            
            # Extract activities list
            activities_matches = re.findall(r'(?:Visit|Explore|Experience|Activity)(?:[\s\-:]+)([^#\n]+)', day_content, re.IGNORECASE)
            if activities_matches:
                day_data["activities"] = [activity.strip() for activity in activities_matches]
            else:
                # Try to extract from bullet points or numbered lists
                bullet_activities = re.findall(r'(?:^|\n)(?:\d+\.|\*|\-)\s*([^*#\n]+)', day_content, re.MULTILINE)
                if bullet_activities:
                    # Filter out section headers and keep actual activities
                    all_activities = []
                    for activity in bullet_activities:
                        activity = activity.strip()
                        # Skip if it looks like a section header
                        if not re.match(r'\*\*(Morning|Afternoon|Evening|Meals|Accommodation):', activity):
                            # Split by common separators and clean
                            section_activities = re.split(r'[,;]', activity)
                            all_activities.extend([act.strip() for act in section_activities if act.strip()])
                
                day_data["activities"] = all_activities
            
            # Enhanced fallback: Extract any structured content from day text
            if not day_data["morning"] and not day_data["afternoon"] and not day_data["evening"]:
                # Try to extract from bullet points or numbered lists
                content_lines = [line.strip() for line in day_content.split('\n') if line.strip()]
                activities = []
                
                for line in content_lines:
                    # Skip headers and formatting
                    if re.match(r'^\*+|^#+|^Day \d+|^\-+$', line):
                        continue
                    # Look for bullet points or activities
                    if re.match(r'^[\*\-•]\s*', line) or re.match(r'^\d+\.', line):
                        activity = re.sub(r'^[\*\-•]\s*|\d+\.\s*', '', line).strip()
                        if activity and len(activity) > 10:
                            activities.append(activity)
                    elif len(line) > 15 and not line.startswith('**'):
                        activities.append(line)
                
                # Distribute activities across time periods
                if activities:
                    if len(activities) >= 1:
                        day_data["morning"] = activities[0]
                    if len(activities) >= 2:
                        day_data["afternoon"] = activities[1]
                    if len(activities) >= 3:
                        day_data["evening"] = activities[2]
                    else:
                        # If only 1-2 activities, use generic content
                        if not day_data["afternoon"]:
                            day_data["afternoon"] = "Continue exploring local attractions"
                        if not day_data["evening"]:
                            day_data["evening"] = "Evening leisure time"
            
            # Enhanced meal extraction if still empty  
            if not day_data["meals"]["breakfast"] and not day_data["meals"]["lunch"] and not day_data["meals"]["dinner"]:
                # Try to find meal references in the day content
                for meal_type in ["breakfast", "lunch", "dinner"]:
                    # Look for various meal patterns
                    meal_patterns = [
                        rf'{meal_type}[:\-]\s*([^\n,;]+)',
                        rf'\b{meal_type}\s+at\s+([^\n,;]+)',
                        rf'for\s+{meal_type}[:\-]?\s*([^\n,;]+)',
                        rf'{meal_type.capitalize()}[:\-]\s*([^\n,;]+)'
                    ]
                    
                    for pattern in meal_patterns:
                        meal_match = re.search(pattern, day_content, re.IGNORECASE)
                        if meal_match:
                            meal_info = meal_match.group(1).strip()
                            if len(meal_info) > 5:  # Only use if substantial
                                day_data["meals"][meal_type] = meal_info
                                break
                
                # If still no meals found, add realistic defaults
                if not day_data["meals"]["breakfast"]:
                    day_data["meals"]["breakfast"] = "Local breakfast restaurant or hotel dining"
                if not day_data["meals"]["lunch"]:
                    day_data["meals"]["lunch"] = "Traditional local cuisine for lunch"
                if not day_data["meals"]["dinner"]:
                    day_data["meals"]["dinner"] = "Local restaurant with regional specialties"
            
            # Enhanced accommodation extraction if still empty
            if not day_data["accommodation"]:
                # Try to find accommodation references in the day content
                accommodation_patterns = [
                    r'(?:accommodation|hotel|stay|lodge|resort)[:\-]\s*([^\n,;]+)',
                    r'stay\s+at\s+([^\n,;]+)',
                    r'overnight\s+at\s+([^\n,;]+)',
                    r'accommodation[:\-]?\s*([^\n,;]+)'
                ]
                
                for pattern in accommodation_patterns:
                    acc_match = re.search(pattern, day_content, re.IGNORECASE)
                    if acc_match:
                        acc_info = acc_match.group(1).strip()
                        if len(acc_info) > 5:  # Only use if substantial
                            day_data["accommodation"] = acc_info
                            break
                
                # If still no accommodation found, add a realistic default
                if not day_data["accommodation"]:
                    day_data["accommodation"] = "Mid-range hotel or guesthouse in city center"
            
            parsed_data["days"].append(day_data)
        
        # Extract transportation details
        extract_transportation(itinerary_text, parsed_data)
    
    return parsed_data

def extract_transportation(itinerary_text, parsed_data):
    """Extract transportation information from the itinerary"""
    
    transportation_info = []
    
    # Look for flight information
    flight_matches = re.findall(r'(?:flight|airline|plane|air travel)[\s\-:]*([^\n.]+)', itinerary_text, re.IGNORECASE)
    for match in flight_matches:
        if len(match.strip()) > 10:
            transportation_info.append(f"✈️ Flight: {match.strip()}")
    
    # Look for train information
    train_matches = re.findall(r'(?:train|railway|rail)[\s\-:]*([^\n.]+)', itinerary_text, re.IGNORECASE)
    for match in train_matches:
        if len(match.strip()) > 10:
            transportation_info.append(f"🚂 Train: {match.strip()}")
    
    # Look for local transport
    local_transport_matches = re.findall(r'(?:taxi|bus|metro|local transport|public transport)[\s\-:]*([^\n.]+)', itinerary_text, re.IGNORECASE)
    for match in local_transport_matches:
        if len(match.strip()) > 10:
            transportation_info.append(f"🚌 Local Transport: {match.strip()}")
    
    if transportation_info:
        parsed_data["transportation"] = "\n".join(transportation_info)
    else:
        parsed_data["transportation"] = "Transportation details will vary based on your preferences and final bookings."

def display_day_details(day_data):
    """Display detailed information for a specific day"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Morning activities
        st.subheader("🌅 Morning")
        if day_data["morning"]:
            st.write(day_data["morning"])
        else:
            st.info("Morning activities not specified")
        
        # Afternoon activities
        st.subheader("☀️ Afternoon") 
        if day_data["afternoon"]:
            st.write(day_data["afternoon"])
        else:
            st.info("Afternoon activities not specified")
        
        # Evening activities
        st.subheader("🌙 Evening")
        if day_data["evening"]:
            st.write(day_data["evening"])
        else:
            st.info("Evening activities not specified")
    
    with col2:
        # Meals
        st.subheader("🍽️ Meals")
        
        # Breakfast
        if day_data["meals"]["breakfast"]:
            st.write(f"**Breakfast:** {day_data['meals']['breakfast']}")
        
        # Lunch
        if day_data["meals"]["lunch"]:
            st.write(f"**Lunch:** {day_data['meals']['lunch']}")
        
        # Dinner
        if day_data["meals"]["dinner"]:
            st.write(f"**Dinner:** {day_data['meals']['dinner']}")
        
        # Accommodation
        st.subheader("🏨 Accommodation")
        if day_data["accommodation"]:
            st.write(day_data["accommodation"])
        else:
            st.info("Accommodation details not specified")
        
        # Activities
        if day_data["activities"]:
            st.subheader("🎯 Key Activities")
            for activity in day_data["activities"]:
                st.write(f"• {activity}")

def create_trip_examples():
    """Create example trip requests for users"""
    
    examples = {
        "Beach Vacation": {
            "text": "Plan a 7-day beach vacation to Goa for 2 people in December with a budget of ₹50,000. We want to relax, enjoy water sports, and experience local cuisine.",
            "icon": "🏖️"
        },
        "Adventure Trip": {
            "text": "I want to go on a 10-day adventure trip to Nepal for trekking in the Himalayas. Budget is $2000, traveling solo in March.",
            "icon": "🏔️"
        },
        "Cultural Tour": {
            "text": "Plan a cultural tour of Rajasthan for 2 weeks in January. Family of 4, interested in heritage sites, local crafts, and traditional food. Budget ₹1.5 lakh.",
            "icon": "🏛️"
        },
        "International Trip": {
            "text": "From Mumbai to Thailand for 8 days in February. Couple trip, mid-range hotels, interested in temples, street food, and shopping. Budget ₹80,000.",
            "icon": "🌍"
        },
        "Weekend Getaway": {
            "text": "Quick weekend trip from Delhi to Shimla for 3 days in April. Budget ₹15,000 for 2 people, prefer hill stations and pleasant weather.",
            "icon": "🚗"
        }
    }
    
    return examples

# Main application
def main():
    # Header
    st.title("🌍 Travel Planner Pro")
    st.markdown("### Plan your perfect trip with AI-powered recommendations")
    
    # Sidebar
    with st.sidebar:
        st.header("✈️ Quick Trip Examples")
        st.markdown("Click on any example to get started:")
        
        examples = create_trip_examples()
        
        for title, example in examples.items():
            if st.button(f"{example['icon']} {title}", key=f"example_{title}", use_container_width=True):
                st.session_state.example_text = example['text']
        
        st.markdown("---")
        st.markdown("### 💡 Tips for Better Results")
        st.markdown("""
        - Mention specific dates or months
        - Include your budget range
        - Specify number of travelers
        - Mention preferences (adventure, relaxation, culture)
        - Include starting city for better recommendations
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 What You'll Get")
        st.markdown("""
        - **Detailed daily itinerary**
        - **Hotel recommendations**
        - **Restaurant suggestions**
        - **Local attractions**
        - **Budget breakdown**
        - **Travel tips**
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input area
        default_text = st.session_state.get('example_text', '')
        user_input = st.text_area(
            "📝 Describe your dream trip:",
            placeholder="Example: Plan a 5-day trip to Paris for 2 people in June. We love art, good food, and romantic experiences. Budget is $3000.",
            height=120,
            value=default_text,
            key="trip_input"
        )
        
        # Clear the example text after it's been used
        if 'example_text' in st.session_state:
            del st.session_state.example_text
        
        # Generate button
        generate_col1, generate_col2, generate_col3 = st.columns([1, 2, 1])
        with generate_col2:
            generate_button = st.button("🚀 Generate My Itinerary", type="primary", use_container_width=True)
    
    with col2:
        # Trip details preview
        if user_input:
            st.subheader("🔍 Trip Details Preview")
            with st.spinner("Analyzing your request..."):
                details = extract_details(user_input)
                
                if details:
                    # Create a nice display of extracted details
                    for key, value in details.items():
                        if value:
                            if key == "Destination":
                                st.write(f"🎯 **{key}:** {value}")
                            elif key == "Starting Location":
                                st.write(f"📍 **{key}:** {value}")
                            elif key == "Trip Duration":
                                st.write(f"⏰ **{key}:** {value}")
                            elif key == "Budget Range":
                                st.write(f"💰 **{key}:** {value}")
                            elif key == "Number of Travelers":
                                st.write(f"👥 **{key}:** {value}")
                            elif key == "Trip Type":
                                st.write(f"🎨 **{key}:** {value}")
                            else:
                                st.write(f"**{key}:** {value}")
                else:
                    st.info("Enter your trip details to see the analysis")
    
    # Generate itinerary when button is clicked
    if generate_button and user_input:
        with st.spinner("🤖 AI is crafting your perfect itinerary... This may take a few moments."):
            details = extract_details(user_input)
            itinerary = generate_itinerary(details, user_input)
            
            if itinerary:
                # Store in session state
                st.session_state.itinerary = itinerary
                st.session_state.details = details
                st.session_state.user_input = user_input
                st.success("🎉 Your itinerary is ready!")
                st.rerun()
    
    # Display itinerary if available
    if hasattr(st.session_state, 'itinerary') and st.session_state.itinerary:
        st.markdown("---")
        
        # Parse the itinerary data
        parsed_data = parse_itinerary_data(st.session_state.itinerary)
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📋 Overview", 
            "📅 Daily Itinerary", 
            "🏨 Accommodation", 
            "🍽️ Dining", 
            "🎯 Attractions", 
            "💰 Budget", 
            "ℹ️ Essential Info"
        ])
        
        with tab1:
            st.header("🌟 Trip Overview")
            if parsed_data["overview"]:
                st.markdown(parsed_data["overview"])
            else:
                st.info("Overview information is being processed...")
            
            # Display key trip details
            if hasattr(st.session_state, 'details'):
                st.subheader("📊 Trip Summary")
                details_df_data = []
                for key, value in st.session_state.details.items():
                    if value:
                        details_df_data.append({"Detail": key, "Information": str(value)})
                
                if details_df_data:
                    try:
                        details_df = pd.DataFrame(details_df_data)
                        st.dataframe(details_df, use_container_width=True, hide_index=True)
                    except Exception as e:
                        # Fallback to simple display if DataFrame creation fails
                        for item in details_df_data:
                            st.write(f"**{item['Detail']}:** {item['Information']}")
        
        with tab2:
            st.header("📅 Daily Itinerary")
            
            if parsed_data["days"]:
                for day in parsed_data["days"]:
                    with st.expander(f"🗓️ {day['title']}", expanded=False):
                        display_day_details(day)
            else:
                st.info("Daily itinerary is being processed...")
                # Show raw itinerary as fallback
                if st.session_state.itinerary:
                    daily_section = re.search(r'##\s*2\.\s*Daily Itinerary(.*?)(?=##\s*3\.|$)', st.session_state.itinerary, re.DOTALL | re.IGNORECASE)
                    if daily_section:
                        st.markdown(daily_section.group(1).strip())
        
        with tab3:
            st.header("🏨 Accommodation Recommendations")
            if parsed_data["accommodation"]:
                st.markdown(parsed_data["accommodation"])
            else:
                st.info("Accommodation recommendations are being processed...")
                # Extract and show accommodation info as fallback
                accommodation_section = re.search(r'##\s*3\.\s*Accommodation(.*?)(?=##\s*4\.|$)', st.session_state.itinerary, re.DOTALL | re.IGNORECASE)
                if accommodation_section:
                    st.markdown(accommodation_section.group(1).strip())
        
        with tab4:
            st.header("🍽️ Dining Recommendations")
            if parsed_data["dining"]:
                st.markdown(parsed_data["dining"])
            else:
                st.info("Dining recommendations are being processed...")
                # Extract and show dining info as fallback
                dining_section = re.search(r'##\s*4\.\s*Dining(.*?)(?=##\s*5\.|$)', st.session_state.itinerary, re.DOTALL | re.IGNORECASE)
                if dining_section:
                    st.markdown(dining_section.group(1).strip())
        
        with tab5:
            st.header("🎯 Attractions & Activities")
            if parsed_data["attractions"]:
                st.markdown(parsed_data["attractions"])
            else:
                st.info("Attractions and activities are being processed...")
                # Extract and show attractions info as fallback
                attractions_section = re.search(r'##\s*5\.\s*Attractions(.*?)(?=##\s*6\.|$)', st.session_state.itinerary, re.DOTALL | re.IGNORECASE)
                if attractions_section:
                    st.markdown(attractions_section.group(1).strip())
        
        with tab6:
            st.header("💰 Budget Breakdown")
            if parsed_data["budget"]:
                st.markdown(parsed_data["budget"])
            else:
                st.info("Budget breakdown is being processed...")
                # Extract and show budget info as fallback
                budget_section = re.search(r'##\s*6\.\s*Budget(.*?)(?=##\s*7\.|$)', st.session_state.itinerary, re.DOTALL | re.IGNORECASE)
                if budget_section:
                    st.markdown(budget_section.group(1).strip())
        
        with tab7:
            st.header("ℹ️ Essential Travel Information")
            if parsed_data["essential_info"]:
                st.markdown(parsed_data["essential_info"])
            else:
                st.info("Essential information is being processed...")
                # Extract and show essential info as fallback
                essential_section = re.search(r'##\s*7\.\s*Essential(.*?)$', st.session_state.itinerary, re.DOTALL | re.IGNORECASE)
                if essential_section:
                    st.markdown(essential_section.group(1).strip())
            
            # Add transportation info if available
            if parsed_data["transportation"]:
                st.subheader("🚗 Transportation Details")
                st.markdown(parsed_data["transportation"])
        
        # Download options
        st.markdown("---")
        st.header("📥 Download Your Itinerary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Text download
            st.download_button(
                label="📄 Download as Text",
                data=st.session_state.itinerary,
                file_name=f"travel_itinerary_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
        
        with col2:
            # JSON download for structured data
            json_data = {
                "user_input": st.session_state.user_input,
                "extracted_details": st.session_state.details,
                "itinerary": st.session_state.itinerary,
                "parsed_data": parsed_data,
                "generated_date": datetime.now().isoformat()
            }
            
            st.download_button(
                label="📊 Download as JSON",
                data=json.dumps(json_data, indent=2),
                file_name=f"travel_data_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        with col3:
            # Clear session to start over
            if st.button("🔄 Plan Another Trip", type="secondary"):
                # Clear session state
                for key in ['itinerary', 'details', 'user_input']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

if __name__ == "__main__":
    main()
