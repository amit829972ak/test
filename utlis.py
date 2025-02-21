from datetime import datetime
import re

def extract_travel_info(text):
    """Extract travel information from the input text."""
    
    # Initialize default values
    info = {
        "Starting Location": "Not specified",
        "Destination": "Not specified",
        "Start Date": "Not specified",
        "End Date": "Not specified",
        "Trip Duration": "Not specified",
        "Trip Type": "Not specified",
        "Number of Travelers": {
            "Adults": 0,
            "Children": 0,
            "Infants": 0
        },
        "Budget Range": "Not specified",
        "Transportation Preferences": "Not specified",
        "Accommodation Preferences": "Not specified",
        "Special Requirements": "None specified"
    }
    
    if not text:
        return info
    
    # Convert to lowercase for easier matching
    text = text.lower()
    
    # Extract locations
    from_to_match = re.search(r'from\s+([a-zA-Z\s]+)\s+to\s+([a-zA-Z\s]+)', text)
    if from_to_match:
        info["Starting Location"] = from_to_match.group(1).strip().title()
        info["Destination"] = from_to_match.group(2).strip().title()
    
    # Extract dates
    date_pattern = r'(?:from|on|between)?\s*([a-zA-Z]+\s*\d{1,2}(?:\s*,?\s*\d{4})?)'
    dates = re.findall(date_pattern, text)
    if len(dates) >= 2:
        try:
            start_date = datetime.strptime(dates[0].strip(), '%B%d %Y')
            end_date = datetime.strptime(dates[1].strip(), '%B%d %Y')
            info["Start Date"] = start_date.strftime('%B %d, %Y')
            info["End Date"] = end_date.strftime('%B %d, %Y')
        except ValueError:
            pass
    
    # Extract trip duration
    duration_match = re.search(r'(\d+)\s*days?', text)
    if duration_match:
        info["Trip Duration"] = f"{duration_match.group(1)} days"
    
    # Extract trip type
    if 'round trip' in text:
        info["Trip Type"] = "Round Trip"
    elif 'one way' in text:
        info["Trip Type"] = "One Way"
    
    # Extract number of travelers
    adults = re.search(r'(\d+)\s*adults?', text)
    children = re.search(r'(\d+)\s*(?:kid|child|children)', text)
    infants = re.search(r'(\d+)\s*infant', text)
    
    if adults:
        info["Number of Travelers"]["Adults"] = int(adults.group(1))
    if children:
        info["Number of Travelers"]["Children"] = int(children.group(1))
    if infants:
        info["Number of Travelers"]["Infants"] = int(infants.group(1))
    
    # Extract budget
    budget_match = re.search(r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)', text)
    if budget_match:
        info["Budget Range"] = f"${budget_match.group(1)}"
    
    # Extract transportation preferences
    transport_keywords = ['flight', 'plane', 'train', 'bus', 'car', 'ship', 'cruise']
    for keyword in transport_keywords:
        if keyword in text:
            info["Transportation Preferences"] = keyword.title()
            break
    
    # Extract accommodation preferences
    accommodation_keywords = ['hotel', 'hostel', 'airbnb', 'apartment', 'resort']
    for keyword in accommodation_keywords:
        if keyword in text:
            info["Accommodation Preferences"] = keyword.title()
            break
    
    return info

def format_travelers(travelers):
    """Format the travelers information for display."""
    parts = []
    if travelers["Adults"] > 0:
        parts.append(f"{travelers['Adults']} Adult{'s' if travelers['Adults'] > 1 else ''}")
    if travelers["Children"] > 0:
        parts.append(f"{travelers['Children']} Child{'ren' if travelers['Children'] > 1 else ''}")
    if travelers["Infants"] > 0:
        parts.append(f"{travelers['Infants']} Infant{'s' if travelers['Infants'] > 1 else ''}")
    return ", ".join(parts) if parts else "Not specified"
