"""
Geographical Utilities for Nassau Candy Factory Reallocation Project
-------------------------------------------------------------------
This module handles:
1. Factory location coordinates (Latitude and Longitude).
2. Product-to-Factory default legacy mappings.
3. State/City coordinate approximations.
4. Haversine formula to compute transit distance in miles.
"""

import math

# 1. Master dictionary of the 5 factories and their GPS coordinates
# Coordinates are provided in the official project instructions.
FACTORIES = {
    "Lot's O' Nuts": {
        "latitude": 32.881893,
        "longitude": -111.768036,
        "state": "Arizona",
        "city": "Casa Grande",
        "region_hub": "West / Pacific"
    },
    "Wicked Choccy's": {
        "latitude": 32.076176,
        "longitude": -81.088371,
        "state": "Georgia",
        "city": "Savannah",
        "region_hub": "Southeast / Atlantic"
    },
    "Sugar Shack": {
        "latitude": 48.119140,
        "longitude": -96.181150,
        "state": "Minnesota",
        "city": "Thief River Falls",
        "region_hub": "Upper Midwest / Interior"
    },
    "Secret Factory": {
        "latitude": 41.446333,
        "longitude": -90.565487,
        "state": "Illinois",
        "city": "Rock Island",
        "region_hub": "Central Midwest / Interior"
    },
    "The Other Factory": {
        "latitude": 35.117500,
        "longitude": -89.971107,
        "state": "Tennessee",
        "city": "Memphis",
        "region_hub": "Mid-South / Gulf Corridor"
    }
}

# 2. Baseline legacy assignment: which factory currently makes which product
PRODUCT_FACTORY_MAP = {
    # Chocolate Division
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar - Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    
    # Sugar Division
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    
    # Other Division
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory"
}

# 3. Approximate center coordinates for US States to map customer locations
# Used to calculate distance from factory to customer state
US_STATE_COORDINATES = {
    'Alabama': (32.806671, -86.791130),
    'Alaska': (61.370716, -152.404419),
    'Arizona': (33.729759, -111.431221),
    'Arkansas': (34.969704, -92.373123),
    'California': (36.116203, -119.681564),
    'Colorado': (39.059811, -105.311104),
    'Connecticut': (41.597782, -72.755371),
    'Delaware': (39.318523, -75.507141),
    'District of Columbia': (38.897438, -77.026817),
    'Florida': (27.766279, -81.686783),
    'Georgia': (33.040619, -83.643074),
    'Hawaii': (21.094318, -157.498337),
    'Idaho': (44.240459, -114.478828),
    'Illinois': (40.349457, -88.986137),
    'Indiana': (39.849426, -86.258278),
    'Iowa': (42.011539, -93.210526),
    'Kansas': (38.526600, -96.726486),
    'Kentucky': (37.668140, -84.670067),
    'Louisiana': (31.169546, -91.867805),
    'Maine': (44.693947, -69.381927),
    'Maryland': (39.063946, -76.802101),
    'Massachusetts': (42.230171, -71.530106),
    'Michigan': (43.326618, -84.536095),
    'Minnesota': (45.694454, -93.900192),
    'Mississippi': (32.741646, -89.678696),
    'Missouri': (38.456085, -92.288368),
    'Montana': (46.921925, -110.454353),
    'Nebraska': (41.125370, -98.268082),
    'Nevada': (38.313515, -117.055374),
    'New Hampshire': (43.452492, -71.563896),
    'New Jersey': (40.298904, -74.521011),
    'New Mexico': (34.840515, -106.248482),
    'New York': (42.165726, -74.948051),
    'North Carolina': (35.630066, -79.806419),
    'North Dakota': (47.528912, -99.784012),
    'Ohio': (40.388783, -82.764915),
    'Oklahoma': (35.565342, -96.928917),
    'Oregon': (44.572021, -122.070938),
    'Pennsylvania': (40.590752, -77.209755),
    'Rhode Island': (41.680893, -71.511780),
    'South Carolina': (33.856892, -80.945007),
    'South Dakota': (44.299782, -99.438828),
    'Tennessee': (35.747845, -86.692345),
    'Texas': (31.054487, -97.563461),
    'Utah': (40.150032, -111.862434),
    'Vermont': (44.045876, -72.710686),
    'Virginia': (37.769337, -78.169968),
    'Washington': (47.400902, -121.490494),
    'West Virginia': (38.491226, -80.954453),
    'Wisconsin': (44.268543, -89.616508),
    'Wyoming': (42.755966, -107.302490)
}

# Major US City coordinates to provide finer geographic resolution
US_MAJOR_CITIES = {
    'New York City': (40.7128, -74.0060),
    'Los Angeles': (34.0522, -118.2437),
    'Chicago': (41.8781, -87.6298),
    'Houston': (29.7604, -95.3698),
    'Phoenix': (33.4484, -112.0740),
    'Philadelphia': (39.9526, -75.1652),
    'San Antonio': (29.4241, -98.4936),
    'San Diego': (32.7157, -117.1611),
    'Dallas': (32.7767, -96.7970),
    'San Jose': (37.3382, -121.8863),
    'Austin': (30.2672, -97.7431),
    'Jacksonville': (30.3322, -81.6557),
    'San Francisco': (37.7749, -122.4194),
    'Columbus': (39.9612, -82.9988),
    'Fort Worth': (32.7555, -97.3308),
    'Indianapolis': (39.7684, -86.1581),
    'Charlotte': (35.2271, -80.8431),
    'Seattle': (47.6062, -122.3321),
    'Denver': (39.7392, -104.9903),
    'Washington': (38.9072, -77.0369),
    'Boston': (42.3601, -71.0589),
    'El Paso': (31.7619, -106.4850),
    'Nashville': (36.1627, -86.7816),
    'Detroit': (42.3314, -83.0458),
    'Oklahoma City': (35.4676, -97.5164),
    'Portland': (45.5152, -122.6784),
    'Las Vegas': (36.1699, -115.1398),
    'Memphis': (35.1495, -90.0490),
    'Louisville': (38.2527, -85.7585),
    'Baltimore': (39.2904, -76.6122),
    'Milwaukee': (43.0389, -87.9065),
    'Albuquerque': (35.0844, -106.6504),
    'Tucson': (32.2226, -110.9747),
    'Fresno': (36.7468, -119.7726),
    'Sacramento': (38.5816, -121.4944),
    'Mesa': (33.4152, -111.8315),
    'Atlanta': (33.7490, -84.3880),
    'Kansas City': (39.0997, -94.5786),
    'Colorado Springs': (38.8339, -104.8214),
    'Miami': (25.7617, -80.1918),
    'Raleigh': (35.7796, -78.6382),
    'Omaha': (41.2565, -95.9345),
    'Long Beach': (33.7701, -118.1937),
    'Virginia Beach': (36.8529, -75.9780),
    'Oakland': (37.8044, -122.2712),
    'Minneapolis': (44.9778, -93.2650),
    'Tulsa': (36.1540, -95.9928),
    'Tampa': (27.9506, -82.4572),
    'Arlington': (32.7357, -97.1081),
    'New Orleans': (29.9511, -90.0715)
}


def get_customer_coordinates(city, state):
    """
    Returns (latitude, longitude) for a customer based on city or state.
    If the city is found in US_MAJOR_CITIES, use it for higher precision;
    otherwise, fallback to state centroid.
    """
    if city in US_MAJOR_CITIES:
        return US_MAJOR_CITIES[city]
    if state in US_STATE_COORDINATES:
        return US_STATE_COORDINATES[state]
    # Default center of continental US if state is unknown
    return (39.8283, -98.5795)


def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points on Earth
    using the Haversine formula.
    
    Parameters:
        lat1, lon1: Coordinates of Point 1 (Origin Factory)
        lat2, lon2: Coordinates of Point 2 (Customer Destination)
        
    Returns:
        Distance in miles (rounded to 2 decimal places).
    """
    # Earth radius in miles
    earth_radius_miles = 3958.8

    # Convert degrees to radians
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    r_lat1 = math.radians(lat1)
    r_lat2 = math.radians(lat2)

    # Haversine formula
    a = (math.sin(d_lat / 2) ** 2 +
         math.cos(r_lat1) * math.cos(r_lat2) * math.sin(d_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance_miles = earth_radius_miles * c
    return round(distance_miles, 2)


def get_distance_to_factory(factory_name, customer_lat, customer_lon):
    """
    Computes distance from a specific factory to a customer location.
    """
    if factory_name not in FACTORIES:
        raise ValueError(f"Unknown factory: {factory_name}")
    
    f_lat = FACTORIES[factory_name]["latitude"]
    f_lon = FACTORIES[factory_name]["longitude"]
    return calculate_haversine_distance(f_lat, f_lon, customer_lat, customer_lon)
