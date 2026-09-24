import json
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

landmarks = [
    # ── Italy & Southern Europe ──
    {"name": "Leaning Tower of Pisa", "city": "Pisa", "country": "Italy", "region": "Europe"},
    {"name": "Colosseum", "city": "Rome", "country": "Italy", "region": "Europe"},
    {"name": "Pantheon", "city": "Rome", "country": "Italy", "region": "Europe"},
    {"name": "Trevi Fountain", "city": "Rome", "country": "Italy", "region": "Europe"},
    {"name": "Florence Cathedral (Duomo)", "city": "Florence", "country": "Italy", "region": "Europe"},
    {"name": "St. Mark's Basilica", "city": "Venice", "country": "Italy", "region": "Europe"},
    {"name": "Milan Cathedral (Duomo di Milano)", "city": "Milan", "country": "Italy", "region": "Europe"},
    {"name": "Ponte Vecchio", "city": "Florence", "country": "Italy", "region": "Europe"},
    {"name": "Grand Canal", "city": "Venice", "country": "Italy", "region": "Europe"},
    {"name": "Rialto Bridge", "city": "Venice", "country": "Italy", "region": "Europe"},
    {"name": "Roman Forum", "city": "Rome", "country": "Italy", "region": "Europe"},
    {"name": "Pompeii Archaeological Ruins", "city": "Pompeii", "country": "Italy", "region": "Europe"},
    {"name": "St. Peter's Basilica", "city": "Vatican City", "country": "Vatican City", "region": "Europe"},
    {"name": "Sistine Chapel", "city": "Vatican City", "country": "Vatican City", "region": "Europe"},
    {"name": "Sagrada Familia", "city": "Barcelona", "country": "Spain", "region": "Europe"},
    {"name": "Park Guell", "city": "Barcelona", "country": "Spain", "region": "Europe"},
    {"name": "Alhambra", "city": "Granada", "country": "Spain", "region": "Europe"},
    {"name": "Royal Palace of Madrid", "city": "Madrid", "country": "Spain", "region": "Europe"},
    {"name": "Plaza de Espana", "city": "Seville", "country": "Spain", "region": "Europe"},
    {"name": "Seville Cathedral and Giralda", "city": "Seville", "country": "Spain", "region": "Europe"},
    {"name": "Pena Palace", "city": "Sintra", "country": "Portugal", "region": "Europe"},
    {"name": "Belem Tower", "city": "Lisbon", "country": "Portugal", "region": "Europe"},
    {"name": "Parthenon / Acropolis of Athens", "city": "Athens", "country": "Greece", "region": "Europe"},
    {"name": "Santorini Caldera & White Villages", "city": "Santorini", "country": "Greece", "region": "Europe"},
    {"name": "Meteora Monasteries", "city": "Kalabaka", "country": "Greece", "region": "Europe"},
    {"name": "Dubrovnik Old Town Walls", "city": "Dubrovnik", "country": "Croatia", "region": "Europe"},

    # ── Western & Northern Europe ──
    {"name": "Eiffel Tower", "city": "Paris", "country": "France", "region": "Europe"},
    {"name": "Louvre Museum", "city": "Paris", "country": "France", "region": "Europe"},
    {"name": "Notre-Dame Cathedral", "city": "Paris", "country": "France", "region": "Europe"},
    {"name": "Arc de Triomphe", "city": "Paris", "country": "France", "region": "Europe"},
    {"name": "Palace of Versailles", "city": "Versailles", "country": "France", "region": "Europe"},
    {"name": "Mont Saint-Michel", "city": "Normandy", "country": "France", "region": "Europe"},
    {"name": "Sacre-Coeur", "city": "Paris", "country": "France", "region": "Europe"},
    {"name": "Sainte-Chapelle", "city": "Paris", "country": "France", "region": "Europe"},
    {"name": "Big Ben / Elizabeth Tower", "city": "London", "country": "United Kingdom", "region": "Europe"},
    {"name": "Tower Bridge", "city": "London", "country": "United Kingdom", "region": "Europe"},
    {"name": "Tower of London", "city": "London", "country": "United Kingdom", "region": "Europe"},
    {"name": "Stonehenge", "city": "Wiltshire", "country": "United Kingdom", "region": "Europe"},
    {"name": "London Eye", "city": "London", "country": "United Kingdom", "region": "Europe"},
    {"name": "Buckingham Palace", "city": "London", "country": "United Kingdom", "region": "Europe"},
    {"name": "Westminster Abbey", "city": "London", "country": "United Kingdom", "region": "Europe"},
    {"name": "Houses of Parliament", "city": "London", "country": "United Kingdom", "region": "Europe"},
    {"name": "Edinburgh Castle", "city": "Edinburgh", "country": "United Kingdom", "region": "Europe"},
    {"name": "Cliffs of Moher", "city": "County Clare", "country": "Ireland", "region": "Europe"},
    {"name": "Giant's Causeway", "city": "Bushmills", "country": "United Kingdom", "region": "Europe"},
    {"name": "Brandenburg Gate", "city": "Berlin", "country": "Germany", "region": "Europe"},
    {"name": "Neuschwanstein Castle", "city": "Bavaria", "country": "Germany", "region": "Europe"},
    {"name": "Cologne Cathedral", "city": "Cologne", "country": "Germany", "region": "Europe"},
    {"name": "Reichstag Building", "city": "Berlin", "country": "Germany", "region": "Europe"},
    {"name": "Rijksmuseum", "city": "Amsterdam", "country": "Netherlands", "region": "Europe"},
    {"name": "Atomium", "city": "Brussels", "country": "Belgium", "region": "Europe"},
    {"name": "Matterhorn", "city": "Zermatt", "country": "Switzerland", "region": "Europe"},
    {"name": "Charles Bridge", "city": "Prague", "country": "Czech Republic", "region": "Europe"},
    {"name": "Prague Castle", "city": "Prague", "country": "Czech Republic", "region": "Europe"},
    {"name": "Hungarian Parliament Building", "city": "Budapest", "country": "Hungary", "region": "Europe"},
    {"name": "Bran Castle (Dracula's Castle)", "city": "Bran", "country": "Romania", "region": "Europe"},
    {"name": "Saint Basil's Cathedral", "city": "Moscow", "country": "Russia", "region": "Europe"},
    {"name": "Moscow Kremlin", "city": "Moscow", "country": "Russia", "region": "Europe"},
    {"name": "Hermitage Museum / Winter Palace", "city": "Saint Petersburg", "country": "Russia", "region": "Europe"},
    {"name": "Church of the Savior on Spilled Blood", "city": "Saint Petersburg", "country": "Russia", "region": "Europe"},

    # ── India (Comprehensive Heritage & Culture) ──
    {"name": "Taj Mahal", "city": "Agra", "country": "India", "region": "Asia"},
    {"name": "Red Fort", "city": "Delhi", "country": "India", "region": "Asia"},
    {"name": "India Gate", "city": "New Delhi", "country": "India", "region": "Asia"},
    {"name": "Qutub Minar", "city": "Delhi", "country": "India", "region": "Asia"},
    {"name": "Humayun's Tomb", "city": "Delhi", "country": "India", "region": "Asia"},
    {"name": "Lotus Temple", "city": "Delhi", "country": "India", "region": "Asia"},
    {"name": "Akshardham Temple", "city": "New Delhi", "country": "India", "region": "Asia"},
    {"name": "Rashtrapati Bhavan", "city": "New Delhi", "country": "India", "region": "Asia"},
    {"name": "Jama Masjid", "city": "Delhi", "country": "India", "region": "Asia"},
    {"name": "Gateway of India", "city": "Mumbai", "country": "India", "region": "Asia"},
    {"name": "Chhatrapati Shivaji Maharaj Terminus", "city": "Mumbai", "country": "India", "region": "Asia"},
    {"name": "Elephanta Caves", "city": "Mumbai", "country": "India", "region": "Asia"},
    {"name": "Ajanta and Ellora Caves", "city": "Aurangabad", "country": "India", "region": "Asia"},
    {"name": "Shaniwar Wada", "city": "Pune", "country": "India", "region": "Asia"},
    {"name": "Golden Temple (Harmandir Sahib)", "city": "Amritsar", "country": "India", "region": "Asia"},
    {"name": "Hawa Mahal", "city": "Jaipur", "country": "India", "region": "Asia"},
    {"name": "Amer Fort (Amber Palace)", "city": "Jaipur", "country": "India", "region": "Asia"},
    {"name": "City Palace Jaipur", "city": "Jaipur", "country": "India", "region": "Asia"},
    {"name": "Jantar Mantar Jaipur", "city": "Jaipur", "country": "India", "region": "Asia"},
    {"name": "Mehrangarh Fort", "city": "Jodhpur", "country": "India", "region": "Asia"},
    {"name": "Umaid Bhawan Palace", "city": "Jodhpur", "country": "India", "region": "Asia"},
    {"name": "Jaisalmer Fort (Sonar Qila)", "city": "Jaisalmer", "country": "India", "region": "Asia"},
    {"name": "City Palace Udaipur & Lake Pichola", "city": "Udaipur", "country": "India", "region": "Asia"},
    {"name": "Chittorgarh Fort", "city": "Chittorgarh", "country": "India", "region": "Asia"},
    {"name": "Ranakpur Jain Temple", "city": "Ranakpur", "country": "India", "region": "Asia"},
    {"name": "Chand Baori Stepwell", "city": "Abhaneri", "country": "India", "region": "Asia"},
    {"name": "Fatehpur Sikri & Buland Darwaza", "city": "Agra", "country": "India", "region": "Asia"},
    {"name": "Kashi Vishwanath Temple", "city": "Varanasi", "country": "India", "region": "Asia"},
    {"name": "Varanasi Ghats & Ganges River", "city": "Varanasi", "country": "India", "region": "Asia"},
    {"name": "Khajuraho Group of Monuments", "city": "Khajuraho", "country": "India", "region": "Asia"},
    {"name": "Sanchi Stupa", "city": "Sanchi", "country": "India", "region": "Asia"},
    {"name": "Gwalior Fort", "city": "Gwalior", "country": "India", "region": "Asia"},
    {"name": "Victoria Memorial", "city": "Kolkata", "country": "India", "region": "Asia"},
    {"name": "Howrah Bridge", "city": "Kolkata", "country": "India", "region": "Asia"},
    {"name": "Konark Sun Temple", "city": "Puri / Konark", "country": "India", "region": "Asia"},
    {"name": "Jagannath Temple", "city": "Puri", "country": "India", "region": "Asia"},
    {"name": "Statue of Unity", "city": "Kevadia", "country": "India", "region": "Asia"},
    {"name": "Somnath Temple", "city": "Prabhas Patan", "country": "India", "region": "Asia"},
    {"name": "Dwarkadhish Temple", "city": "Dwarka", "country": "India", "region": "Asia"},
    {"name": "Sun Temple Modhera", "city": "Modhera", "country": "India", "region": "Asia"},
    {"name": "Rani ki Vav (Queen's Stepwell)", "city": "Patan", "country": "India", "region": "Asia"},
    {"name": "Charminar", "city": "Hyderabad", "country": "India", "region": "Asia"},
    {"name": "Golconda Fort", "city": "Hyderabad", "country": "India", "region": "Asia"},
    {"name": "Ramappa Temple", "city": "Warangal", "country": "India", "region": "Asia"},
    {"name": "Thousand Pillar Temple", "city": "Warangal", "country": "India", "region": "Asia"},
    {"name": "Warangal Fort", "city": "Warangal", "country": "India", "region": "Asia"},
    {"name": "Virupaksha Temple", "city": "Hampi", "country": "India", "region": "Asia"},
    {"name": "Stone Chariot & Vittala Temple", "city": "Hampi", "country": "India", "region": "Asia"},
    {"name": "Mysore Palace (Amba Vilas)", "city": "Mysore", "country": "India", "region": "Asia"},
    {"name": "Gol Gumbaz", "city": "Bijapur", "country": "India", "region": "Asia"},
    {"name": "Chennakeshava Temple Belur", "city": "Belur", "country": "India", "region": "Asia"},
    {"name": "Hoysaleswara Temple Halebidu", "city": "Halebidu", "country": "India", "region": "Asia"},
    {"name": "Brihadisvara Temple (Big Temple)", "city": "Thanjavur", "country": "India", "region": "Asia"},
    {"name": "Meenakshi Amman Temple", "city": "Madurai", "country": "India", "region": "Asia"},
    {"name": "Shore Temple & Pancha Rathas", "city": "Mahabalipuram", "country": "India", "region": "Asia"},
    {"name": "Ramanathaswamy Temple", "city": "Rameswaram", "country": "India", "region": "Asia"},
    {"name": "Padmanabhaswamy Temple", "city": "Thiruvananthapuram", "country": "India", "region": "Asia"},
    {"name": "Basilica of Bom Jesus", "city": "Old Goa", "country": "India", "region": "Asia"},
    {"name": "Tirumala Venkateswara Temple", "city": "Tirupati", "country": "India", "region": "Asia"},
    {"name": "Lepakshi Veerabhadra Temple", "city": "Lepakshi", "country": "India", "region": "Asia"},
    {"name": "Mahabodhi Temple", "city": "Bodh Gaya", "country": "India", "region": "Asia"},
    {"name": "Nalanda Mahavihara Ruins", "city": "Nalanda", "country": "India", "region": "Asia"},
    {"name": "Kamakhya Temple", "city": "Guwahati", "country": "India", "region": "Asia"},
    {"name": "Cellular Jail", "city": "Port Blair", "country": "India", "region": "Asia"},

    # ── East & Southeast Asia ──
    {"name": "Great Wall of China", "city": "Beijing", "country": "China", "region": "Asia"},
    {"name": "Forbidden City", "city": "Beijing", "country": "China", "region": "Asia"},
    {"name": "Terracotta Army", "city": "Xi'an", "country": "China", "region": "Asia"},
    {"name": "Temple of Heaven", "city": "Beijing", "country": "China", "region": "Asia"},
    {"name": "Potala Palace", "city": "Lhasa", "country": "China", "region": "Asia"},
    {"name": "Oriental Pearl Tower", "city": "Shanghai", "country": "China", "region": "Asia"},
    {"name": "Mount Fuji", "city": "Honshu", "country": "Japan", "region": "Asia"},
    {"name": "Fushimi Inari Shrine", "city": "Kyoto", "country": "Japan", "region": "Asia"},
    {"name": "Senso-ji Temple", "city": "Tokyo", "country": "Japan", "region": "Asia"},
    {"name": "Kinkaku-ji (Golden Pavilion)", "city": "Kyoto", "country": "Japan", "region": "Asia"},
    {"name": "Himeji Castle", "city": "Himeji", "country": "Japan", "region": "Asia"},
    {"name": "Tokyo Tower", "city": "Tokyo", "country": "Japan", "region": "Asia"},
    {"name": "Itsukushima Floating Torii Gate", "city": "Miyajima", "country": "Japan", "region": "Asia"},
    {"name": "Gyeongbokgung Palace", "city": "Seoul", "country": "South Korea", "region": "Asia"},
    {"name": "N Seoul Tower", "city": "Seoul", "country": "South Korea", "region": "Asia"},
    {"name": "Taipei 101", "city": "Taipei", "country": "Taiwan", "region": "Asia"},
    {"name": "Angkor Wat", "city": "Siem Reap", "country": "Cambodia", "region": "Asia"},
    {"name": "Bayon Temple", "city": "Siem Reap", "country": "Cambodia", "region": "Asia"},
    {"name": "Petronas Twin Towers", "city": "Kuala Lumpur", "country": "Malaysia", "region": "Asia"},
    {"name": "Batu Caves", "city": "Selangor", "country": "Malaysia", "region": "Asia"},
    {"name": "Marina Bay Sands", "city": "Singapore", "country": "Singapore", "region": "Asia"},
    {"name": "Gardens by the Bay", "city": "Singapore", "country": "Singapore", "region": "Asia"},
    {"name": "Borobudur Temple", "city": "Java", "country": "Indonesia", "region": "Asia"},
    {"name": "Prambanan Temple", "city": "Java", "country": "Indonesia", "region": "Asia"},
    {"name": "Wat Arun (Temple of Dawn)", "city": "Bangkok", "country": "Thailand", "region": "Asia"},
    {"name": "Grand Palace Bangkok", "city": "Bangkok", "country": "Thailand", "region": "Asia"},
    {"name": "Wat Phra Kaew (Temple of Emerald Buddha)", "city": "Bangkok", "country": "Thailand", "region": "Asia"},
    {"name": "Bagan Pagodas", "city": "Mandalay", "country": "Myanmar", "region": "Asia"},
    {"name": "Shwedagon Pagoda", "city": "Yangon", "country": "Myanmar", "region": "Asia"},
    {"name": "Ha Long Bay", "city": "Quang Ninh", "country": "Vietnam", "region": "Asia"},
    {"name": "Sigiriya Ancient Rock Fortress", "city": "Sigiriya", "country": "Sri Lanka", "region": "Asia"},
    {"name": "Temple of the Sacred Tooth Relic", "city": "Kandy", "country": "Sri Lanka", "region": "Asia"},
    {"name": "Boudhanath Stupa", "city": "Kathmandu", "country": "Nepal", "region": "Asia"},
    {"name": "Tiger's Nest Monastery (Paro Taktsang)", "city": "Paro", "country": "Bhutan", "region": "Asia"},

    # ── Middle East ──
    {"name": "Burj Khalifa", "city": "Dubai", "country": "United Arab Emirates", "region": "Middle East"},
    {"name": "Sheikh Zayed Grand Mosque", "city": "Abu Dhabi", "country": "United Arab Emirates", "region": "Middle East"},
    {"name": "Burj Al Arab", "city": "Dubai", "country": "United Arab Emirates", "region": "Middle East"},
    {"name": "Museum of the Future", "city": "Dubai", "country": "United Arab Emirates", "region": "Middle East"},
    {"name": "Petra (The Treasury / Al-Khazneh)", "city": "Ma'an", "country": "Jordan", "region": "Middle East"},
    {"name": "Hagia Sophia", "city": "Istanbul", "country": "Turkey", "region": "Middle East"},
    {"name": "Blue Mosque (Sultan Ahmed Mosque)", "city": "Istanbul", "country": "Turkey", "region": "Middle East"},
    {"name": "Topkapi Palace", "city": "Istanbul", "country": "Turkey", "region": "Middle East"},
    {"name": "Cappadocia Fairy Chimneys", "city": "Anatolia", "country": "Turkey", "region": "Middle East"},
    {"name": "Western Wall", "city": "Jerusalem", "country": "Israel", "region": "Middle East"},
    {"name": "Dome of the Rock", "city": "Jerusalem", "country": "Israel", "region": "Middle East"},

    # ── North America ──
    {"name": "Statue of Liberty", "city": "New York", "country": "United States", "region": "North America"},
    {"name": "Golden Gate Bridge", "city": "San Francisco", "country": "United States", "region": "North America"},
    {"name": "Empire State Building", "city": "New York", "country": "United States", "region": "North America"},
    {"name": "One World Trade Center", "city": "New York", "country": "United States", "region": "North America"},
    {"name": "Brooklyn Bridge", "city": "New York", "country": "United States", "region": "North America"},
    {"name": "Times Square", "city": "New York", "country": "United States", "region": "North America"},
    {"name": "Mount Rushmore", "city": "South Dakota", "country": "United States", "region": "North America"},
    {"name": "The White House", "city": "Washington, D.C.", "country": "United States", "region": "North America"},
    {"name": "United States Capitol", "city": "Washington, D.C.", "country": "United States", "region": "North America"},
    {"name": "Lincoln Memorial", "city": "Washington, D.C.", "country": "United States", "region": "North America"},
    {"name": "Grand Canyon", "city": "Arizona", "country": "United States", "region": "North America"},
    {"name": "Antelope Canyon", "city": "Arizona", "country": "United States", "region": "North America"},
    {"name": "Horseshoe Bend", "city": "Arizona", "country": "United States", "region": "North America"},
    {"name": "Space Needle", "city": "Seattle", "country": "United States", "region": "North America"},
    {"name": "Alcatraz Island", "city": "San Francisco", "country": "United States", "region": "North America"},
    {"name": "Hollywood Sign", "city": "Los Angeles", "country": "United States", "region": "North America"},
    {"name": "Gateway Arch", "city": "St. Louis", "country": "United States", "region": "North America"},
    {"name": "Cloud Gate (The Bean)", "city": "Chicago", "country": "United States", "region": "North America"},
    {"name": "Niagara Falls", "city": "Ontario / New York", "country": "Canada / US", "region": "North America"},
    {"name": "CN Tower", "city": "Toronto", "country": "Canada", "region": "North America"},
    {"name": "Chateau Frontenac", "city": "Quebec City", "country": "Canada", "region": "North America"},
    {"name": "Banff National Park & Lake Louise", "city": "Alberta", "country": "Canada", "region": "North America"},
    {"name": "Moraine Lake", "city": "Alberta", "country": "Canada", "region": "North America"},
    {"name": "Chichen Itza (El Castillo)", "city": "Yucatan", "country": "Mexico", "region": "North America"},
    {"name": "Teotihuacan Pyramids (Pyramid of the Sun)", "city": "Mexico State", "country": "Mexico", "region": "North America"},

    # ── South America ──
    {"name": "Christ the Redeemer", "city": "Rio de Janeiro", "country": "Brazil", "region": "South America"},
    {"name": "Sugarloaf Mountain", "city": "Rio de Janeiro", "country": "Brazil", "region": "South America"},
    {"name": "Machu Picchu", "city": "Cusco", "country": "Peru", "region": "South America"},
    {"name": "Iguazu Falls", "city": "Misiones / Parana", "country": "Argentina / Brazil", "region": "South America"},
    {"name": "Easter Island Moai Statues", "city": "Easter Island", "country": "Chile", "region": "South America"},
    {"name": "Salar de Uyuni Salt Flats", "city": "Uyuni", "country": "Bolivia", "region": "South America"},
    {"name": "Perito Moreno Glacier", "city": "Santa Cruz", "country": "Argentina", "region": "South America"},

    # ── Africa ──
    {"name": "Great Pyramids of Giza", "city": "Cairo", "country": "Egypt", "region": "Africa"},
    {"name": "Great Sphinx of Giza", "city": "Giza", "country": "Egypt", "region": "Africa"},
    {"name": "Karnak Temple", "city": "Luxor", "country": "Egypt", "region": "Africa"},
    {"name": "Valley of the Kings", "city": "Luxor", "country": "Egypt", "region": "Africa"},
    {"name": "Abu Simbel Temples", "city": "Aswan", "country": "Egypt", "region": "Africa"},
    {"name": "Table Mountain", "city": "Cape Town", "country": "South Africa", "region": "Africa"},
    {"name": "Cape of Good Hope", "city": "Cape Town", "country": "South Africa", "region": "Africa"},
    {"name": "Victoria Falls", "city": "Livingstone", "country": "Zambia / Zimbabwe", "region": "Africa"},
    {"name": "Mount Kilimanjaro", "city": "Kilimanjaro", "country": "Tanzania", "region": "Africa"},
    {"name": "Djemaa el-Fna", "city": "Marrakech", "country": "Morocco", "region": "Africa"},
    {"name": "Hassan II Mosque", "city": "Casablanca", "country": "Morocco", "region": "Africa"},

    # ── Oceania ──
    {"name": "Sydney Opera House", "city": "Sydney", "country": "Australia", "region": "Oceania"},
    {"name": "Sydney Harbour Bridge", "city": "Sydney", "country": "Australia", "region": "Oceania"},
    {"name": "Uluru (Ayers Rock)", "city": "Northern Territory", "country": "Australia", "region": "Oceania"},
    {"name": "Great Barrier Reef", "city": "Queensland", "country": "Australia", "region": "Oceania"},
    {"name": "Twelve Apostles", "city": "Victoria", "country": "Australia", "region": "Oceania"},
    {"name": "Milford Sound", "city": "Fiordland", "country": "New Zealand", "region": "Oceania"},
    {"name": "Hobbiton Movie Set", "city": "Matamata", "country": "New Zealand", "region": "Oceania"},
]

print(f"Total landmarks in database: {len(landmarks)}")
with open('models/clip/landmarks_db.json', 'w', encoding='utf-8') as f:
    json.dump(landmarks, f, ensure_ascii=False, indent=2)

# Compute text embeddings for all landmarks
tokenizer = Tokenizer.from_file('models/clip/tokenizer.json')
tokenizer.enable_padding(length=77, pad_id=0, pad_token="<|endoftext|>")
tokenizer.enable_truncation(max_length=77)

session = ort.InferenceSession('models/clip/model_quantized.onnx')

prompts = [f"a photo of {lm['name']}, a famous landmark in {lm['city']}, {lm['country']}" for lm in landmarks]
encodings = tokenizer.encode_batch(prompts)
input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

dummy_pixel = np.zeros((1, 3, 224, 224), dtype=np.float32)

print("Computing pre-baked landmark text embeddings...")
text_embeds_list = []
batch_size = 32
for i in range(0, len(prompts), batch_size):
    batch_ids = input_ids[i:i+batch_size]
    batch_mask = attention_mask[i:i+batch_size]
    out = session.run(['text_embeds'], {
        'input_ids': batch_ids,
        'attention_mask': batch_mask,
        'pixel_values': dummy_pixel
    })
    embeds = out[0]
    # Normalize embeddings
    embeds = embeds / np.linalg.norm(embeds, axis=-1, keepdims=True)
    text_embeds_list.append(embeds)

all_text_embeds = np.concatenate(text_embeds_list, axis=0)
print("Computed embeddings shape:", all_text_embeds.shape)
np.save('models/clip/landmarks_embeddings.npy', all_text_embeds)
print("Saved models/clip/landmarks_embeddings.npy successfully.")
