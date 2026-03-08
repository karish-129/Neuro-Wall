import pandas as pd
import random
import urllib.parse

# --- CONFIGURATION ---
OUTPUT_FILE = "mixed_security_dataset.csv" # Overwrites the old one
NUM_SAMPLES = 20000 # How many rows to generate

# --- 1. THE ATTACK SEED (The "DNA") ---
base_attacks = [
    "UNION SELECT * FROM users",
    "OR 1=1",
    "DROP TABLE users",
    "<script>alert('XSS')</script>",
    "javascript:alert(1)",
    "/etc/passwd",
    "| cat /etc/passwd",
    "&& whoami",
    "exec(chr(112))",
    "SELECT password FROM admins"
]

# --- 2. THE MUTATORS (The "Sophistication") ---
def mutate_sql_case(payload):
    # Changes "SELECT" to "SeLeCt"
    return "".join(random.choice([c.upper(), c.lower()]) for c in payload)

def mutate_url_encode(payload):
    # Changes "SELECT" to "%53%45%4c%45%43%54"
    return urllib.parse.quote(payload)

def mutate_sql_comments(payload):
    # Changes "UNION SELECT" to "UNION/**/SELECT"
    return payload.replace(" ", "/**/")

def mutate_double_encode(payload):
    # Changes "<script>" to "%253Cscript%253E"
    return urllib.parse.quote(urllib.parse.quote(payload))

def no_mutation(payload):
    return payload

mutators = [mutate_sql_case, mutate_url_encode, mutate_sql_comments, mutate_double_encode, no_mutation]

# --- 3. THE CONTEXT WRAPPERS (The "Disguise") ---
def wrap_json(payload):
    return (f"POST /api/login HTTP/1.1\nContent-Type: application/json\n\n"
            f'{{"user": "admin", "pass": "{payload}"}}')

def wrap_query(payload):
    return (f"GET /search?q={payload} HTTP/1.1\nHost: example.com\n"
            f"User-Agent: Mozilla/5.0")

def wrap_cookie(payload):
    return (f"GET /dashboard HTTP/1.1\nHost: example.com\n"
            f"Cookie: session_id={payload}")

wrappers = [wrap_json, wrap_query, wrap_cookie]

# --- 4. BENIGN TRAFFIC (The "Good Guys") ---
benign_words = ["search", "iphone", "login", "dashboard", "12345", "user_id", "admin", "contact"]
def generate_benign():
    word = random.choice(benign_words)
    wrapper = random.choice(wrappers)
    return wrapper(word)

# --- 5. GENERATE ---
print(f"🚀 Generating {NUM_SAMPLES} sophisticated requests...")
data = []

for _ in range(NUM_SAMPLES // 2):
    # Generate 50% Attacks
    attack = random.choice(base_attacks)
    mutator = random.choice(mutators)
    wrapper = random.choice(wrappers)
    
    sophisticated_attack = wrapper(mutator(attack))
    data.append({"text": sophisticated_attack, "label": 1})

for _ in range(NUM_SAMPLES // 2):
    # Generate 50% Safe Traffic
    data.append({"text": generate_benign(), "label": 0})

# --- SAVE ---
df = pd.DataFrame(data)
# Shuffle the data
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Success! Saved to {OUTPUT_FILE}")