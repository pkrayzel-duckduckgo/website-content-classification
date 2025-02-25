from sentence_transformers import SentenceTransformer
import random
import pandas as pd

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# Force CPU execution
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
torch.set_default_device("cpu")

gpt_model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(gpt_model_name)
model = AutoModelForCausalLM.from_pretrained(gpt_model_name)
# Initialize the paraphraser model
sentence_paraphraser = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Ensure padding token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

data_sources = [
    {"title": "How to change your Xbox gamertag", "url": "https://support.xbox.com/en-US/help/account-profile/profile/change-xbox-live-gamertag", "keywords": ["Xbox", "gamertag", "change", "profile", "account"]},
    {"title": "Guide to baking the perfect sourdough bread", "url": "https://www.foodblog.com/sourdough-guide", "keywords": ["baking", "bread", "sourdough", "recipe", "fermentation"]},
    {"title": "Understanding the stock market for beginners", "url": "https://www.investopedia.com/stock-market-basics", "keywords": ["stock", "market", "investing", "finance", "shares"]},
    {"title": "The ultimate travel checklist", "url": "https://www.travelchecklist.com", "keywords": ["travel", "checklist", "packing", "trip", "adventure"]},
    {"title": "How to install Python on Mac", "url": "https://www.python.org/install-mac", "keywords": ["Python", "install", "Mac", "development", "programming"]},
    {"title": "Best hiking trails in Europe", "url": "https://www.hikingworld.com/top10-europe", "keywords": ["hiking", "trails", "mountains", "nature", "adventure"]},
    {"title": "Machine learning basics for beginners", "url": "https://www.mlbasics.com/intro", "keywords": ["machine learning", "AI", "model", "training", "algorithm"]},
    {"title": "Web scraping techniques and tools", "url": "https://www.webscraping101.com", "keywords": ["web scraping", "tools", "Python", "data extraction", "parsing"]},
    {"title": "Dog training tips for new owners", "url": "https://www.dogtraining.com/tips", "keywords": ["dog training", "obedience", "behavior", "treats", "commands"]},
    {"title": "The history of electric vehicles", "url": "https://www.evhistory.com", "keywords": ["electric vehicles", "EV", "battery", "charging", "transport"]},
    {"title": "How to grow tomatoes at home", "url": "https://www.gardeningguide.com/tomatoes", "keywords": ["gardening", "tomatoes", "soil", "watering", "harvest"]},
    {"title": "Introduction to blockchain technology", "url": "https://www.crypto101.com/blockchain", "keywords": ["blockchain", "crypto", "decentralized", "ledger", "mining"]},
    {"title": "Top 10 video games of all time", "url": "https://www.gamershub.com/top10", "keywords": ["video games", "gaming", "console", "PC", "multiplayer"]},
    {"title": "Effective email marketing strategies", "url": "https://www.marketingpro.com/email-strategies", "keywords": ["email marketing", "campaign", "newsletter", "open rate", "engagement"]},
    {"title": "The art of minimalist living", "url": "https://www.simpleliving.com/minimalism", "keywords": ["minimalism", "declutter", "simple", "mindset", "lifestyle"]},
    {"title": "How to build a treehouse", "url": "https://www.diyprojects.com/treehouse", "keywords": ["treehouse", "DIY", "construction", "wood", "design"]},
    {"title": "Learning JavaScript for beginners", "url": "https://www.learnjavascript.com/beginners", "keywords": ["JavaScript", "programming", "web development", "frontend", "ES6"]},
    {"title": "Yoga poses for stress relief", "url": "https://www.yogajourney.com/poses", "keywords": ["yoga", "stress relief", "mindfulness", "stretching", "breathing"]},
    {"title": "How to start a small business", "url": "https://www.businessstarter.com/guide", "keywords": ["business", "startup", "entrepreneur", "funding", "marketing"]},
    {"title": "Top 5 productivity techniques", "url": "https://www.productivityhub.com/techniques", "keywords": ["productivity", "focus", "time management", "Pomodoro", "deep work"]},
    {"title": "Understanding climate change", "url": "https://www.sciencefacts.com/climate", "keywords": ["climate change", "warming", "emissions", "carbon", "environment"]},
    {"title": "How to play chess like a pro", "url": "https://www.chessmasters.com/guide", "keywords": ["chess", "strategy", "openings", "tactics", "checkmate"]},
    {"title": "The basics of healthy eating", "url": "https://www.nutritionworld.com/healthy-eating", "keywords": ["healthy eating", "diet", "nutrition", "vitamins", "macros"]},
    {"title": "How to set up a home office", "url": "https://www.remotehub.com/home-office", "keywords": ["home office", "workspace", "productivity", "remote work", "equipment"]},
    {"title": "Beginner's guide to photography", "url": "https://www.photogenius.com/guide", "keywords": ["photography", "camera", "aperture", "ISO", "composition"]},
    {"title": "How to repair a leaky faucet", "url": "https://www.diyplumbing.com/faucet-repair", "keywords": ["plumbing", "leaky faucet", "DIY", "tools", "repair"]},
    {"title": "Introduction to 3D printing", "url": "https://www.printingworld.com/3d-printing", "keywords": ["3D printing", "filament", "printer", "models", "design"]},
    {"title": "How to plan a budget-friendly vacation", "url": "https://www.travelplanner.com/budget-vacation", "keywords": ["vacation", "budget", "travel", "savings", "trip"]},
    {"title": "Exploring the deep sea", "url": "https://www.marineworld.com/deep-sea", "keywords": ["deep sea", "ocean", "marine life", "exploration", "diving"]},
    {"title": "Tips for writing a novel", "url": "https://www.writersguide.com/novel-tips", "keywords": ["writing", "novel", "plot", "characters", "editing"]},
    {"title": "How to meditate effectively", "url": "https://www.mindfulme.com/meditation-guide", "keywords": ["meditation", "mindfulness", "focus", "breathing", "relaxation"]},
    {"title": "Learning Spanish for beginners", "url": "https://www.language101.com/spanish", "keywords": ["Spanish", "language", "grammar", "vocabulary", "learning"]},
    {"title": "Understanding cryptocurrency basics", "url": "https://www.coinworld.com/crypto-basics", "keywords": ["cryptocurrency", "Bitcoin", "Ethereum", "trading", "blockchain"]},
    {"title": "The history of aviation", "url": "https://www.aerojournal.com/history", "keywords": ["aviation", "planes", "flight", "pioneers", "aircraft"]},
    {"title": "Tips for successful gardening", "url": "https://www.greenfingers.com/gardening-tips", "keywords": ["gardening", "plants", "soil", "watering", "fertilizer"]},
    {"title": "How to build a mobile app", "url": "https://www.apptutorials.com/build-app", "keywords": ["mobile app", "development", "iOS", "Android", "coding"]},
    {"title": "Understanding quantum physics", "url": "https://www.sciencemaster.com/quantum", "keywords": ["quantum physics", "particles", "wave function", "theory", "experiments"]},
    {"title": "How to start a YouTube channel", "url": "https://www.contentcreators.com/youtube-guide", "keywords": ["YouTube", "channel", "video", "content", "audience"]},
    {"title": "Guide to brewing craft beer", "url": "https://www.brewmasters.com/craft-beer", "keywords": ["beer", "brewing", "hops", "yeast", "fermentation"]},
    {"title": "The evolution of the internet", "url": "https://www.webworld.com/internet-history", "keywords": ["internet", "web", "protocols", "history", "technology"]},
    {"title": "How to create digital art", "url": "https://www.digitalartworld.com/guide", "keywords": ["digital art", "drawing", "software", "tablet", "creativity"]},
    {"title": "Tips for marathon training", "url": "https://www.runnerlife.com/marathon-tips", "keywords": ["marathon", "running", "endurance", "training", "schedule"]},
    {"title": "Basics of investing in real estate", "url": "https://www.reinvesting.com/basics", "keywords": ["real estate", "investment", "property", "income", "market"]},
    {"title": "How to bake a chocolate cake", "url": "https://www.sweettreats.com/chocolate-cake", "keywords": ["chocolate cake", "baking", "recipe", "oven", "ingredients"]},
    {"title": "Exploring ancient Egypt", "url": "https://www.historytrips.com/egypt", "keywords": ["Egypt", "pharaohs", "pyramids", "hieroglyphs", "archaeology"]},
    {"title": "Introduction to computer networks", "url": "https://www.techguide.com/networks", "keywords": ["computer networks", "protocols", "routers", "IP", "TCP"]},
    {"title": "How to knit a scarf", "url": "https://www.knittingworld.com/scarf-guide", "keywords": ["knitting", "scarf", "yarn", "patterns", "needles"]},
    {"title": "Tips for effective public speaking", "url": "https://www.speakup.com/public-speaking", "keywords": ["public speaking", "confidence", "presentation", "speech", "communication"]},
    {"title": "How to start a blog", "url": "https://www.blogging101.com/start-blog", "keywords": ["blog", "writing", "content", "platform", "website"]},
    {"title": "Understanding personal finance", "url": "https://www.financialbasics.com/personal-finance", "keywords": ["finance", "budgeting", "saving", "investment", "money"]},
    {"title": "The basics of machine learning", "url": "https://www.mlworld.com/basics", "keywords": ["machine learning", "AI", "algorithms", "data", "training"]},
    {"title": "How to build a PC", "url": "https://www.techbuilder.com/pc-guide", "keywords": ["PC", "hardware", "build", "components", "installation"]},
    {"title": "Exploring Mars: Current Missions", "url": "https://www.spaceexplorer.com/mars-missions", "keywords": ["Mars", "space", "exploration", "NASA", "rovers"]},
    {"title": "Learning to play the guitar", "url": "https://www.musiclessons.com/guitar", "keywords": ["guitar", "music", "chords", "notes", "practice"]},
    {"title": "How to write a business plan", "url": "https://www.businessworld.com/write-plan", "keywords": ["business", "plan", "strategy", "growth", "entrepreneur"]},
    {"title": "Introduction to coding for kids", "url": "https://www.kidscodinghub.com/start", "keywords": ["coding", "kids", "education", "fun", "Python"]},
    {"title": "Home fitness: Workout routines", "url": "https://www.fitnesshome.com/routines", "keywords": ["fitness", "workout", "home", "exercise", "strength"]},
    {"title": "Travel essentials for backpackers", "url": "https://www.backpackersguide.com/essentials", "keywords": ["travel", "backpacking", "gear", "adventure", "checklist"]},
    {"title": "Understanding basic algebra", "url": "https://www.mathhelp.com/algebra", "keywords": ["math", "algebra", "equations", "unknowns", "variables"]},
    {"title": "Tips for building muscle", "url": "https://www.bodybuildworld.com/tips", "keywords": ["muscle", "growth", "exercise", "protein", "fitness"]},
    {"title": "How to cook vegan meals", "url": "https://www.veganrecipes.com/cook", "keywords": ["vegan", "cooking", "meals", "healthy", "diet"]},
    {"title": "The history of ancient Greece", "url": "https://www.historyportal.com/greece", "keywords": ["Greece", "history", "civilization", "mythology", "ancient"]},
    {"title": "How to manage project deadlines", "url": "https://www.projectmaster.com/deadlines", "keywords": ["project management", "deadlines", "timing", "tasks", "planning"]},
    {"title": "How to start a YouTube gaming channel", "url": "https://www.gaminghub.com/start-channel", "keywords": ["YouTube", "gaming", "content creation", "views", "streaming"]},
    {"title": "Exploring coral reefs", "url": "https://www.marineworld.com/coral-reefs", "keywords": ["coral reefs", "ocean", "marine life", "diving", "conservation"]},
    {"title": "Introduction to astronomy", "url": "https://www.spaceguide.com/astronomy", "keywords": ["astronomy", "stars", "planets", "telescopes", "cosmos"]},
    {"title": "Tips for learning Japanese", "url": "https://www.languagehub.com/japanese", "keywords": ["Japanese", "language", "learning", "kanji", "grammar"]},
    {"title": "How to start a podcast", "url": "https://www.podcasterpro.com/start", "keywords": ["podcast", "audio", "microphone", "content", "episodes"]},
    {"title": "Introduction to psychology", "url": "https://www.psychology101.com/intro", "keywords": ["psychology", "mind", "behavior", "cognition", "emotion"]},
    {"title": "How to bake sourdough pizza", "url": "https://www.pizzalovers.com/sourdough", "keywords": ["pizza", "sourdough", "recipe", "oven", "crust"]},
    {"title": "Exploring volcanoes worldwide", "url": "https://www.geologyworld.com/volcanoes", "keywords": ["volcanoes", "eruption", "lava", "tectonics", "earth"]},
    {"title": "How to build a bookshelf", "url": "https://www.diyprojects.com/bookshelf", "keywords": ["bookshelf", "wood", "DIY", "tools", "crafting"]},
    {"title": "Tips for faster typing", "url": "https://www.typemaster.com/speed", "keywords": ["typing", "speed", "keyboard", "productivity", "practice"]},
    {"title": "Understanding insurance policies", "url": "https://www.insurancehelp.com/policies", "keywords": ["insurance", "policy", "coverage", "premiums", "protection"]},
    {"title": "How to grow succulents indoors", "url": "https://www.plantlife.com/succulents", "keywords": ["succulents", "indoor plants", "water", "sunlight", "soil"]},
    {"title": "Learning graphic design basics", "url": "https://www.designhub.com/basics", "keywords": ["graphic design", "color theory", "illustration", "tools", "creativity"]},
    {"title": "How to master Excel formulas", "url": "https://www.spreadsheetpro.com/formulas", "keywords": ["Excel", "formulas", "spreadsheets", "data", "functions"]},
    {"title": "Travel guide to Iceland", "url": "https://www.travelworld.com/iceland", "keywords": ["Iceland", "travel", "landscape", "geysers", "northern lights"]},
    {"title": "Introduction to biohacking", "url": "https://www.healthtech.com/biohacking", "keywords": ["biohacking", "health", "performance", "supplements", "technology"]},
    {"title": "Understanding electric cars", "url": "https://www.evnews.com/intro", "keywords": ["electric cars", "EVs", "batteries", "charging", "sustainability"]},
    {"title": "How to plan a wedding", "url": "https://www.weddingworld.com/plan", "keywords": ["wedding", "planning", "event", "ceremony", "celebration"]},
    {"title": "Exploring the Amazon rainforest", "url": "https://www.naturelife.com/amazon", "keywords": ["Amazon", "rainforest", "wildlife", "ecology", "climate"]},
    {"title": "How to build a drone", "url": "https://www.droneworks.com/build", "keywords": ["drone", "aerial", "assembly", "technology", "components"]},
    {"title": "The history of cinema", "url": "https://www.filmguide.com/history", "keywords": ["cinema", "film", "movies", "directors", "Hollywood"]},
    {"title": "How to write a screenplay", "url": "https://www.screenwriters.com/write", "keywords": ["screenplay", "film", "story", "characters", "dialogue"]},
    {"title": "Exploring space telescopes", "url": "https://www.spaceview.com/telescopes", "keywords": ["space telescopes", "Hubble", "observation", "astronomy", "cosmos"]},
    {"title": "The science of memory", "url": "https://www.mindscience.com/memory", "keywords": ["memory", "brain", "cognition", "recall", "neuroscience"]},
    {"title": "How to knit socks", "url": "https://www.knitting101.com/socks", "keywords": ["knitting", "socks", "patterns", "yarn", "handmade"]},
    {"title": "How to build a chatbot", "url": "https://www.aiprojects.com/chatbot", "keywords": ["chatbot", "AI", "NLP", "development", "automation"]},
    {"title": "Learning Korean from scratch", "url": "https://www.languagejourney.com/korean", "keywords": ["Korean", "language", "hangul", "grammar", "conversation"]},
    {"title": "Introduction to philosophy", "url": "https://www.philosophyworld.com/intro", "keywords": ["philosophy", "ethics", "logic", "existence", "knowledge"]},
    {"title": "How to make sushi at home", "url": "https://www.sushichef.com/home", "keywords": ["sushi", "Japanese food", "fish", "rolls", "rice"]},
    {"title": "The fundamentals of programming", "url": "https://www.codeworld.com/fundamentals", "keywords": ["programming", "coding", "Python", "JavaScript", "syntax"]},
    {"title": "Understanding sleep cycles", "url": "https://www.healthystates.com/sleep", "keywords": ["sleep", "REM", "circadian rhythm", "rest", "health"]},
    {"title": "How to build a rocket", "url": "https://www.rocketryclub.com/build", "keywords": ["rocket", "space", "fuel", "thrust", "design"]},
]

# Irrelevant content samples
irrelevant_sentences = [
    "You need to enable JavaScript to view this content.",
    "Please accept cookies to continue.",
    "Subscribe to access premium content.",
    "This site uses cookies for analytics and personalization.",
    "Advertisement: Get 20% off today!",
    "Error: Page could not be loaded.",
    "Sign in to view this content.",
    "This content is protected by copyright.",
    "Your session has expired.",
    "Click here to see more articles.",
    "Content requires an active subscription.",
    "Please disable your ad blocker to proceed.",
    "We value your privacy; review our cookie settings.",
    "Please accept cookies to continue.",
    "This site uses cookies for better performance.",
    "Sign up to read more!",
    "Advertisement: Buy our product now!",
    "Error: Page not found.",
    "Loading content... please wait.",
    "Subscribe to our newsletter!",
    "This website uses cookies to enhance your experience.",
    "Your session has expired. Please refresh.",
    "Click here to accept marketing preferences.",
    "Ad banner: 50% off for today only!",
    "This content is protected by copyright.",
    "Please disable your adblocker to view this page.",
    "Access restricted. Login required.",
    "This article is behind a paywall.",
    "Contact support if the problem persists.",
    "Your privacy is important to us.",
    "End of article.",
    "Click here to see related articles.",
       "JavaScript must be enabled to view this site.",
    "Content loading... please wait.",
    "This page requires Flash Player.",
    "Access denied. Insufficient permissions.",
    "Please verify you're not a robot.",
    "Page under maintenance.",
    "Limited-time offer: Sign up now!",
    "This content is protected by reCAPTCHA.",
    "Advertisement: Save 30% on laptops!",
    "Data not found.",
    "Subscribe to view more articles.",
    "404 - Page not found.",
    "Your connection is not private.",
    "Sign in to continue reading.",
    "Promo: Get a free month now!",
    "This section is unavailable.",
    "Click here to upgrade your account.",
    "Please accept our privacy policy.",
    "Free trial ended. Please subscribe.",
    "System error. Please retry later.",
    "No results found.",
    "This page was last updated in 1999.",
    "Session timeout. Re-login required.",
    "Enjoy ad-free browsing - subscribe today!",
    "Enable cookies for full functionality.",
    "Video player requires update.",
    "Your download is starting...",
    "Please provide feedback.",
    "Unsupported browser detected.",
    "Limited content for free accounts.",
    "Server overloaded. Try again later.",
    "You must be an admin to access this.",
    "Your preferences were not saved.",
    "Premium content for members only.",
    "Survey: How did you like the content?",
    "Password change required.",
    "Network issues detected.",
    "Device not supported.",
    "Invalid request. Please refresh.",
    "Searching for more results...",
    "Authentication required.",
    "This content is restricted.",
    "Bandwidth limit exceeded.",
    "Ads blocked. Please whitelist us.",
    "No comments yet. Be the first!",
    "Invalid session token.",
    "Slow network connection detected.",
    "Please wait while the content loads.",
    "Unsupported media format.",
    "Captcha verification failed.",
]

def generate_sentences(prompt, min_sentences=5, max_sentences=10):
    """Generate a random number of sentences between min_sentences and max_sentences."""
    try:
        # Determine the number of sentences to generate
        num_sentences = random.randint(min_sentences, max_sentences)

        # Encode input and create attention mask
        input_ids = tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True)
        attention_mask = torch.ones_like(input_ids)

        # Generate text
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=150,  # Increased max length to allow more sentences
            num_return_sequences=1,
            temperature=0.7,  # Slightly increased for more variability
            top_k=50,
            top_p=0.92,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

        # Decode and split into sentences
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        sentences = re.split('[.!?]', text)
        sentences = [s.strip() for s in sentences if len(s.split()) > 5][:num_sentences]

        return sentences

    except Exception as e:
        print(f"Error during generation: {e}")
        return []



def paraphrase_sentences(sentences, num_variants=2):
    paraphrased = []
    for sentence in sentences:
        # Generate variations by adding keywords or reordering phrases
        variations = [f"{sentence} - explained in simpler terms",
                      f"Understanding {sentence}",
                      f"Detailed insights about {sentence}"]
        paraphrased.extend(variations)
    return list(set(paraphrased))

# Generate dataset
data = []

for source in data_sources:
    # Generate relevant sentences
    prompt = f"Generate sentences about {source['title']} with keywords: {', '.join(source['keywords'])}."
    generated_sentences = generate_sentences(prompt)
    paraphrased = paraphrase_sentences(generated_sentences)

    for sentence in paraphrased:
        data.append({
            "title": source['title'],
            "url": source['url'],
            "sentence": sentence,
            "label": 1
        })

# Add irrelevant noise
for source in data_sources:
    for _ in range(5):
        sentence = random.choice(irrelevant_sentences)
        data.append({
            "title": source['title'],
            "url": source['url'],
            "sentence": sentence,
            "label": 0
        })

# Shuffle dataset
random.shuffle(data)

# Save dataset
df = pd.DataFrame(data)
df.to_csv("labeled_dataset_large.csv", index=False)

print(f"Labeled dataset created with {len(df)} rows.")
