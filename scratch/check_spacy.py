try:
    import spacy
    print("spacy is installed")
    try:
        nlp = spacy.load("en_core_web_sm")
        print("en_core_web_sm is loaded")
    except Exception as e:
        print(f"model load failed: {e}")
except ImportError:
    print("spacy not found")
