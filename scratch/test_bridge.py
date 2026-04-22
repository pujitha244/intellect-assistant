import sys
other_project_packages = r"C:\Users\pooji\Downloads\NLP\nlp_env\Lib\site-packages"
if other_project_packages not in sys.path:
    sys.path.append(other_project_packages)

try:
    import spacy
    print("spacy import successful")
    nlp = spacy.load("en_core_web_sm")
    print("en_core_web_sm loaded successfully from other project")
except Exception as e:
    print(f"Failed: {e}")

try:
    from fpdf import FPDF
    print("fpdf import successful")
except Exception as e:
    print(f"fpdf failed: {e}")
