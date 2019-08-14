from .agreementr import Agreementr

ar = None

def predict(text):
    global ar
    if ar is None:
        try:
            ar = Agreementr()
        except Exception as e:
            print("Failed to initialize Agreementr")
            print(e)

    return ar.predict(text)