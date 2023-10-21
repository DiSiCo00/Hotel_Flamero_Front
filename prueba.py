from transformers import pipeline
import joblib

categorias = ["Limpieza", "Confort", "Ubicación", "Instalaciones", "Personal"]


cls_zero = joblib.load("cero_shot_classifier.pkl")
print(cls_zero("suciedad", categorias))
# classifier = pipeline("zero-shot-classification",
#                     model="facebook/bart-large-mnli",
#                     revision = "c626438")



#     classifier = pipeline("zero-shot-classification")
#                         # model="facebook/bart-large-mnli",
#                         # revision = "c626438")
# joblib.dump(classifier, "cero_shot_classifier.pkl")

#     resultados = classifier(_text, categorias)

#     with open("_Data/obj.json") as file:
#         obj = json.load(file)
#         file.close()
#     obj[resultados['labels'][0]]["len"] = obj[resultados['labels'][0]]["len"]+1
#     obj[resultados['labels'][0]]["Score"] = (obj[resultados['labels'][0]]["Score"]*obj[resultados['labels'][0]]["len"] + score)/obj[resultados['labels'][0]]["len"]

#     obj['General']["len"] = obj['General']["len"] + 1
#     obj['General']["Score"] = (obj['General']["Score"] * obj['General']["len"] + score)/obj['General']["len"]

