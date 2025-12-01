from model.core import model_handler


data = "My mother died"
prediction = model_handler.predict(x=data)
print(prediction)
